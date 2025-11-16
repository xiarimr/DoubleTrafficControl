import os
import numpy as np
import tensorflow as tf

from agents.mappo_agent import MAPPOAgent
from env.sumo_env import SumoEnvTwoAgents

# ----------------- utility: GAE -----------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: np.array shape (T,)
    values: np.array shape (T+1,)  # last bootstrap value appended already
    returns: np.array shape (T,)
    advs: np.array shape (T,)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns

# ----------------- main training -----------------
def train(
    SUMO_CFG="small_net/exp.sumocfg",
    SUMO_BIN="sumo",            # or "sumo-gui" for debugging
    total_epochs=10000,
    batch_steps=2048,
    gamma=0.99,
    lam=0.95,
    ppo_epochs=6,
    mini_batch=64,
    pi_lr=3e-4,
    v_lr=1e-3,
    lstm_size=64,
    ent_coef=0.01,
    clip_ratio=0.2,
    model_dir="./models_tf2",
    save_every=20
):
    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:", gpus)
        try:
            # allow memory growth
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU found, using CPU.")

    os.makedirs(model_dir, exist_ok=True)

    # Create env
    env = SumoEnvTwoAgents(sumocfg_path=SUMO_CFG, sumo_bin=SUMO_BIN,
                           delta_time=4.0, switch_penalty=1.0)
    AGENTS = ["A", "B"]
    obs_dim = 5
    act_dim = 6
    gobs_dim = obs_dim * 2

    # instantiate a single critic model to be shared
    central_agent = MAPPOAgent(obs_dim, act_dim, gobs_dim,
                               lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
                               clip_ratio=clip_ratio, ent_coef=ent_coef)
    agent_A = MAPPOAgent(obs_dim, act_dim, gobs_dim,
                         lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
                         clip_ratio=clip_ratio, ent_coef=ent_coef)
    agent_B = MAPPOAgent(obs_dim, act_dim, gobs_dim,
                         lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
                         clip_ratio=clip_ratio, ent_coef=ent_coef)

    # Replace both critics by a single centralized critic so critic parameters are shared
    shared_critic = central_agent.critic
    agent_A.critic = shared_critic
    agent_B.critic = shared_critic

    # Also ensure their v_optimizer is the same optimizer instance so updates won't conflict (optional)
    agent_A.v_optimizer = central_agent.v_optimizer
    agent_B.v_optimizer = central_agent.v_optimizer

    # Checkpointing
    ckpt = tf.train.Checkpoint(agentA_actor=agent_A.actor,
                               agentB_actor=agent_B.actor,
                               critic=shared_critic,
                               pi_opt_A=agent_A.pi_optimizer,
                               pi_opt_B=agent_B.pi_optimizer,
                               v_opt=central_agent.v_optimizer)  # optional
    manager = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=5)

    # training loop
    for ep in range(total_epochs):
        # buffers: per-agent
        obs_buf = {aid: [] for aid in AGENTS}
        act_buf = {aid: [] for aid in AGENTS}
        oldlogp_buf = {aid: [] for aid in AGENTS}
        rew_buf = {aid: [] for aid in AGENTS}
        probs_buf = {aid: [] for aid in AGENTS}
        gobs_buf = []

        # rnn states per agent (h,c) initial zeros
        rnn_state = {
            "A": (tf.zeros((1, lstm_size)), tf.zeros((1, lstm_size))),
            "B": (tf.zeros((1, lstm_size)), tf.zeros((1, lstm_size)))
        }

        o = env.reset()
        steps = 0
        while steps < batch_steps:
            # choose actions
            acts = {}
            probs_map = {}
            old_logps = {}
            # agent A
            a_A, logp_A, probs_A, next_state_A = agent_A.act(o["A"], rnn_state["A"])
            acts["A"] = int(a_A); old_logps["A"] = float(logp_A); probs_map["A"] = probs_A
            rnn_state["A"] = next_state_A
            # agent B
            a_B, logp_B, probs_B, next_state_B = agent_B.act(o["B"], rnn_state["B"])
            acts["B"] = int(a_B); old_logps["B"] = float(logp_B); probs_map["B"] = probs_B
            rnn_state["B"] = next_state_B

            # step env; pass neighbor probs for entropy fingerprint
            o2, r, done, info = env.step(acts, neighbor_policy_probs=probs_map)

            # store data
            gobs_buf.append(np.concatenate([o["A"], o["B"]], axis=0).astype(np.float32))
            for aid in AGENTS:
                obs_buf[aid].append(o[aid].astype(np.float32))
                act_buf[aid].append(acts[aid])
                oldlogp_buf[aid].append(old_logps[aid])
                rew_buf[aid].append(r[aid])
                probs_buf[aid].append(probs_map[aid])
            o = o2
            steps += 1
            # Note: done is always False in current implementation

        # convert to arrays
        for aid in AGENTS:
            obs_buf[aid] = np.vstack(obs_buf[aid]).astype(np.float32)
            act_buf[aid] = np.array(act_buf[aid], dtype=np.int32)
            oldlogp_buf[aid] = np.array(oldlogp_buf[aid], dtype=np.float32)
            rew_buf[aid] = np.array(rew_buf[aid], dtype=np.float32)

        gobs = np.vstack(gobs_buf).astype(np.float32)

        # Critic value estimates v_t for t=0..T-1
        vals = shared_critic(gobs).numpy()  # shape (T,)
        # bootstrap last value (for last state after last step)
        last_gobs = gobs[-1:]
        last_val = shared_critic(last_gobs).numpy()[0]
        vals_full = np.append(vals, last_val)  # shape (T+1,)

        # use summed rewards as global reward (MAPPO common choice)
        rewards_sum = rew_buf["A"] + rew_buf["B"]  # shape (T,)
        advs, returns = compute_gae(rewards_sum, vals_full, gamma=gamma, lam=lam)
        # normalize advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Prepare global returns for critic training (shape (T,))
        returns = returns.astype(np.float32)

        # Update actors and shared critic
        # Note: each actor uses its own obs/action/oldlogp but same advs (from centralized critic)
        # Critic is updated in each agent.update() call, but since they share the same critic, it's updated twice
        # We only need to update it once, so we'll do it separately
        agent_A.update(obs_buf["A"], act_buf["A"], advs, oldlogp_buf["A"],
                       gobs=gobs, returns=returns, batch_size=mini_batch, epochs=ppo_epochs, update_critic=False)
        agent_B.update(obs_buf["B"], act_buf["B"], advs, oldlogp_buf["B"],
                       gobs=gobs, returns=returns, batch_size=mini_batch, epochs=ppo_epochs, update_critic=False)

        # Update shared critic once
        n = gobs.shape[0]
        inds = np.arange(n)
        for _ in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, mini_batch):
                mb = inds[start:start+mini_batch]
                gobs_b = gobs[mb]
                returns_b = returns[mb]
                _ = central_agent.train_critic_step(gobs_b, returns_b)

        # logging
        avg_rew = {aid: float(np.mean(rew_buf[aid])) for aid in AGENTS}
        print(f"[EP {ep}] avg_rewards: {avg_rew} peak_flag:{info.get('peak_flag',None)} alpha:{info.get('alpha',None)}")

        # save
        if ep % save_every == 0:
            # save actor and critic weights
            agent_A.actor.save_weights(os.path.join(model_dir, f"actorA_ep{ep}.ckpt"))
            agent_B.actor.save_weights(os.path.join(model_dir, f"actorB_ep{ep}.ckpt"))
            shared_critic.save_weights(os.path.join(model_dir, f"critic_ep{ep}.ckpt"))
            print("Saved checkpoints at ep", ep)

    # final save
    agent_A.actor.save_weights(os.path.join(model_dir, "actorA_final.ckpt"))
    agent_B.actor.save_weights(os.path.join(model_dir, "actorB_final.ckpt"))
    shared_critic.save_weights(os.path.join(model_dir, "critic_final.ckpt"))
    print("Training finished, models saved.")


if __name__ == "__main__":
    train(SUMO_CFG="small_net/exp.sumocfg", SUMO_BIN="sumo", total_epochs=2000, batch_steps=1024)