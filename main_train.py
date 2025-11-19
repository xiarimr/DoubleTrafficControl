import os
import numpy as np
import tensorflow as tf

from agents.mappo_agent import MAPPOAgent
from env.sumo_env import BatchSumoEnvManager

# GAE computation
def compute_gae(rewards, values, gamma=0.99, lam=0.95, num_envs=4):
    """
    为多个并行环境计算 GAE，数据以交错方式存储
    
    数据存储格式: [env0_t0, env1_t0, env2_t0, env3_t0, env0_t1, env1_t1, ...]
    
    Args:
        rewards: np.array shape (env_steps * num_envs)
        values: np.array shape (env_steps * num_envs + num_envs,)
        gamma: discount factor
        lam: GAE lambda
    
    Returns:
        advs: np.array shape (env_steps * num_envs,)
        returns: np.array shape (env_steps * num_envs,)
    """
    total_steps = len(rewards)
    env_steps = total_steps // num_envs  # 每个环境的步数
    
    advs = np.zeros(total_steps, dtype=np.float32)
    returns = np.zeros(total_steps, dtype=np.float32)
    
    # 为每个环境单独计算 GAE
    for env_idx in range(num_envs):
        lastgaelam = 0.0
        
        # 从后向前计算该环境的 GAE
        for t in reversed(range(env_steps)):
            # 计算该环境在时间 t 的样本索引
            idx = env_idx + t * num_envs  # 交错存储的索引
            
            # 下一个时间步的索引
            if t == env_steps - 1:
                # 最后一步：使用该环境的 bootstrap value
                # bootstrap values 存储在 values 的最后 N 个位置
                next_value = values[total_steps + env_idx]
            else:
                # 非最后一步：使用同一环境下一时间步的 value
                next_idx = env_idx + (t + 1) * num_envs
                next_value = values[next_idx]
            
            # 计算 TD error
            delta = rewards[idx] + gamma * next_value - values[idx]
            
            # 计算 GAE
            lastgaelam = delta + gamma * lam * lastgaelam
            advs[idx] = lastgaelam
        
        # 计算该环境的 returns
        for t in range(env_steps):
            idx = env_idx + t * num_envs
            returns[idx] = advs[idx] + values[idx]
    
    return advs, returns


def train(
    SUMO_CFG="small_net/exp.sumocfg",
    SUMO_BIN="sumo",
    base_port=8813,
    total_epochs=10000,
    env_steps=2048,
    num_envs=4,  # Number of parallel SUMO instances
    gamma=0.99,
    lam=0.95,
    ppo_epochs=6,
    mini_batch=64,
    pi_lr=3e-4,
    v_lr=1e-3,
    lstm_size=64,
    ent_coef=0.01,
    clip_ratio=0.2,
    time_series_len=16,  # LSTM time-series length
    model_dir="./models_tf2",
    save_every=20
):
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:", gpus)
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU found, using CPU.")

    os.makedirs(model_dir, exist_ok=True)

    # Create batch environment manager
    env_manager = BatchSumoEnvManager(
        num_envs=num_envs,
        sumocfg_path=SUMO_CFG,
        sumo_bin=SUMO_BIN,
        base_port=base_port,
        time_series_len=time_series_len
    )

    AGENTS = ["A", "B"]
    obs_dim = 5
    act_dim = 8
    gobs_dim = obs_dim * 2

    # Initialize shared agents
    central_agent = MAPPOAgent(
        obs_dim, act_dim, gobs_dim,
        lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
        clip_ratio=clip_ratio, ent_coef=ent_coef,
        use_time_series=True, time_series_len=time_series_len
    )
    agent_A = MAPPOAgent(
        obs_dim, act_dim, gobs_dim,
        lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
        clip_ratio=clip_ratio, ent_coef=ent_coef,
        use_time_series=True, time_series_len=time_series_len
    )
    agent_B = MAPPOAgent(
        obs_dim, act_dim, gobs_dim,
        lstm_size=lstm_size, pi_lr=pi_lr, v_lr=v_lr,
        clip_ratio=clip_ratio, ent_coef=ent_coef,
        use_time_series=True, time_series_len=time_series_len
    )

    # Share critic
    shared_critic = central_agent.critic
    agent_A.critic = shared_critic
    agent_B.critic = shared_critic
    agent_A.v_optimizer = central_agent.v_optimizer
    agent_B.v_optimizer = central_agent.v_optimizer

    # Training loop
    for ep in range(total_epochs):
        # Initialize buffers for all data collected from all environments
        obs_buf = {aid: [] for aid in AGENTS}
        act_buf = {aid: [] for aid in AGENTS}
        oldlogp_buf = {aid: [] for aid in AGENTS}
        rew_buf = {aid: [] for aid in AGENTS}
        probs_buf = {aid: [] for aid in AGENTS}
        gobs_buf = []
        time_series_buf = {aid: [] for aid in AGENTS}  # Time-series buffers

        # RNN states for each environment and agent
        rnn_states = {}
        for env_idx in range(num_envs):
            rnn_states[env_idx] = {
                "A": (tf.zeros((1, lstm_size)), tf.zeros((1, lstm_size))),
                "B": (tf.zeros((1, lstm_size)), tf.zeros((1, lstm_size)))
            }

        # Reset all environments
        obs_list = env_manager.reset_all(full_restart=(ep == 0))
        
        steps = 0
        
        # Main collection loop: collect batch_steps from all environments combined
        while steps < env_steps:
            # Prepare actions for all environments
            actions_list = []
            neighbor_probs_list = []
            logp_dict_list = []
            
            for env_idx in range(num_envs):
                
                obs = obs_list[env_idx]
                
                # Get time-series for current env
                ts_A = np.array(list(env_manager.time_series_buffers[env_idx]["A"]), dtype=np.float32)
                ts_B = np.array(list(env_manager.time_series_buffers[env_idx]["B"]), dtype=np.float32)
                
                # Select actions with time-series input
                a_A, logp_A, probs_A, next_state_A = agent_A.act(
                    obs["A"], rnn_states[env_idx]["A"], time_series=ts_A
                )
                a_B, logp_B, probs_B, next_state_B = agent_B.act(
                    obs["B"], rnn_states[env_idx]["B"], time_series=ts_B
                )
                
                rnn_states[env_idx]["A"] = next_state_A
                rnn_states[env_idx]["B"] = next_state_B
                
                actions_list.append({"A": a_A, "B": a_B})
                neighbor_probs_list.append({"A": probs_A, "B": probs_B})
                logp_dict_list.append({"A": logp_A, "B": logp_B})
            
            # Step all environments
            obs_list, rewards_list, done_list, info_list, time_series_list = env_manager.step_all(
                actions_list, neighbor_probs_list
            )
            
            # Store data from each environment
            for env_idx, (obs_dict, rewards_dict, ts_dict) in enumerate(
                zip(obs_list, rewards_list, time_series_list)
            ):

                actions = actions_list[env_idx]
                probs_map = neighbor_probs_list[env_idx]
                logp_map = logp_dict_list[env_idx]
                
                # Store global observation
                gobs = np.concatenate([obs_dict["A"], obs_dict["B"]], axis=0).astype(np.float32)
                gobs_buf.append(gobs)
                
                # Store per-agent data
                for aid in AGENTS:
                    obs_buf[aid].append(obs_dict[aid].astype(np.float32))
                    act_buf[aid].append(actions[aid])
                    oldlogp_buf[aid].append(logp_map[aid])
                    rew_buf[aid].append(rewards_dict[aid])
                    probs_buf[aid].append(probs_map[aid])
                    time_series_buf[aid].append(ts_dict[aid])  # Store time-series
                
            steps += 1
                

        # Convert buffers to arrays
        for aid in AGENTS:
            obs_buf[aid] = np.vstack(obs_buf[aid]).astype(np.float32)
            act_buf[aid] = np.array(act_buf[aid], dtype=np.int32)
            oldlogp_buf[aid] = np.array(oldlogp_buf[aid], dtype=np.float32)
            rew_buf[aid] = np.array(rew_buf[aid], dtype=np.float32)
            time_series_buf[aid] = np.array(time_series_buf[aid], dtype=np.float32)  # (steps, time_series_len, obs_dim)

        gobs = np.vstack(gobs_buf).astype(np.float32)

        vals = shared_critic(gobs).numpy()  # shape: (env_steps * num_envs,)
        # Compute advantages and returns
        last_vals_list = []
        for env_idx in range(num_envs):
            obs = obs_list[env_idx]
            gobs_next = np.concatenate([obs["A"], obs["B"]], axis=0).astype(np.float32)
            last_val = shared_critic(gobs_next.reshape(1, -1)).numpy()[0]
            last_vals_list.append(last_val)

        last_vals = np.array(last_vals_list, dtype=np.float32)  # shape: (num_envs,)
        vals_full = np.append(vals, last_vals)  # shape: (env_steps*num_envs + num_envs,)


        # Use summed rewards as global reward
        rewards_sum = rew_buf["A"] + rew_buf["B"]
        advs, returns = compute_gae(rewards_sum, vals_full, gamma=gamma, lam=lam, num_envs=num_envs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        returns = returns.astype(np.float32)

        # Update actors with time-series
        agent_A.update(
            obs_buf["A"], act_buf["A"], advs, oldlogp_buf["A"],
            gobs=gobs, returns=returns, batch_size=mini_batch, 
            epochs=ppo_epochs, update_critic=False, time_series=time_series_buf["A"]
        )
        agent_B.update(
            obs_buf["B"], act_buf["B"], advs, oldlogp_buf["B"],
            gobs=gobs, returns=returns, batch_size=mini_batch,
            epochs=ppo_epochs, update_critic=False, time_series=time_series_buf["B"]
        )

        # Update shared critic
        n = gobs.shape[0]
        inds = np.arange(n)
        for _ in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, mini_batch):
                mb = inds[start:start+mini_batch]
                gobs_b = gobs[mb]
                returns_b = returns[mb]
                _ = central_agent.train_critic_step(gobs_b, returns_b)

        # Logging
        avg_rew = {aid: float(np.mean(rew_buf[aid])) for aid in AGENTS}
        print(f"[EP {ep}] avg_rewards: {avg_rew}")

        # Save checkpoints
        if ep % save_every == 0:
            agent_A.actor.save_weights(os.path.join(model_dir, f"actorA_ep{ep}.ckpt"))
            agent_B.actor.save_weights(os.path.join(model_dir, f"actorB_ep{ep}.ckpt"))
            shared_critic.save_weights(os.path.join(model_dir, f"critic_ep{ep}.ckpt"))
            print("Saved checkpoints at ep", ep)

    # Final save
    agent_A.actor.save_weights(os.path.join(model_dir, "actorA_final.ckpt"))
    agent_B.actor.save_weights(os.path.join(model_dir, "actorB_final.ckpt"))
    shared_critic.save_weights(os.path.join(model_dir, "critic_final.ckpt"))
    print("Training finished, models saved.")
    
    env_manager.close_all()


if __name__ == "__main__":
    train(
        SUMO_CFG="small_net/exp.sumocfg",
        SUMO_BIN="sumo",
        total_epochs=2000,
        env_steps=1024,
        num_envs=4,  # Run 4 SUMO instances in parallel
        time_series_len=16  # LSTM processes 16 time-steps
    )