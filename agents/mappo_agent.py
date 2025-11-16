import tensorflow as tf
import numpy as np


#  Actor Network (LSTM + PPO)
class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, lstm_size=64):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_state=True, return_sequences=False)
        self.logits_layer = tf.keras.layers.Dense(act_dim)

    @tf.function
    def call(self, obs, states):
        """
        obs: (batch, obs_dim)
        states: tuple (h, c)
        """
        x = self.fc1(obs)
        x = tf.expand_dims(x, axis=1)  # LSTM time dimension = 1
        output, h, c = self.lstm(x, initial_state=states)
        logits = self.logits_layer(output)
        probs = tf.nn.softmax(logits)
        return logits, probs, (h, c)


#  Centralized Critic
class Critic(tf.keras.Model):
    def __init__(self, gobs_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.v_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, gobs):
        x = self.fc1(gobs)
        x = self.fc2(x)
        v = self.v_layer(x)
        return tf.squeeze(v, axis=1)  # shape = (batch,)


#  MAPPO Agent
class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, gobs_dim,
                 lstm_size=64, pi_lr=3e-4, v_lr=1e-3,
                 clip_ratio=0.2, ent_coef=0.01):

        self.actor = Actor(obs_dim, act_dim, lstm_size)
        self.critic = Critic(gobs_dim)

        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(v_lr)

        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.act_dim = act_dim

    # Action selection (with LSTM state)
    def act(self, obs, rnn_state):
        obs = obs.reshape(1, -1).astype(np.float32)
        logits, probs, next_state = self.actor(obs, rnn_state)
        probs = probs.numpy()[0]
        action = np.random.choice(self.act_dim, p=probs)
        logp = np.log(probs[action] + 1e-8)
        return action, logp, probs, next_state

    # PPO actor update
    @tf.function
    def train_actor_step(self, obs, act, adv, old_logp):
        with tf.GradientTape() as tape:
            # init state = zero (training不需要LSTM记忆)
            batch = tf.shape(obs)[0]
            h0 = tf.zeros((batch, self.actor.lstm.units))
            c0 = tf.zeros((batch, self.actor.lstm.units))

            logits, probs, _ = self.actor(obs, (h0, c0))
            logp_all = tf.nn.log_softmax(logits)
            act_onehot = tf.one_hot(act, depth=self.act_dim)
            logp = tf.reduce_sum(act_onehot * logp_all, axis=1)

            ratio = tf.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio,
                                     1.0-self.clip_ratio,
                                     1.0+self.clip_ratio) * adv
            pi_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=1))
            loss = pi_loss - self.ent_coef * entropy

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return pi_loss, entropy

    # Critic update
    @tf.function
    def train_critic_step(self, gobs, returns):
        with tf.GradientTape() as tape:
            v = self.critic(gobs)
            loss = tf.reduce_mean(tf.square(v - returns))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.v_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    # Batch update for PPO
    def update(self, obs, act, adv, old_logp, gobs, returns, batch_size=64, epochs=4, update_critic=True):
        n = len(obs)
        inds = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, n, batch_size):
                mb = inds[start:start+batch_size]
                obs_b = obs[mb]
                act_b = act[mb]
                adv_b = adv[mb]
                old_logp_b = old_logp[mb]
                returns_b = returns[mb]
                gobs_b = gobs[mb]

                self.train_actor_step(obs_b, act_b, adv_b, old_logp_b)
                if update_critic:
                    self.train_critic_step(gobs_b, returns_b)
