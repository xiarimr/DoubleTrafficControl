import tensorflow as tf
import numpy as np


# Actor Network (LSTM + PPO) with time-series input
class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, lstm_size=64, use_time_series=True, time_series_len=16):
        super().__init__()
        self.use_time_series = use_time_series
        self.time_series_len = time_series_len
        
        if use_time_series:
            # Time-series LSTM branch
            self.ts_lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=False, return_state=True)
            # Flatten current obs and combine with LSTM output
            self.fc_combine = tf.keras.layers.Dense(128, activation="relu")
            # Policy head
            self.logits_layer = tf.keras.layers.Dense(act_dim)
        else:
            # Original architecture
            self.fc1 = tf.keras.layers.Dense(64, activation="relu")
            self.lstm = tf.keras.layers.LSTM(lstm_size, return_state=True, return_sequences=False)
            self.logits_layer = tf.keras.layers.Dense(act_dim)
        
        self.lstm_size = lstm_size
        self.act_dim = act_dim

    def call(self, obs, states=None, time_series=None):
        """
        obs: (batch, obs_dim) - current observation
        states: tuple (h, c) - LSTM hidden states (for compatibility)
        time_series: (batch, time_series_len, obs_dim) - time-series data
        """
        if self.use_time_series and time_series is not None:
            # Process time-series through LSTM
            ts_output, next_h, next_c = self.ts_lstm(time_series)  # (batch, lstm_size)
            # 如果提供了当前观测，则拼接
            if obs is not None:
                combined = tf.concat([obs, ts_output], axis=1)  # (batch, obs_dim + lstm_size)
            else:
                # 否则使用时间序列的最后一个时间步
                last_obs = time_series[:, -1, :]  # (batch, obs_dim)
                combined = tf.concat([last_obs, ts_output], axis=1)
            x = self.fc_combine(combined)  # (batch, 128)
            logits = self.logits_layer(x)  # (batch, act_dim)
            probs = tf.nn.softmax(logits)
            return logits, probs, (next_h, next_c)
        else:
            # Original path (without time-series)
            x = self.fc1(obs)
            x = tf.expand_dims(x, axis=1)
            output, h, c = self.lstm(x, initial_state=states)
            logits = self.logits_layer(output)
            probs = tf.nn.softmax(logits)
            return logits, probs, (h, c)


# Centralized Critic
class Critic(tf.keras.Model):
    def __init__(self, gobs_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.v_layer = tf.keras.layers.Dense(1)


    def call(self, gobs):
        x = self.fc1(gobs)
        x = self.fc2(x)
        v = self.v_layer(x)
        return tf.squeeze(v, axis=-1)  # shape = (batch,)


# MAPPO Agent with time-series support
class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, gobs_dim,
                 lstm_size=64, pi_lr=3e-4, v_lr=1e-3,
                 clip_ratio=0.2, ent_coef=0.01, 
                 use_time_series=True, time_series_len=16):

        self.actor = Actor(obs_dim, act_dim, lstm_size, 
                          use_time_series=use_time_series, 
                          time_series_len=time_series_len)
        self.critic = Critic(gobs_dim)

        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(v_lr)

        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.act_dim = act_dim
        self.use_time_series = use_time_series
        self.time_series_len = time_series_len

    def act(self, obs, rnn_state, time_series=None):
        """
        Select action
        
        Args:
            obs: (obs_dim,) - current observation
            rnn_state: tuple (h, c) - LSTM hidden states
            time_series: (time_series_len, obs_dim) - time-series sequence
            
        Returns:
            action, logp, probs, next_state
        """
        obs = obs.reshape(1, -1).astype(np.float32)

        if time_series is not None and len(time_series.shape) == 2:
            time_series = np.expand_dims(time_series, axis=0)  # (1, time_series_len, obs_dim)
        
        if self.use_time_series and time_series is not None:
            logits, probs, next_state = self.actor(obs, time_series=time_series)
        else:
            logits, probs, next_state = self.actor(obs, states=rnn_state)
        
        probs = probs.numpy()[0]
        action = np.random.choice(self.act_dim, p=probs)
        logp = np.log(probs[action] + 1e-8)
        return action, logp, probs, next_state

    @tf.function
    def train_actor_step(self, obs, act, adv, old_logp, time_series=None):
        with tf.GradientTape() as tape:
            if self.use_time_series and time_series is not None:
                logits, probs, _ = self.actor(obs, time_series=time_series)
            else:
                batch = tf.shape(obs)[0]
                h0 = tf.zeros((batch, self.actor.lstm_size))
                c0 = tf.zeros((batch, self.actor.lstm_size))
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

    @tf.function
    def train_critic_step(self, gobs, returns):
        with tf.GradientTape() as tape:
            v = self.critic(gobs)
            loss = tf.reduce_mean(tf.square(v - returns))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.v_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    def update(self, obs, act, adv, old_logp, gobs, returns, 
               batch_size=64, epochs=4, update_critic=True, time_series=None):
        n = len(time_series) if (self.use_time_series and time_series is not None) else len(obs)
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
                
                ts_b = None
                if self.use_time_series and time_series is not None:
                    ts_b = time_series[mb]

                self.train_actor_step(obs_b, act_b, adv_b, old_logp_b, time_series=ts_b)
                if update_critic:
                    self.train_critic_step(gobs_b, returns_b)