import tensorflow as tf
import numpy as np
import math

def tanh_squash(mu, log_std, low, high, eps=1e-6):
    """高斯采样+Tanh压缩到区间 [low, high]，返回 sample, logp"""
    std = tf.exp(log_std)
    z = mu + std * tf.random.normal(tf.shape(mu))
    y = tf.tanh(z)
    # 反映射到区间
    out = low + (y + 1.0) * 0.5 * (high - low)
    # logp 修正（tanh 变换的雅可比）
    logp = -0.5 * ((z - mu) / (std + eps))**2 - log_std - 0.5 * math.log(2 * math.pi)
    logp = tf.reduce_sum(logp, axis=-1)
    logp -= tf.reduce_sum(tf.math.log(1 - tf.square(y) + eps), axis=-1)
    return out, logp


# Actor Network (LSTM + PPO) with time-series input
class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, lstm_size=64, use_time_series=True, time_series_len=16,
                 offset_range=(0.0, 8.0), duration_range=(8.0, 40.0)):
        super().__init__()
        self.use_time_series = use_time_series
        self.time_series_len = time_series_len
        self.offset_range = offset_range
        self.duration_range = duration_range

        if use_time_series:
            # Time-series LSTM branch
            self.ts_lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=False, return_state=True)
            # Flatten current obs and combine with LSTM output
            self.fc_combine = tf.keras.layers.Dense(128, activation="tanh")
            # Policy head
            self.logits_layer = tf.keras.layers.Dense(act_dim)
        else:
            # Original architecture
            self.fc1 = tf.keras.layers.Dense(64, activation="tanh")
            self.lstm = tf.keras.layers.LSTM(lstm_size, return_state=True, return_sequences=False)
            self.logits_layer = tf.keras.layers.Dense(act_dim)
        # coop head
        self.coop_head = tf.keras.layers.Dense(64, activation="tanh")
        self.offset_mu = tf.keras.layers.Dense(1)
        self.offset_log_std = self.add_weight("offset_log_std", shape=(1,), initializer="zeros")
        self.dur_mu = tf.keras.layers.Dense(1)
        self.dur_log_std = self.add_weight("dur_log_std", shape=(1,), initializer="zeros")
        
        self.lstm_size = lstm_size
        self.act_dim = act_dim

    def call(self, obs, states=None, time_series=None):
        """
        obs: (batch, obs_dim) - current observation
        states: tuple (h, c) - LSTM hidden states (for compatibility)
        time_series: (batch, time_series_len, obs_dim) - time-series data

        returns:
            logits: (batch, act_dim) - action logits
            probs: (batch, act_dim) - action probabilities
            next_states: tuple (next_h, next_c) - next LSTM hidden states
        """
        if self.use_time_series and time_series is not None:
            if states is not None:
                ts_output, next_h, next_c = self.ts_lstm(time_series, initial_state=states)
            else:
                ts_output, next_h, next_c = self.ts_lstm(time_series)
            combined = tf.concat([obs if obs is not None else time_series[:, -1, :], ts_output], axis=1)
            x = self.fc_combine(combined)
            # logits = self.logits_layer(x) * (0.5 + alpha)
            logits = self.logits_layer(x)
            probs = tf.nn.softmax(logits)
            h_coop = self.coop_head(x)
            return logits, probs, h_coop, (next_h, next_c)
        else:
            x = self.fc1(obs)
            h_coop = self.coop_head(x)
            x = tf.expand_dims(x, axis=1)
            output, h, c = self.lstm(x, initial_state=states)
            logits = self.logits_layer(output)
            probs = tf.nn.softmax(logits)
            return logits, probs, h_coop, (h, c)

    def coop_params(self, h_coop):
        offset_mu = self.offset_mu(h_coop)
        offset_log_std = tf.broadcast_to(self.offset_log_std, tf.shape(offset_mu))
        dur_mu = self.dur_mu(h_coop)
        dur_log_std = tf.broadcast_to(self.dur_log_std, tf.shape(dur_mu))
        return offset_mu, offset_log_std, dur_mu, dur_log_std


# Centralized Critic
class Critic(tf.keras.Model):
    def __init__(self, gobs_dim, l2_reg=1e-4):
        super().__init__()
        self.l2_reg = l2_reg
        self.fc1 = tf.keras.layers.Dense(128, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.fc2 = tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.v_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))


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
                 use_time_series=True, time_series_len=16,
                 max_grad_norm=0.5, value_loss_coef=1.0,
                 offset_range=(0.0, 8.0), duration_range=(8.0, 40.0)):

        self.actor = Actor(obs_dim, act_dim, lstm_size,
                          use_time_series=use_time_series,
                          time_series_len=time_series_len,
                          offset_range=offset_range,
                          duration_range=duration_range)
        self.critic = Critic(gobs_dim)

        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(v_lr)

        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.act_dim = act_dim
        self.use_time_series = use_time_series
        self.time_series_len = time_series_len
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef

        # coop
        self.offset_range = offset_range
        self.duration_range = duration_range

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
            logits, probs, h_coop, next_state = self.actor(obs, states=rnn_state, time_series=time_series)
        else:
            logits, probs, h_coop, next_state = self.actor(obs, states=rnn_state)
        
        probs = probs.numpy()[0]
        
        # discrete action sampling
        action = np.random.choice(self.act_dim, p=probs)
        logp_dis = np.log(probs[action] + 1e-8)

        # continuous parameter sampling
        offset_mu, offset_log_std, dur_mu, dur_log_std = self.actor.coop_params(h_coop)
        offset, offset_logp = tanh_squash(offset_mu, offset_log_std, *self.offset_range)
        duration, dur_logp = tanh_squash(dur_mu, dur_log_std, *self.duration_range)

        coop_params_info = {
            "offset": float(offset.numpy()[0, 0]),
            "duration": float(duration.numpy()[0, 0])
        }
        logp_total = logp_dis + offset_logp.numpy()[0] + dur_logp.numpy()[0]

        return action, logp_total, probs, coop_params_info, next_state

    @tf.function
    def train_actor_step(self, obs, act_dis, adv, old_logp,
                         coop_offset, coop_duration,
                         time_series=None):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act_dis = tf.convert_to_tensor(act_dis, dtype=tf.int32)
        adv = tf.convert_to_tensor(adv, dtype=tf.float32)
        old_logp = tf.convert_to_tensor(old_logp, dtype=tf.float32)
        if time_series is not None:
            time_series = tf.convert_to_tensor(time_series, dtype=tf.float32)
            
        with tf.GradientTape() as tape:
            if self.use_time_series and time_series is not None:
                logits, probs, h_coop, _ = self.actor(obs, time_series=time_series)
            else:
                batch = tf.shape(obs)[0]
                h0 = tf.zeros((batch, self.actor.lstm_size))
                c0 = tf.zeros((batch, self.actor.lstm_size))
                logits, probs, h_coop, _ = self.actor(obs, (h0, c0))
            
            # discrete action logp
            logp_all = tf.nn.log_softmax(logits)
            act_onehot = tf.one_hot(act_dis, depth=self.act_dim)
            logp_dis = tf.reduce_sum(act_onehot * logp_all, axis=1)

            # continuous coop params logp
            offset_mu, offset_log_std, dur_mu, dur_log_std = self.actor.coop_params(h_coop)
            def inv_tanh(x, eps=1e-6):
                x = tf.clip_by_value(x, -1 + eps, 1 - eps)
                return 0.5 * tf.math.log((1 + x) / (1 - x))
            def to_y(x, low, high):
                return 2.0 * (x - low) / (high - low) - 1.0

            y_off = to_y(coop_offset, *self.offset_range)
            z_off = inv_tanh(y_off)
            std_off = tf.exp(offset_log_std)
            logp_off = -0.5 * ((z_off - offset_mu) / std_off)**2 - offset_log_std - 0.5 * math.log(2 * math.pi)
            logp_off = tf.squeeze(logp_off, axis=-1) - tf.math.log(1 - tf.square(y_off) + 1e-6)

            y_dur = to_y(coop_duration, *self.duration_range)
            z_dur = inv_tanh(y_dur)
            std_dur = tf.exp(dur_log_std)
            logp_dur = -0.5 * ((z_dur - dur_mu) / std_dur)**2 - dur_log_std - 0.5 * math.log(2 * math.pi)
            logp_dur = tf.squeeze(logp_dur, axis=-1) - tf.math.log(1 - tf.square(y_dur) + 1e-6)
            
            logp = logp_dis + logp_off + logp_dur
            ratio = tf.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio,
                                     1.0-self.clip_ratio,
                                     1.0+self.clip_ratio) * adv
            pi_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=1))
            loss = pi_loss - self.ent_coef * entropy

        grads = tape.gradient(loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.pi_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return pi_loss, entropy

    @tf.function
    def train_critic_step(self, gobs, returns):
        with tf.GradientTape() as tape:
            v = self.critic(gobs)
            loss = tf.reduce_mean(tf.square(v - returns))
            reg_loss = tf.reduce_sum(self.critic.losses)
            total_loss = loss + self.value_loss_coef * reg_loss
        grads = tape.gradient(total_loss, self.critic.trainable_variables)
        self.v_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return total_loss

    def update(self, obs, act, adv, old_logp, gobs, returns, coop_offset, coop_duration,
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
                coop_off_b = coop_offset[mb]
                coop_dur_b = coop_duration[mb]

                ts_b = None
                if self.use_time_series and time_series is not None:
                    ts_b = time_series[mb]

                self.train_actor_step(obs_b, act_b, adv_b, old_logp_b, time_series=ts_b, coop_offset=coop_off_b, coop_duration=coop_dur_b)
                if update_critic:
                    self.train_critic_step(gobs_b, returns_b)