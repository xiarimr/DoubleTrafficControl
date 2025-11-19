import tensorflow as tf
import numpy as np


# Actor Network (LSTM + PPO) with time-series input
class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, lstm_size=64, use_time_series=True, time_series_len=16):
        super().__init__()
        self.use_time_series = use_time_series
        self.time_series_len = time_series_len
        
        if use_time_series:
            # Time-series LSTM branch with masking
            self.ts_lstm = tf.keras.layers.LSTM(
                lstm_size, 
                return_sequences=False, 
                return_state=True,
                mask_zero=True  # 自动忽略零填充
            )
            # Combine LSTM output with current obs
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

    def call(self, time_series, states=None):
        """
        统一接口：只使用时间序列输入
        
        Args:
            time_series: (batch, time_series_len, obs_dim) - 时间序列数据
            states: tuple (h, c) - LSTM 隐藏状态（仅用于非时间序列模式）
        
        Returns:
            logits: (batch, act_dim)
            probs: (batch, act_dim)
            states: tuple (h, c)
        """
        if self.use_time_series:
            # Process time-series through LSTM
            ts_output, next_h, next_c = self.ts_lstm(time_series)  # (batch, lstm_size)
            
            # 使用时间序列的最后一个时间步作为当前观测
            last_obs = time_series[:, -1, :]  # (batch, obs_dim)
            combined = tf.concat([last_obs, ts_output], axis=1)  # (batch, obs_dim + lstm_size)
            
            x = self.fc_combine(combined)  # (batch, 128)
            logits = self.logits_layer(x)  # (batch, act_dim)
            probs = tf.nn.softmax(logits)
            
            return logits, probs, (next_h, next_c)
        else:
            # Original path (without time-series)
            # time_series shape: (batch, 1, obs_dim)
            obs = tf.squeeze(time_series, axis=1)  # (batch, obs_dim)
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
        """
        Args:
            gobs: (batch, gobs_dim) - 全局观测
        Returns:
            v: (batch,) - 状态价值
        """
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

    def act(self, time_series):
        """
        单个环境选择动作（推理）
        
        Args:
            time_series: (time_series_len, obs_dim) - 单个环境的时间序列
            
        Returns:
            action: int - 选择的动作
            logp: float - 动作的对数概率
            probs: (act_dim,) - 动作概率分布
        """
        # 转换为 (1, time_series_len, obs_dim)
        if len(time_series.shape) == 2:
            time_series = np.expand_dims(time_series, axis=0)
        
        time_series = time_series.astype(np.float32)
        
        # 前向传播
        logits, probs, _ = self.actor(time_series)
        
        # 采样动作
        probs = probs.numpy()[0]
        action = np.random.choice(self.act_dim, p=probs)
        logp = np.log(probs[action] + 1e-8)
        
        return action, logp, probs

    def act_batch(self, time_series_batch):
        """
        批量选择动作（用于多环境并行）
        
        Args:
            time_series_batch: (num_envs, time_series_len, obs_dim) - 多个环境的时间序列
            
        Returns:
            actions: (num_envs,) - 选择的动作
            logps: (num_envs,) - 动作的对数概率
            probs: (num_envs, act_dim) - 动作概率分布
        """
        time_series_batch = time_series_batch.astype(np.float32)
        
        # 前向传播
        logits, probs, _ = self.actor(time_series_batch)
        
        # 批量采样
        probs_np = probs.numpy()
        actions = []
        logps = []
        
        for i in range(len(probs_np)):
            p = probs_np[i]
            action = np.random.choice(self.act_dim, p=p)
            logp = np.log(p[action] + 1e-8)
            actions.append(action)
            logps.append(logp)
        
        return np.array(actions), np.array(logps), probs_np

    @tf.function
    def train_actor_step(self, time_series, act, adv, old_logp):
        """
        Actor 训练步骤
        
        Args:
            time_series: (batch, time_series_len, obs_dim)
            act: (batch,)
            adv: (batch,)
            old_logp: (batch,)
        """
        with tf.GradientTape() as tape:
            logits, probs, _ = self.actor(time_series)
            
            # 计算当前策略的 log_prob
            logp_all = tf.nn.log_softmax(logits)
            act_onehot = tf.one_hot(act, depth=self.act_dim)
            logp = tf.reduce_sum(act_onehot * logp_all, axis=1)

            # PPO clipped loss
            ratio = tf.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio,
                                     1.0 - self.clip_ratio,
                                     1.0 + self.clip_ratio) * adv
            pi_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=1))
            loss = pi_loss - self.ent_coef * entropy

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return pi_loss, entropy

    @tf.function
    def train_critic_step(self, gobs, returns):
        """
        Critic 训练步骤
        
        Args:
            gobs: (batch, gobs_dim)
            returns: (batch,)
        """
        with tf.GradientTape() as tape:
            v = self.critic(gobs)
            loss = tf.reduce_mean(tf.square(v - returns))
        
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.v_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss

    def update(self, time_series, act, adv, old_logp, gobs, returns, 
               batch_size=4, epochs=4, update_critic=True):
        """
        更新 Actor 和 Critic
        
        Args:
            time_series: (n, time_series_len, obs_dim) - 收集的时间序列数据
            act: (n,) - 动作
            adv: (n,) - 优势
            old_logp: (n,) - 旧的对数概率
            gobs: (n, gobs_dim) - 全局观测
            returns: (n,) - 回报
            batch_size: int - mini-batch 大小（默认=num_envs=4）
            epochs: int - 更新轮数
            update_critic: bool - 是否更新 critic
        """
        n = len(time_series)
        inds = np.arange(n)
        
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                mb = inds[start:end]
                
                # 提取 mini-batch
                ts_b = time_series[mb]
                act_b = act[mb]
                adv_b = adv[mb]
                old_logp_b = old_logp[mb]
                returns_b = returns[mb]
                gobs_b = gobs[mb]
                
                # 训练
                self.train_actor_step(ts_b, act_b, adv_b, old_logp_b)
                if update_critic:
                    self.train_critic_step(gobs_b, returns_b)