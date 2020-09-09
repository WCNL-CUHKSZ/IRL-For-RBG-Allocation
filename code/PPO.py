import tensorflow as tf
import pandas as pd
import numpy as np
import os


class Actor:
    def __init__(self, user_num, feature_dim, rbg_num, lr, loss):
        self.user_num = user_num
        self.feature_dim = feature_dim
        self.rbg_num = rbg_num
        self.lr = lr
        self.loss = loss

    def backbone(self, inputs):
        x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(128, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(self.user_num, activation='softmax')(x)

        return outputs

    def build(self):
        inputs = tf.keras.Input([self.user_num * self.feature_dim, ])
        outputs = self.backbone(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=self.loss
        )

        return model


class Critic:
    def __init__(self, user_num, feature_dim, rbg_num, lr, loss):
        self.user_num = user_num
        self.feature_dim = feature_dim
        self.rbg_num = rbg_num
        self.lr = lr
        self.loss = loss

    def backbone(self, inputs):
        x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(128, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(1)(x)

        return outputs

    def build(self):
        inputs = tf.keras.Input([self.feature_dim * self.user_num, ])
        outputs = self.backbone(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=self.loss
        )

        return model

class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field \
                in self.memory.columns)

class PPOAgent:
    def __init__(self,
                 user_num,
                 feature_dim,
                 rbg_num,
                 clip_ratio=0.1,
                 gamma=0.99,
                 lambd=0.99,
                 min_trajectory_length=1000,
                 batch_size=64,
                 batches=2,
                 lr=0.001
                 ):
        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim

        self.gamma = gamma
        self.lambd = lambd
        self.min_trajectory_length = min_trajectory_length
        self.batches = batches
        self.batch_size = batch_size
        self.lr = lr

        self.replayer = PPOReplayer()

        def ppo_loss(y_true, y_pred):
            # 真实值 y_true : (2*action_n,) 旧策略的策略概率 + 优势函数
            p = y_pred  # 新策略概率
            p_old = y_true[:, :self.user_num]  # 旧策略概率
            advantage = y_true[:, self.user_num:]  # 优势
            surrogate_advantage = (p / p_old) * advantage  # 代理优势
            clip_times_advantage = clip_ratio * advantage
            max_surrogate_advantage = advantage + tf.where(advantage > 0.,
                                                           clip_times_advantage, -clip_times_advantage)
            clipped_surrogate_advantage = tf.minimum(surrogate_advantage,
                                                     max_surrogate_advantage)
            return - tf.reduce_mean(clipped_surrogate_advantage, axis=-1)

        self.actor_net = Actor(
            self.user_num,
            self.feature_dim,
            self.rbg_num,
            self.lr,
            loss=ppo_loss
        ).build()

        self.critic_net = Critic(
            self.user_num,
            self.feature_dim,
            self.rbg_num,
            self.lr,
            loss='MSE'
        ).build()

    def decide(self, state):
        probs = self.actor_net.predict(state)[0]
        action = np.random.choice(self.user_num, p=probs)
        return action

    def learn(self, s, a, r):
        df = pd.DataFrame()  # 开始对本回合经验进行重构
        df['state'] = s
        df['action'] = a
        df['reward'] = r
    
        states = np.stack(df['state'])
        df['v'] = self.critic_net.predict(states)
        pis = self.actor_net.predict(states)
        df['pi'] = [a.flatten() for a in np.split(pis, pis.shape[0])]
        df['next_v'] = df['v'].shift(-1).fillna(0.)
        df['u'] = df['reward'] + self.gamma * df['next_v']
        df['delta'] = df['u'] - df['v']  # 时序差分误差
        df['return'] = df['reward']  # 初始化优势估计，后续会再更新
        df['advantage'] = df['delta']  # 初始化优势估计，后续会再更新
        for i in df.index[-2::-1]:  # 指数加权平均
            df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
            df.loc[i, 'advantage'] += self.gamma * self.lambd * \
                                      df.loc[i + 1, 'advantage']  # 估计优势
        fields = ['state', 'action', 'pi', 'advantage', 'return']
        self.replayer.store(df[fields])  # 存储重构后的回合经验

        for i in range(self.batches):
            states, actions, pis, advantages, returns = self.replayer.sample(size=self.batch_size)
            ext_advantages = np.zeros_like(pis)
            ext_advantages[range(self.batch_size), actions] = advantages
            actor_targets = np.hstack([pis, ext_advantages])  # 执行者目标
            self.actor_net.fit(states, actor_targets, verbose=0)
            self.critic_net.fit(states, returns, verbose=0)

        self.replayer = PPOReplayer()  # 为下一回合初始化经验回放
