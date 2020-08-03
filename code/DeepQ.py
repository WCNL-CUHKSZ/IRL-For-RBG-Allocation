import tensorflow as tf
import numpy as np
import pandas as pd

class Network:
    def __init__(
            self,
            user_num,
            feature_dim,
            l1=0.01,
            l2=0.01
                 ):
        self.user_num = user_num
        self.feature_dim = feature_dim
        self.l1 = l1
        self.l2 = l2

    def backbone(self, input_tensor):
        regularzier = tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2)

        x = tf.keras.layers.Dense(64, activation='tanh', kernel_regularizer=regularzier, name='layer1')(input_tensor)
        x = tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=regularzier, name='layer2')(x)
        x = tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=regularzier, name='layer3')(x)
        outputs = tf.keras.layers.Dense(self.user_num)(x)

        return outputs

    def build(self):
        input_tensor = tf.keras.Input(shape=[self.feature_dim * self.user_num])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor, name='AGENT')

        model.compile(
            optimizer='Adam',
            loss='MSE'
        )

        return model

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward',
                                            'next_state', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        self.memory.dropna(inplace=True)
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class DQNAgent:
    def __init__(self,
                 user_num,
                 rbg_num,
                 feature_dim,
                 gamma=0.99,
                 epsilon=0.001,
                 replayer_capacity=10000,
                 batch_size=64):

        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim

        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放

        self.network = Network(
            self.user_num,
            self.feature_dim
        )
        self.evaluate_net = self.network.build()# 评估网络
        self.target_net = self.network.build()# 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def learn(self, state, action, reward, next_state, done):
        if done:  # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())
        else:
            self.replayer.store(state, action, reward, next_state, done)  # 存储经验

            states, actions, rewards, next_states, dones = \
                self.replayer.sample(self.batch_size)  # 经验回放

            next_qs = self.target_net.predict(next_states)
            next_max_qs = next_qs.max(axis=-1)
            us = rewards + self.gamma * (1. - dones) * next_max_qs
            targets = self.evaluate_net.predict(states)
            targets[np.arange(us.shape[0]), actions] = us
            self.evaluate_net.fit(states, targets, verbose=0)

    def decide(self, state):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return [np.random.randint(self.user_num)]
        qs = self.evaluate_net.predict(state[np.newaxis])
        action = [np.argmax(qs)]
        return action

    def save(self, epoch):
        self.target_net.save('snapshots/'+'target_net_epoch'+str(epoch)+'.h5')

