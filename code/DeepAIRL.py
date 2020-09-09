from PPO import PPOAgent

import tensorflow as tf
from scipy import stats
import pandas as pd
import numpy as np
import time

class Reward:
    def __init__(
            self,
            user_num,
            feature_dim,
            l1,
            l2
    ):
        self.user_num = user_num
        self.feature_dim = feature_dim
        self.l1 = l1
        self.l2 = l2

    def backbone(self, input_tensor):
        x = tf.keras.layers.Dense(64, activation='tanh', name='layer1')(input_tensor)
        x = tf.keras.layers.Dense(128, activation='tanh', name='layer2')(x)
        outputs = tf.keras.layers.Dense(1, )(x)

        return outputs

    def build(self):
        input_tensor = tf.keras.Input(shape=[self.user_num * self.feature_dim + self.user_num, ])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor, name='REWARD')

        model.compile(
            optimizer='Adam',
            loss='MSE'
        )

        return model

class Discriminator:
    def __init__(
            self,
            user_num,
            feature_dim,
            loss,
            l1,
            l2
    ):
        self.user_num = user_num
        self.feature_dim = feature_dim
        self.loss = loss
        self.l1 = l1
        self.l2 = l2

    def backbone(self, input_tensor):
        x = tf.keras.layers.Dense(1024, )(input_tensor)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(512, )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(256, )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(128, )(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(64, )(x)
        x = tf.keras.layers.Activation('relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return outputs

    def build(self):
        input_tensor = tf.keras.Input(shape=[self.user_num * self.feature_dim + self.user_num, ])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor, name='DISCRIMINATOR')

        model.compile(
            optimizer='Adam',
            loss=self.loss
        )

        return model

class DeepAIRL:
    def __init__(
            self,
            env_flag,
            user_num,
            rbg_num,
            feature_dim,
            n_states,
            n_actions,
            traj,
            traj_length,
            gamma=0.01,
            l1=0.01,
            l2=0.01,
            epochs=100,
            lr=0.001,
            lr_decay=0.0001,
            batch_size=64
    ):
        self.env_flag = env_flag
        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim
        self.n_states = n_states
        self.n_actions = n_actions

        self.gamma = gamma
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size

        self.discriminator = Discriminator(
            user_num=self.user_num,
            feature_dim=self.feature_dim,
            loss='binary_crossentropy',
            l1=self.l1,
            l2=self.l2
        ).build()

        self.policy = None # initial policy
        self.traj = traj
        self.traj_length = traj_length
        self.PPOAgent = PPOAgent(
            user_num=self.user_num,
            feature_dim=self.feature_dim,
            rbg_num=self.rbg_num
        )

        self.e_traj = self.traj[0].copy()
        self.g_traj =  self.traj[1].copy()
        self.mixed_traj = None

    def execute_policy(self):
        states = np.array(self.g_traj['state'].tolist())
        actions = self.PPOAgent.actor_net.predict(states)
        actions = np.argmax(actions, axis=1)

        self.g_traj['action'] = actions
        self.g_traj['label'] = np.zeros((self.traj_length[1]))

    def update_policy(self):
        states = self.traj_batch[0].numpy().tolist()
        actions = self.traj_batch[1].numpy().tolist()
        rewards = self.reward.numpy().reshape(-1)
        
        self.PPOAgent.learn(states, actions, rewards)
        
    def update_reward(self):
        reward = tf.math.log(self.D_output) - tf.math.log(1 - self.D_output)
        return reward
    
    def logistic_loss(self):
        loss = tf.keras.losses.binary_crossentropy(y_pred=self.D_output, y_true=self.traj_batch[3])
        return tf.reduce_mean(loss)

    def hinge_loss(self):
        loss = tf.keras.losses.hinge(y_pred=self.D_output, y_true=self.traj_batch[3])
        return tf.reduce_mean(loss)

    def get_JS(self):
        expert_policy = np.array(self.e_traj['action'].tolist()) + 1
        states = np.array(self.e_traj['state'].tolist())
        current_policy = self.PPOAgent.actor_net.predict(states)
        current_policy = np.argmax(current_policy, axis=1) + 1

        """ get KL divergence between two policy """
        self.JS = 0.5 * stats.entropy(expert_policy, current_policy) + 0.5 * stats.entropy(current_policy, expert_policy)

    def create_dataset(self, traj):
        sts = np.array(traj['state'].tolist())
        acs = np.array(traj['action'].tolist())
        res = np.array(traj['reward'].tolist())
        las = np.array(traj['label'].tolist())
        datasets = tf.data.Dataset.from_tensor_slices((sts, acs, res, las))
        datasets = datasets.batch(self.batch_size)
        iterator = iter(datasets)

        return iterator

    def train_step(self):
        encoded_actions = tf.keras.utils.to_categorical(self.traj_batch[1], num_classes=self.user_num)
        sa_pairs = tf.concat((self.traj_batch[0], encoded_actions), axis=1)

        with tf.GradientTape(persistent=True) as tape:
            self.D_output = self.discriminator(sa_pairs)
            self.reward = self.update_reward()
            self.loss = self.logistic_loss()
            """ update discriminator and reward """
            gradients = tape.gradient(self.loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        del tape

        """ release resources """
        print("INFO: iteration={}, discriminator loss={:.4}".format(self.step, self.loss))
        
        """ summarying the weights variation """
        with self.writer.as_default():
            tf.summary.scalar(name='DISCRIMINATOR LOSS', data=self.loss, step=self.step) # test code
            tf.summary.histogram(name='REWARD', data=self.reward, step=self.step)  # test code
            for index in range(len(self.discriminator.weights)):
                if len(self.discriminator.weights[index].shape) == 2:
                    tf.summary.histogram(name='D_THETA_' + str(index), data=self.discriminator.weights[index], step=self.step)
                else:
                    tf.summary.histogram(name='D_BIAS_' + str(index), data=self.discriminator.weights[index], step=self.step)
            self.writer.flush()

        self.step += 1

    def train(self):
        """ Collect trajectories by executing policy π """
        self.execute_policy()
        self.e_iterator = self.create_dataset(self.e_traj)
        self.g_iterator = self.create_dataset(self.g_traj)

        while True:
            try:
                self.traj_batch = next(self.e_iterator)
                self.train_step()
                self.update_policy()
                
                self.traj_batch = next(self.g_iterator)
                self.train_step()
                self.update_policy()
            except:
                break
        
        self.get_JS()
        with self.writer.as_default():
            tf.summary.scalar(name='JS DIVERGENCE', data=self.JS, step=self.epoch) # test code
            self.writer.flush()
        
        self.epoch += 1
            
    def irl(self):
        print('INFO: Training start! Env = {}'.format(self.env_flag))
        self.optimizer = tf.optimizers.Adam(self.lr, self.lr_decay)
        self.writer = tf.summary.create_file_writer('./log/train_irl/' + self.env_flag)
        self.step = 0
        self.epoch = 0

        for epoch in range(self.epochs):
            t_s = time.clock()
            print("INFO: Epoch={}".format(epoch + 1))
            self.train()
            t_e = time.clock()
            
            if (epoch+1) % 10 == 0:
                self.discriminator.save('./snapshots/'+self.env_flag+'_D_epoch'+ str(epoch + 1) +'.h5')
                self.PPOAgent.actor_net.save('./snapshots/'+self.env_flag+'_actor_epoch'+ str(epoch + 1) +'.h5')