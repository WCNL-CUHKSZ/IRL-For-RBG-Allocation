import tensorflow as tf
import numpy as np
import time

import value_iteration
from itertools import product


class Network:
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
        x = tf.keras.layers.Dense(256, activation='tanh', name='layer3')(x)
        x = tf.keras.layers.Dropout(rate=0.01)(x)
        outputs = tf.keras.layers.Dense(1, )(x)

        return outputs

    def build(self):
        input_tensor = tf.keras.Input(shape=[self.feature_dim * self.user_num])
        output_tensor = self.backbone(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor, name='REWARD')

        model.compile(
            optimizer='Adam',
            loss='MSE'
        )

        return model


class DeepMaxEntIRL:
    def __init__(
            self,
            user_num,
            rbg_num,
            feature_dim,
            trajectories,
            feature_matrix,
            n_states,
            n_actions,
            p_sas,
            discount=0.01,
            l1=0.01,
            l2=0.01,
            epochs=100,
            lr=0.0001
    ):
        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim

        self.discount = discount
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr

        self.reward = Network(
            self.user_num,
            self.feature_dim,
            self.l1,
            self.l2
        ).build()

        self.feature_matrix = feature_matrix
        self.p_sas = p_sas
        self.n_states = n_states
        self.n_actions = n_actions

        self.trajectories = np.array(trajectories['s'])
        self.trajectories = self.transform_traj()
        """
        self.trajectories: Numpy array with shape=[N, L, S], N is the quantity of the trajectories,
        L is length of trajectory, S is the feature shape of state.
        """

    def transform_traj(self):
        self.trajectories = np.reshape(self.trajectories,
                                       (self.trajectories.shape[0],
                                        self.trajectories.shape[1],
                                        self.feature_dim * self.user_num))

        self.traj_traj = np.zeros((self.trajectories.shape[0],
                                   self.trajectories.shape[1] - 1,
                                   self.trajectories.shape[2]))

        """
        calculate the relay state
        """
        for i in range(self.trajectories.shape[0]):
            for j in range(self.trajectories.shape[1] - 1):
                self.traj_traj[i][j] = self.trajectories[i][j + 1] - self.trajectories[i][j]

        self.traj_traj[self.traj_traj < 0] = -1
        self.traj_traj[self.traj_traj == 0] = 0
        self.traj_traj[self.traj_traj > 0] = 1

        traj_index = np.zeros((self.trajectories.shape[0], self.trajectories.shape[1], 1)).astype('int32')
        for i in range(self.traj_traj.shape[0]):
            for j in range(self.traj_traj.shape[1]):
                for k in range(self.feature_matrix.shape[0]):
                    if list(self.traj_traj[i][j]) == list(self.feature_matrix[k]):
                        traj_index[i][j] = k

        return traj_index

    def get_svf(self):
        svf = np.zeros(self.n_states)

        for trajectory in self.trajectories:
            for state in trajectory:
                svf[state[0]] += 1

        svf /= self.trajectories.shape[0]

        return svf

    def get_expected_svf(self, r):
        n_trajectories = self.trajectories.shape[0]
        trajectory_length = self.trajectories.shape[1]

        policy = value_iteration.find_policy(
            self.n_states,
            self.n_actions,
            self.p_sas,
            r,
            self.discount)

        start_state_count = np.zeros(self.n_states)
        for trajectory in self.trajectories:
            start_state_count[trajectory[0, 0]] += 1
        p_start_state = start_state_count / n_trajectories

        expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
        for t in range(1, trajectory_length):
            expected_svf[:, t] = 0
            for i, j, k in product(range(self.n_states), range(self.n_actions), range(self.n_states)):
                expected_svf[k, t] += (expected_svf[i, t - 1] *
                                       policy[i, j] *  # Stochastic policy
                                       self.p_sas[i, j, k])

        return expected_svf.sum(axis=1)

    def train_step(self):
        with tf.GradientTape(persistent=True) as tape:
            reward = self.reward(self.feature_matrix.astype('float32'))
            expected_svf = self.get_expected_svf(reward[..., 0])
            svf_difference = tf.cast(self.svf - expected_svf, 'float32')

            for j in range(self.n_states):
                grad_reward = tape.gradient(reward[j], self.reward.trainable_variables)
                grad_for_state = []
                for k in range(len(grad_reward)):
                    regluarizer = self.l1 * tf.reduce_sum(tf.abs(grad_reward[k]))
                    regluarizer += self.l2 * tf.reduce_sum(tf.pow(grad_reward[k], 2))
                    grad_for_state.append(grad_reward[k] * svf_difference[j] - regluarizer)

                # grad_for_state = [grad_reward[k] * svf_difference[j] for k in range(len(grad_reward))]
            self.optimizer.apply_gradients(zip(grad_for_state, self.reward.trainable_variables))

        """release resources"""
        del tape

    def irl(self):
        print('INFO: Training start!')
        self.svf = self.get_svf()
        self.optimizer = tf.optimizers.Adam(self.lr)
        self.writer = tf.summary.create_file_writer('./log/train_irl')

        for i in range(self.epochs):
            t_s = time.clock()
            self.train_step()
            t_e = time.clock()

            reward_list = self.reward(self.feature_matrix.astype('float32')) # test code

            print("INFO: iteration={}, time cost={}s".format(i, t_e - t_s))

            """ summarying the weights variation """
            with self.writer.as_default():
                tf.summary.histogram(name='reward', data=reward_list, step=i) # test code
                for index in range(len(self.reward.weights)):
                    if len(self.reward.weights[index].shape) == 2:
                        tf.summary.histogram(name='Theta_' + str(index), data=self.reward.weights[index], step=i)
                    else:
                        tf.summary.histogram(name='Bias_' + str(index), data=self.reward.weights[index], step=i)

                self.writer.flush()

        self.reward.save('reward.h5')
