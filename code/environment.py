"""
For generating trajectories
"""

RBG_NUM = 17
USER_NUM = 10
FEATURE_NUM = 57

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import time
import os

from simulator.ProportionalFairness import PROPORTIONALFAIRNESS

class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['state',
                     'action',
                     'reward',
                     'next_state',
                     'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, frac):
        if frac == 1.0:
            return self.memory
        else:
            self.memory.dropna(inplace=True)
            sample = self.memory.sample(frac=frac)
            return sample

    def clear(self):
        self.__init__(capacity=self.capacity)

class ENV:
    def __init__(
            self,
            user_num,
            rbg_num,
            feature_dim,
            trajectory_length,
            trajectory_number
                 ):
        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim

        self.n_states = trajectory_number * trajectory_length
        self.n_actions = user_num

        self.av_ues_info = self.user_info_threshold(
            self.load_av_ue_info(),
            threshold_min=10e+5,
            threshold_max=10e+6)

        self.env = PROPORTIONALFAIRNESS(
            lambda_avg=1,
            lambda_fairness=1,
            reward_flag=None)
        self.length = trajectory_length
        self.number = trajectory_number
        self.replayer = Replayer(capacity=self.length)

    def load_av_ue_info(self, file='simulator/ue_info.pkl'):
        assert (os.path.exists(file) == True)
        with open(file, 'rb') as f:
            ues_info = pickle.load(f)

        for ue_idx in range(len(ues_info) - 1, -1, -1):
            if ues_info[ue_idx]['rsrp'] < -130:
                ues_info.pop(ue_idx)

        return ues_info

    def user_info_threshold(self, user_info, threshold_min, threshold_max):
        filtered_user_info = []
        for i in range(len(user_info)):
            if (user_info[i]['buffer'] > threshold_min) and (user_info[i]['buffer'] < threshold_max):
                filtered_user_info.append(user_info[i])
        return filtered_user_info

    def generate(self):
        print('INFO: Generating trajectories!')
        ge_start = time.clock()
        memory = []
        for i in range(self.number):
            av_ues_idx_start = i
            av_ues_idx = list(range(av_ues_idx_start, av_ues_idx_start + self.user_num))
            state = self.env.reset(self.av_ues_info, av_ues_idx)

            j = 0
            while (j < self.length):
                action = self.env.action()
                next_state, reward, done, info = self.env.step(None, 0, 0)
                ''' save the transition '''
                self.replayer.store(state, action, reward, next_state, done)
                state = next_state

                j += 1

            memory.append(self.replayer.memory[['state', 'action']])
            self.replayer.clear()

        state_list = []
        action_list = []
        for item in memory:
            state_list.append(item['state'])
            action_list.append(item['action'])

        ge_end = time.clock()
        print('INFO: Trajectories generation accomplished, time cost={}s'.format(ge_end - ge_start))

        return {'s':state_list, 'a':action_list}

    def feature_matrix(self):
        fm = np.load('static/fm.npy')

        return fm

    def transition_probability(self):
        P_sas = np.load('static/P_sas.npy')

        return P_sas







