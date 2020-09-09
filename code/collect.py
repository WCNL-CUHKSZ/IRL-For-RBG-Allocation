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
from simulator.Opportunistic import OPPORTUNISTIC
from simulator.RoundRobin import ROUNDROBIN
from simulator.RandomSelection import RANDOMSELECTION

class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(
            index=range(capacity),
            columns=['state',
                     'action',
                     'reward',
                     'label']
)
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

class COLLECT:
    def __init__(
            self,
            env_flag,
            user_num,
            rbg_num,
            feature_dim,
            traj_length,
            traj_number,
            replayer_capacity
                 ):
        self.env_flag = env_flag
        self.user_num = user_num
        self.rbg_num = rbg_num
        self.feature_dim = feature_dim

        self.n_states = traj_number * traj_length
        self.n_actions = user_num

        self.av_ues_info = self.user_info_threshold(
            self.load_av_ue_info(),
            threshold_min=10e+5,
            threshold_max=10e+6)

        if self.env_flag == 'PF':
            self.env = PROPORTIONALFAIRNESS(
                lambda_avg=1,
                lambda_fairness=1,
                reward_flag=None)
        if self.env_flag == 'OP':
            self.env = OPPORTUNISTIC(
                lambda_avg=1,
                lambda_fairness=1,
                reward_flag=None)
        if self.env_flag == 'RR':
            self.env = ROUNDROBIN(
                lambda_avg=1,
                lambda_fairness=1,
                reward_flag=None)
        if self.env_flag == 'RS':
            self.env = RANDOMSELECTION(
                lambda_avg=1,
                lambda_fairness=1,
                reward_flag=None)

        self.length = traj_length
        self.number = traj_number
        self.replayer_capacity = replayer_capacity
        self.e_replayer = Replayer(capacity=self.replayer_capacity)
        self.g_replayer = Replayer(capacity=self.replayer_capacity)

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

    def run(self, replayer, rs, re):
        for i in range(rs, re):
            av_ues_idx_start = i
            av_ues_idx = list(range(av_ues_idx_start, av_ues_idx_start + self.user_num))
            state = self.env.reset(self.av_ues_info, av_ues_idx)

            j = 0
            while (j < self.length):
                action = self.env.action()
                if self.env_flag == 'PF':
                    next_state, reward, done, info = self.env.step(None, 0, 0)
                else:
                    next_state, reward, done, info = self.env.step(action, 0, 0)
                ''' save the transition '''
                # state = np.argsort(state, axis=1)
                replayer.store(state.reshape(-1), action[0], reward, 1.0)
                state = next_state

                j += 1

        replayer.memory.dropna(inplace=True)
        replayer.memory.reset_index(inplace=True, drop=True)

        traj_state = np.array(replayer.memory['state'].tolist())
        for i in range(traj_state.shape[1]):
            traj_state[:, i]= (traj_state[:, i] - traj_state[:, i].mean()) / (traj_state[:, i].std() + 1e-3)
            
        replayer.memory['state'] = traj_state.tolist()

        return replayer.memory

    def generate(self):
        print('INFO: Generating trajectories!')
        ge_start = time.clock()
        
        e_traj = self.run(self.e_replayer, rs=0, re=self.number)
        g_traj = self.run(self.g_replayer, rs=self.number, re=self.number * 2)
        
        ge_end = time.clock()
        print('INFO: Trajectories generation accomplished, time cost={}s'.format(ge_end - ge_start))

        return e_traj, g_traj








