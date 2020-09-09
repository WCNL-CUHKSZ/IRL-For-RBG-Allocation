import numpy as np
import pandas as pd
import tensorflow as tf
from DeepAIRL import DeepAIRL
from simulator.ProportionalFairness import load_av_ue_info, PROPORTIONALFAIRNESS

def user_info_threshold(user_info, threshold_min, threshold_max):
    filtered_user_info = []
    for i in range(len(user_info)):
        if (user_info[i]['buffer'] > threshold_min) and (user_info[i]['buffer'] < threshold_max):
            filtered_user_info.append(user_info[i])
    return filtered_user_info

class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['state', 'action', 'reward', 'label'])
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
av_ues_info = load_av_ue_info()
av_ues_info = user_info_threshold(av_ues_info, threshold_min=1e+6, threshold_max=1e+7)
pf_env = PROPORTIONALFAIRNESS(
    lambda_avg=None,
    lambda_fairness=None,
    reward_flag=None)

INITIAL_USER_START = 0
USER_NUM = 3
av_ues_idx = list(range(INITIAL_USER_START, INITIAL_USER_START + USER_NUM))
state = pf_env.reset(av_ues_info, av_ues_idx)
replayer = Replayer(capacity=500)

tti = 0
while (tti < 500):
    action = pf_env.action()[0]
    next_state, reward, done, info = pf_env.step(None, 0, 0)
    replayer.store(state.reshape(-1), action, reward, 1.0)
    state = next_state
    tti += 1

replayer.memory.dropna(inplace=True)
replayer.memory.reset_index(inplace=True, drop=True)
alg = DeepAIRL(
    env_flag='PF',
    user_num=3,
    rbg_num=1,
    feature_dim=4,
    n_states=1296,
    n_actions=3,
    traj=replayer.memory,
    traj_length=replayer.memory.shape[0]
)

alg.execute_policy()
alg.traj_batch = alg.get_batch()
encoded_actions = tf.keras.utils.to_categorical(alg.traj_batch['action'], num_classes=alg.user_num)
sa_pairs = np.concatenate((alg.traj_batch['state'], encoded_actions), axis=1)
alg.D_output = alg.discriminator.predict(sa_pairs)
alg.reward = alg.update_reward()
alg.loss = alg.logistic_loss()

alg.update_policy()