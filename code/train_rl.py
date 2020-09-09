import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import logging
import os

logging.disable(30)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from simulator.ProportionalFairness import PROPORTIONALFAIRNESS, load_av_ue_info
from simulator.Opportunistic import OPPORTUNISTIC
from simulator.RoundRobin import ROUNDROBIN
from simulator.RandomSelection import RANDOMSELECTION
from DeepQ import DQNAgent

def user_info_threshold(user_info, threshold_min, threshold_max):
    filtered_user_info = []
    for i in range(len(user_info)):
        if (user_info[i]['buffer'] > threshold_min) and (user_info[i]['buffer'] < threshold_max):
            filtered_user_info.append(user_info[i])
    return filtered_user_info

def get_env(env_flag):
    if env_flag == 'PF':
        env = PROPORTIONALFAIRNESS(
            lambda_avg=LAMBDA_AVG,
            lambda_fairness=LAMBDA_FAIRNESS,
            reward_flag=REWARD_FLAG)
    if env_flag == 'OP':
        env = OPPORTUNISTIC(
            lambda_avg=LAMBDA_AVG,
            lambda_fairness=LAMBDA_FAIRNESS,
            reward_flag=REWARD_FLAG)
    if env_flag == 'RR':
        env = ROUNDROBIN(
            lambda_avg=LAMBDA_AVG,
            lambda_fairness=LAMBDA_FAIRNESS,
            reward_flag=REWARD_FLAG)
    else:
        env = RANDOMSELECTION(
            lambda_avg=LAMBDA_AVG,
            lambda_fairness=LAMBDA_FAIRNESS,
            reward_flag=REWARD_FLAG)

    return env

def encoding(state, user_num):
    encoded_state = np.argsort(state, axis=0)

    return encoded_state

if __name__ == '__main__':
    RBG_NUM = 1
    USER_NUM = 3
    FEATURE_DIM = 4
    TTI_SUM = 200
    RANDOM_SEED = 7
    POSSION_AVG = 10 / 1000
    REPLAYER_CAPACITY = 5000
    LAMBDA_AVG = 10000
    LAMBDA_FAIRNESS = 10000
    EPOCH = 100
    INTERNAL = 10
    LR = 0.0005
    SAMPLE_FRAC = 0.6
    REWARD_FLAG = None
    ENV_FLAG = 'OP'

    co_env = get_env(ENV_FLAG)
    rl_env = PROPORTIONALFAIRNESS(
        lambda_avg=LAMBDA_AVG,
        lambda_fairness=LAMBDA_FAIRNESS,
        reward_flag=REWARD_FLAG)

    av_ues_info = load_av_ue_info()
    av_ues_info = user_info_threshold(av_ues_info, threshold_min=10e+5, threshold_max=10e+6)

    agent = DQNAgent(
        user_num=USER_NUM,
        feature_dim=FEATURE_DIM,
        rbg_num=RBG_NUM,
        replayer_capacity=REPLAYER_CAPACITY,
    )

    agent.evaluate_net.summary()
    reward = tf.keras.models.load_model('reward.h5')
    reward.summary()
    writer = tf.summary.create_file_writer('./log/train_rl')

    tti_list = []
    average_list = []
    fariness_list = []
    reward_list = []

    for game in range(EPOCH):
        time_start = time.clock()
        INITIAL_USER_START = int(game / INTERNAL)
        av_ues_idx = list(range(INITIAL_USER_START, INITIAL_USER_START + USER_NUM))

        co_state = co_env.reset(av_ues_info, av_ues_idx)
        rl_state = rl_env.reset(av_ues_info, av_ues_idx)
        rl_state = encoding(rl_state, USER_NUM)

        tti = 0
        episode_reward = 0

        while (tti < TTI_SUM):
            POSSION_ADD_USER = 0

            ''' CO Allocation '''
            co_action = co_env.action()
            if ENV_FLAG == 'PF':
                co_next_state, co_reward, co_done, co_info = co_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)

            else:
                co_next_state, co_reward, co_done, co_info = co_env.step(co_action, POSSION_ADD_USER, INITIAL_USER_START)

            ''' RL Allocation '''
            if rl_env.bs.newtx_rbg_ue == [None for _ in range(RBG_NUM)]:
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(None, POSSION_ADD_USER, INITIAL_USER_START)
            else:
                rl_action = agent.decide(rl_state)
                print('s={}, RL a={}, {} a={}'.format(rl_state, rl_action, ENV_FLAG, co_action))
                rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(rl_action, POSSION_ADD_USER, INITIAL_USER_START)

                rl_next_state = encoding(rl_next_state, USER_NUM)
                relative_reward = reward.predict(rl_state[np.newaxis])[0]
                agent.learn(rl_state, rl_action, relative_reward, rl_next_state, done=False)

                episode_reward += relative_reward
                rl_state = rl_next_state

            tti += 1

        ''' visualize the agent's weights'''
        with writer.as_default():
            for index in range(len(agent.evaluate_net.weights)):
                tf.summary.histogram(str(index), data=agent.evaluate_net.weights[index], step=game)
            writer.flush()

        ''' get result '''
        co_result = co_env.get_result()
        rl_result = rl_env.get_result()

        tti_list.append(tti)
        average_list.append(rl_result['average'])
        fariness_list.append(rl_result['fairness'])
        reward_list.append(episode_reward / TTI_SUM)

        ''' update target netwrok '''
        agent.learn(None, None, None, None, done=True)
        ''' clear the replayer and save the model '''
        agent.save(epoch=game + 1)

        record_file = './log/rl2co_train.txt'

        with open(record_file, 'a') as file:
            record = 'Epoch={}, user_num={}, ' \
                     '{} ues_rate:[avg={},fairness={}], ' \
                     '{} ues_rate:[avg={},fairness={}], ' \
                     'Episode Reward={}'.format(
                game+1, len(rl_env.bs.ues),
                ENV_FLAG, co_result['average'], co_result['fairness'],
                'RL', rl_result['average'], rl_result['fairness'],
                episode_reward
            )
            file.write(record)
            file.write('\n')
            file.close()

        time_end = time.clock()
        record = pd.read_csv(record_file, header=None, sep='[=:, \]]', engine='python')
        print('Epoch={}, time cost={} s, episode reward={}'.format(game + 1, time_end - time_start, episode_reward))
        print('User start index={}'.format(INITIAL_USER_START))

        print('{} avg better={}'.format(ENV_FLAG, record[9][record[9] > record[17]].shape[0]))
        print('{} 5%-tile better={}'.format(ENV_FLAG, record[11][record[11] > record[19]].shape[0]))
        print('{} both better={}'.format(ENV_FLAG, record[(record[9] > record[17]) & (record[11] > record[19])].shape[0]))

        print('RL avg better={}'.format(record[17][record[17] > record[9]].shape[0]))
        print('RL 5%-tile better={}'.format(record[19][record[19] > record[11]].shape[0]))
        print('RL both better={}'.format(record[(record[17] > record[9]) & (record[19] > record[11])].shape[0]))
        print('\n')

        ''' plot the figure '''
        plt.figure(figsize=(15, 7))

        plt.subplot(2,2,1)
        plt.title('Reward')
        plt.plot(reward_list)

        plt.subplot(2,2,2)
        plt.title('Time')
        plt.plot(tti_list)

        plt.subplot(2,2,3)
        plt.title('Average')
        plt.plot(average_list)

        plt.subplot(2,2,4)
        plt.title('Fairness')
        plt.plot(fariness_list)

        plt.savefig('./log/record.png')
