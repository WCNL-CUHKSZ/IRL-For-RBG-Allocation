from DeepAIRL import DeepAIRL
from collect import COLLECT
import logging
import os

logging.disable(30)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    USER_NUM = 3
    RBG_NUM = 1
    FEATURE_DIM = 4
    N_STATES = None
    N_ACTIONS = USER_NUM
    TRAJ_NUMBER = 10
    TRAJ_LENGTH = 500
    EPOCH = 50
    ENV_FLAG = ['PF', 'OP', 'RR', 'RS']

    for flag in ENV_FLAG:
        collect = COLLECT(
            env_flag=flag,
            user_num=USER_NUM,
            feature_dim=FEATURE_DIM,
            rbg_num=RBG_NUM,
            traj_length=TRAJ_LENGTH,
            traj_number=TRAJ_NUMBER,
            replayer_capacity=TRAJ_LENGTH * TRAJ_NUMBER
        )

        e_traj, g_traj = collect.generate()

        alg = DeepAIRL(
            env_flag=flag,
            user_num=USER_NUM,
            rbg_num=RBG_NUM,
            epochs=EPOCH,
            feature_dim=FEATURE_DIM,
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            traj=(e_traj, g_traj),
            traj_length=(e_traj.shape[0], g_traj.shape[0]),
            batch_size=512,
            lr=1e-4,
            lr_decay=1e-5
        )

        alg.irl()