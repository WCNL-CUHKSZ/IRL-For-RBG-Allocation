from DeepMaxEntIRL import DeepMaxEntIRL
from environment import ENV

import logging
logging.disable(30)

if __name__ == '__main__':
    USER_NUM = 3
    RBG_NUM = 1
    FEATURE_DIM = 4
    N_ACTIONS = USER_NUM
    TRAJ_NUMBER = 5
    TRAJ_LENGTH = 50

    env = ENV(
        user_num=USER_NUM,
        rbg_num=RBG_NUM,
        feature_dim=FEATURE_DIM,
        trajectory_number=TRAJ_NUMBER,
        trajectory_length=TRAJ_LENGTH
    )
    trajectories = env.generate()
    feature_matrix = env.feature_matrix()
    transition_probability = env.transition_probability()

    alg = DeepMaxEntIRL(
        user_num=USER_NUM,
        rbg_num=RBG_NUM,
        feature_dim=FEATURE_DIM,
        trajectories=trajectories,
        feature_matrix=feature_matrix,
        n_states=feature_matrix.shape[0],
        n_actions=N_ACTIONS,
        p_sas=transition_probability,
    )

    alg.irl()

