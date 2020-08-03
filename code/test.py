# import tensorflow as tf
# import numpy as np
#
# input_tensor = tf.keras.Input(shape=[30])
# x = tf.keras.layers.Dense(64)(input_tensor)
# x = tf.keras.layers.Dense(128)(x)
# x = tf.keras.layers.Dense(256)(x)
# x = tf.keras.layers.Dense(1)(x)
#
# model = tf.keras.Model(input_tensor, x)
# for item in model.weights:
#     print(len(item.shape))
# writer = tf.summary.create_file_writer('./log/train')
# with writer.as_default():
#     for i in range(100):
#         tf.summary.histogram('model', model.weights[1], step=i)
#         writer.flush()
# def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
#     n_states, _, n_actions = p_transition.shape
#     nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
#
#     # Backward Pass
#     # 1. initialize at terminal states
#     zs = np.zeros(n_states)  # zs: state partition function
#     zs[terminal] = 1.0
#
#     # 2. perform backward pass
#     for _ in range(2 * n_states):  # longest trajectory: n_states
#         # reset action values to zero
#         za = np.zeros((n_states, n_actions))  # za: action partition function
#
#         # for each state-action pair
#         for s_from, a in product(range(n_states), range(n_actions)):
#
#             # sum over s_to
#             for s_to in range(n_states):
#                 za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]
#
#         # sum over all actions
#         zs = za.sum(axis=1)
#
#     # 3. compute local action probabilities
#     p_action = za / zs[:, None]
#
#     # Forward Pass
#     # 4. initialize with starting probability
#     d = np.zeros((n_states, 2 * n_states))  # d: state-visitation frequencies
#     d[:, 0] = p_initial
#
#     # 5. iterate for N steps
#     for t in range(1, 2 * n_states):  # longest trajectory: n_states
#
#         # for all states
#         for s_to in range(n_states):
#
#             # sum over nonterminal state-action pairs
#             for s_from, a in product(nonterminal, range(n_actions)):
#                 d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * p_transition[s_from, s_to, a]
#
#     # 6. sum-up frequencies
#     return d.sum(axis=1)