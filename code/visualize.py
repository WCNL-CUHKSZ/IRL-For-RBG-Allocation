"""
Intend to visualize the model weights

"""

import tensorflow as tf
import numpy as np

# input_tensor = tf.keras.Input(shape=[10])
# x = tf.keras.layers.Dense(64)(input_tensor)
# x = tf.keras.layers.Dense(128)(x)
# x = tf.keras.layers.Dense(256)(x)
# x = tf.keras.layers.Dense(1)(x)
#
# model = tf.keras.Model(input_tensor, x)
# model.save('test.h5')

model = tf.keras.models.load_model('test.h5')

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
    update_freq='epoch', profile_batch=0, embeddings_freq=0,
    embeddings_metadata=None)

model.trainable = False
model.compile('adam', 'mse')
model.fit(x=np.random.rand(1,10),
          y=np.array([[1.0]]),
          epochs=1,
          callbacks=[tensorboard])