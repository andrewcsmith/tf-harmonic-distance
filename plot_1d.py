import hd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

c = 0.3
n_points = 1024

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = tf.get_variable("log_pitches", [n_points, 2], dtype=tf.float64)
init_op = tf.global_variables_initializer()

vectors = hd.vectors.space_graph(5, 4, name="vectors")

xs = np.linspace(0.0, 1.0, n_points)

sess.run([init_op])

slice_level = np.full_like(xs, np.log2(1))
coords = np.hstack([xs[:, None], slice_level[:, None]])

with tf.control_dependencies([log_pitches.assign(coords)]):
    y_op = hd.scaled_hd_graph(log_pitches, vectors, c=c)

ys = sess.run(y_op)

plt.plot(xs, ys)
plt.show()
