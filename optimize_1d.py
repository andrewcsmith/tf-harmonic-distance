import hd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

c = 0.1
n_points = 1024
CONVERGENCE_THRESHOLD = 1.0e-16
LEARNING_RATE = 1.0e-3
MAX_ITERS = 100000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = tf.get_variable("log_pitches", [n_points, 1], dtype=tf.float64)
sess.run(tf.global_variables_initializer())

xs = np.linspace(0.0, 1.0, n_points)
inputs = tf.expand_dims(xs, 1)
sess.run(log_pitches.assign(inputs))

vectors = hd.vectors.space_graph(5, 4, bounds=(0.0, 1.0), name="vectors")
stopping_condition_op = hd.optimize.stopping_op(vectors, log_pitches, lr=LEARNING_RATE, ct=CONVERGENCE_THRESHOLD, c=c)

for idx in range(MAX_ITERS):
    if (sess.run(stopping_condition_op)):
        print("Converged at iteration: ", idx)
        out_pitches = sess.run(log_pitches)
        # print(out_pitches * 1200.0)
        break

winners = sess.run(hd.vectors.closest_from_log(log_pitches, vectors))
ratios = np.apply_along_axis(lambda row: hd.vectors.to_ratio(row), 1, winners)

all_possible_pitches = set()
np.apply_along_axis(lambda ratio: all_possible_pitches.add((ratio[0], ratio[1])), 1, ratios)

print(len(all_possible_pitches))
print(sorted(all_possible_pitches, key=lambda r: r[0] / r[1]))
