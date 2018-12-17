import tensorflow as tf
import numpy as np
import itertools

LOG_2 = tf.log(tf.constant([2.0], dtype=tf.float64))

def log2_graph(val):
    return tf.log(val) / LOG_2

def parabolic_scale(diffs, c):
    return tf.exp(-1.0 * (diffs**2 / (2.0 * c**2)))

def combinatorial_contour(vec): 
    combos = np.array(list(itertools.combinations(vec, 2)))
    return combos[:, 0] - combos[:, 1]

def get_bases(length):
    return np.apply_along_axis(combinatorial_contour, 1, np.eye(length))[1:] * -1
