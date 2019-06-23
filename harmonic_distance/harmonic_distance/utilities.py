import tensorflow as tf
import numpy as np
import itertools

LOG_2 = tf.math.log(tf.constant([2.0], dtype=tf.float64))
E = tf.exp(tf.constant(1.0, dtype=tf.float64))

def log2_graph(val):
    return tf.math.log(val) / LOG_2

def parabolic_scale(diffs, c, coeff=E):
    return tf.pow(coeff, -1.0 * (diffs**2 / (2.0 * c**2)))

def reduce_euclid(coords, axis=1):
    out = tf.square(coords)
    out = tf.reduce_sum(out, axis)
    out = tf.sqrt(out)
    return out

def reduce_parabola(coords, axis=1, curves=None):
    out = tf.square(coords)
    if curves is None:
        curves = tf.ones_like(coords, dtype=tf.float64)
    out = out / curves
    out = tf.reduce_sum(out, axis)
    return out

def combinatorial_contour(vec): 
    combos = np.array(list(itertools.combinations(vec, 2)))
    return combos[:, 0] - combos[:, 1]

def get_bases(length):
    return np.apply_along_axis(combinatorial_contour, 1, np.eye(length))[1:] * -1
