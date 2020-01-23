import tensorflow as tf
import numpy as np
import itertools

LOG_2 = tf.math.log(tf.constant([2.0], dtype=tf.float64))
E = tf.exp(tf.constant(1.0, dtype=tf.float64))

def log2_graph(val):
    return tf.math.log(val) / LOG_2

@tf.function
def transform_to_unit_circle(xys):
    """
    Transform a batch of xy-coordinates to the unit circle
    """
    # Calculate theta in radians
    theta = tf.math.atan(xys[:, 1] / xys[:, 0])
    # Guard against edge case [0.0, 0.0]
    theta = tf.where(tf.math.is_nan(theta), tf.constant(0.0, dtype=tf.float64), theta)
    r = tf.sqrt(tf.reduce_sum(tf.math.square(xys), 1))
    polar_xs = xys[:, 0] * tf.math.cos(theta)
    polar_ys = xys[:, 1] * tf.math.sin(theta)
    polar_xys = tf.stack((polar_xs, polar_ys), 1)
    r = tf.sqrt(r * tf.reduce_max(polar_xys, 1))
    new_x = tf.math.cos(theta) * r
    new_y = tf.math.sin(theta) * r
    return tf.stack((new_x, new_y), 1)

def parabolic_scale(diffs, c, coeff=E):
    return tf.pow(coeff, -1.0 * (diffs**2 / (2.0 * c**2)))

def reduce_euclid(coords, axis=1):
    out = tf.square(coords)
    out = tf.reduce_sum(out, axis)
    out = tf.sqrt(out)
    return out

def reduce_parabola(coords, axis=1, curves=None):
    if curves is None:
        curves = tf.ones_like(coords, dtype=tf.float64)
    return tf.reduce_sum(tf.square(coords) / curves, axis)

def combinatorial_contour(vec): 
    combos = np.array(list(itertools.combinations(vec, 2)))
    return combos[:, 0] - combos[:, 1]

def get_bases(length):
    return np.apply_along_axis(combinatorial_contour, 1, np.eye(length))[1:] * -1
