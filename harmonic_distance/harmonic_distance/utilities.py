import tensorflow as tf
import numpy as np
import itertools

LOG_2 = tf.math.log(tf.constant([2.0], dtype=tf.float64))
E = tf.exp(tf.constant(1.0, dtype=tf.float64))

def log2_graph(val):
    return tf.math.log(val) / LOG_2

def transform_to_unit_circle(xys):
    """
    Rescale each row so that its Euclidean magnitude equals its Chebyshev
    (maximum-coordinate) magnitude, preserving direction. In n dimensions this
    compresses the k-fold diagonal by 1/sqrt(k), so local minima along parallel
    voice motion are spaced the same as minima along the axes. The magnitude of
    a transformed chord equals its span (largest interval from the root).
    """
    xys = tf.convert_to_tensor(xys, dtype=tf.float64)
    linf = tf.reduce_max(tf.abs(xys), axis=-1, keepdims=True)
    l2 = tf.sqrt(tf.reduce_sum(tf.square(xys), axis=-1, keepdims=True))
    return xys * tf.math.divide_no_nan(linf, l2)

def transform_from_unit_circle(xys):
    """
    Exact inverse of transform_to_unit_circle: rescale each row so that its
    Chebyshev magnitude equals its Euclidean magnitude, recovering real
    log-pitch coordinates from the polar-transformed space.
    """
    xys = tf.convert_to_tensor(xys, dtype=tf.float64)
    linf = tf.reduce_max(tf.abs(xys), axis=-1, keepdims=True)
    l2 = tf.sqrt(tf.reduce_sum(tf.square(xys), axis=-1, keepdims=True))
    return xys * tf.math.divide_no_nan(l2, linf)

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
