import tensorflow as tf
import numpy as np

def permutations(a, times=2, name=None):
    """
    Shortcut for generating the Cartesian product of self, using indices so
    that we can work with a small number of elements initially.
    """
    if times > 1:
        options = tf.range(tf.shape(a)[0])
        indices = tf.stack(tf.meshgrid(*[options for _ in range(times)], indexing='ij'), axis=-1)
        indices = tf.reshape(indices, (-1, times))
        return tf.gather(a, indices, name=name)
    else:
        return a
