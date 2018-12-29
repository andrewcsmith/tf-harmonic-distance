import tensorflow as tf
import numpy as np

def cartesian_graph(a):
    """
    Given at least 2 elements in a, generates the Cartesian product of all
    elements in the list.
    """
    tile_a = tf.expand_dims(tf.tile(tf.expand_dims(a[0], 1), [1, tf.shape(a[1])[0]]), 2)
    tile_b = tf.expand_dims(tf.tile(tf.expand_dims(a[1], 0), [tf.shape(a[0])[0], 1]), 2)
    cart = tf.concat([tile_a, tile_b], axis=2)
    cart = tf.reshape(cart, [-1, 2])
    for c in a[2:]:
        tile_c = tf.tile(tf.expand_dims(c, 1), [1, tf.shape(cart)[0]])
        tile_c = tf.expand_dims(tile_c, 2)
        tile_c = tf.reshape(tile_c, [-1, 1])
        cart = tf.tile(cart, [tf.shape(c)[0], 1])
        cart = tf.concat([tile_c, cart], axis=1)
    return cart

def permutations(a, times=2, name=None):
    """
    Shortcut for generating the Cartesian product of self, using indices so
    that we can work with a small number of elements initially.
    """
    if times > 1:
        options = tf.range(tf.shape(a)[0])
        indices = cartesian_graph([options for _ in range(times)])
        return tf.gather(a, indices, name=name)
    else:
        return a
