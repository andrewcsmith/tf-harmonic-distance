import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_permutations():
    exp = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])
    @tf.function
    def res():
        x = np.array([[0], [1]])
        return hd.cartesian.permutations(x)
    np.testing.assert_equal(exp, res())
