import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_to_ratio_single():
    vector = np.array([-1., 1., 0.])
    exp = np.array([[3., 2.]])
    res = hd.vectors.to_ratio(vector)
    np.testing.assert_array_equal(exp, res)

def test_to_ratio_batch():
    vectors = np.array([[-1., 1., 0.], [-2., 0., 1.]])
    exp = np.array([[3., 2.], [5., 4.]])
    res = hd.vectors.to_ratio(vectors)
    np.testing.assert_array_equal(exp, res)
