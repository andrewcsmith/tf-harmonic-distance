import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_to_ratio_single():
    vector = np.array([[[-1., 1., 0.]]])
    exp = np.array([[[3., 2.]]])
    res = hd.vectors.to_ratio(vector)
    np.testing.assert_array_equal(exp, res)

def test_to_ratio_batch():
    vectors = np.array([[[-1., 1., 0.], [-2., 0., 1.]]])
    exp = np.array([[[3., 2.], [5., 4.]]])
    res = hd.vectors.to_ratio(vectors)
    np.testing.assert_array_equal(exp, res)

def test_to_ratio_2d_2b():
    vectors = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    exp = np.array([[[3., 2.], [5., 4.]], [[4., 3.], [6., 5.]]])
    res = hd.vectors.to_ratio(vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_1d():
    log_pitches = np.array([[702.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_1d_2b():
    log_pitches = np.array([[702.], [386.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]], [[-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d():
    log_pitches = np.array([[702., 386.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d_2b():
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_2d_2b():
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2)
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = vs.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_1d():
    log_pitches = np.array([[702.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]]])
    res = vs.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)
