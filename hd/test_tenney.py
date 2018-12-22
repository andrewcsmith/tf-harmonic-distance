import pytest

import hd
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def test_harmonic_distance():
    perfect_fifth = np.array([[-1.0, 1.0, 0.0]])
    exp = np.array([2.584962500721156])
    res = sess.run(hd.hd_graph(3, perfect_fifth))
    np.testing.assert_almost_equal(exp, res)

def test_harmonic_distances():
    two_intervals = np.array([
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0]
    ])
    exp = np.array([4.321928094887363, 2.584962500721156])
    res = sess.run(hd.hd_graph(3, two_intervals))
    np.testing.assert_almost_equal(exp, res)

def test_harmonic_distance_aggregate():
    triad = np.array([[
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0]
    ]])
    exp = np.array([11.813781191217037])
    res = sess.run(hd.hd_aggregate_graph(3, triad))
    np.testing.assert_almost_equal(exp, res)

def test_scaled_hd_graph_off():
    log_pitches = np.array([[7.0]]) / 12.0
    vectors = hd.vectors.space_graph(4, 3)
    # It's one higher because it needs to be scaled
    exp = np.array([[3.586866037637317]])
    res = sess.run([hd.scaled_hd_graph(log_pitches, vectors, c=0.05)])
    np.testing.assert_almost_equal(exp, res)

def test_scaled_hd_graph_2d():
    log_pitches = np.array([[7.0, 10.0]]) / 12.0
    vectors = hd.vectors.space_graph(4, 3)
    exp = np.array([[15.549381852281275]])
    res = sess.run([hd.scaled_hd_graph(log_pitches, vectors, c=0.1)])
    np.testing.assert_almost_equal(exp, res)
