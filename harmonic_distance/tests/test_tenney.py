import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

FIFTH = np.array([[-1.0, 1.0, 0.0]])
FOURTH = np.array([[2.0, -1.0, 0.0]])
UNISON_FIFTH = np.array([
    [0.0, 0.0, 0.0],
    [-1.0, 1.0, 0.0]
    ])
TRIAD = np.array([
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0]
    ])

def test_harmonic_distance():
    exp = np.array([2.584962500721156])
    @tf.function
    def res():
        return hd.hd_graph(FIFTH)
    np.testing.assert_almost_equal(exp, res())

def test_harmonic_distances():
    exp = np.array([4.321928094887363, 2.584962500721156])
    @tf.function
    def res():
        return hd.hd_graph(TRIAD)
    np.testing.assert_almost_equal(exp, res())

def test_hd_root_valence():
    exp = np.array([-1.584962500721156])
    @tf.function
    def res():
        return hd.hd_root_valence(FOURTH)
    np.testing.assert_almost_equal(exp, res())

def test_harmonic_distance_aggregate_2d():
    exp = np.array([2.584962500721156])
    @tf.function
    def res():
        return hd.hd_aggregate_graph(UNISON_FIFTH[None, :, :])
    np.testing.assert_almost_equal(exp, res())

def test_harmonic_distance_aggregate():
    exp = np.array([11.813781191217037])
    @tf.function
    def res():
        return hd.hd_aggregate_graph(TRIAD[None, :, :])
    np.testing.assert_almost_equal(exp, res())

def test_scaled_hd_graph_off():
    log_pitches = np.array([[7.0]]) / 12.0
    vectors = hd.vectors.space_graph(4, 3)
    # It's one higher because it needs to be scaled
    exp = np.array([2.586866037637317])
    @tf.function
    def res():
        return hd.scaled_hd_graph(log_pitches, vectors, c=0.05)
    np.testing.assert_almost_equal(exp, res())

def test_scaled_hd_graph_2d():
    log_pitches = np.array([[7.0, 10.0]]) / 12.0
    vectors = hd.vectors.space_graph(4, 3)
    exp = np.array([4.1831273])
    @tf.function
    def res():
        return hd.scaled_hd_graph(log_pitches, vectors, c=0.1)
    np.testing.assert_almost_equal(exp, res())
