import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

PDS = np.load("test_data/pds.npy")
HDS = np.load("test_data/hds.npy")

def test_stopping_op():
    log_pitches = tf.Variable(tf.random.uniform([64, 2], 0.0, 1.0, dtype=tf.float64))
    @tf.function
    def loss():
        loss = hd.optimize.parabolic_loss_function(PDS, HDS, log_pitches, curves=(0.1, 0.1))
        return loss
    stopping_op = hd.optimize.stopping_op(loss, [log_pitches])

def test_parabolic_loss_function_2d():
    log_pitches = np.log2([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    vs = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2)
    @tf.function
    def loss(x, c=0.01):
        return hd.optimize.parabolic_loss_function(vs.pds, vs.hds, x, curves=(0.01, 0.01))
    exp = np.log2([1.0, 6, 2, 3, 4]) * 2
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)

def test_parabolic_loss_function_1d():
    log_pitches = np.log2([1.0, 1.5, 2.0, 3.0, 4.0])
    vs = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=1)
    @tf.function
    def loss(x, c=0.01):
        return hd.optimize.parabolic_loss_function(vs.pds, vs.hds, x, curves=(0.01))
    exp = np.log2([1.0, 6, 2, 3, 4])
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)
