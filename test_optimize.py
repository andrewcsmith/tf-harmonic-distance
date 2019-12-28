import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_stopping_op():
    log_pitches = tf.Variable(tf.random.uniform([64, 2], 0.0, 1.0, dtype=tf.float64))
    pds = np.load("test_data/pds.npy")
    hds = np.load("test_data/hds.npy")
    @tf.function
    def loss():
        loss = hd.optimize.parabolic_loss_function(pds, hds, log_pitches, curves=(0.1, 0.1))
        return loss
    stopping_op = hd.optimize.stopping_op(loss, [log_pitches])
