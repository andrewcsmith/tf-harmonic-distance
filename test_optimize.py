import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_stopping_op():
    # Load in test data and initiatlize pitches
    pds = np.load("test_data/pds.npy")
    hds = np.load("test_data/hds.npy")
    log_pitches = tf.Variable(np.log2([4.1/3.0, 1.5]), dtype=tf.float64)

    @tf.function
    def get_loss():
        return hd.optimize.parabolic_loss_function(pds, hds, log_pitches, curves=(0.02, 0.02))
    
    def take_step():
        return hd.optimize.stopping_op(get_loss, [log_pitches])
    
    print(take_step())
    print(log_pitches)
    print(take_step())
    print(log_pitches)
    for i in range(1000):
        take_step()
    print(log_pitches)

test_stopping_op()
