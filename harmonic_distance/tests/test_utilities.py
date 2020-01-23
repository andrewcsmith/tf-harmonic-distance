import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

SQRT_OF_HALF = np.sqrt(0.5)

def test_transform_to_unit_circle():
    pds = np.array([
        [0.0, 0.0],
        [1.0, 1.0], 
        [0.0, 1.0], 
        [1.0, 0.0], 
        [0.5, 0.5], 
        [0.5, 0.0], 
        [0.0, 0.5]
        ])
    exp = np.array([
        [0.0, 0.0],
        [SQRT_OF_HALF, SQRT_OF_HALF], 
        [0.0, 1.0], 
        [1.0, 0.0], 
        [SQRT_OF_HALF*0.5, SQRT_OF_HALF*0.5], 
        [0.5, 0.0], 
        [0.0, 0.5]
        ])
    res = hd.utilities.transform_to_unit_circle(pds)
    np.testing.assert_almost_equal(exp, res)
