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

def test_transform_to_unit_circle_3d():
    pds = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
    ])
    sqrt_third = np.sqrt(1.0 / 3.0)
    exp = np.array([
        [0.0, 1.0, 0.0],
        [SQRT_OF_HALF, SQRT_OF_HALF, 0.0],
        [sqrt_third, sqrt_third, sqrt_third],
        [0.5, 0.0, 0.0],
    ])
    res = hd.utilities.transform_to_unit_circle(pds)
    np.testing.assert_almost_equal(exp, res)

def test_transform_to_unit_circle_1d_is_identity():
    pds = np.array([[0.0], [0.3], [1.7]])
    res = hd.utilities.transform_to_unit_circle(pds)
    np.testing.assert_almost_equal(pds, res)

def test_transform_to_unit_circle_permutation_equivariance():
    rng = np.random.default_rng(2026)
    pds = rng.uniform(0.0, 4.0, size=(16, 4))
    perm = [2, 0, 3, 1]
    res_permuted_input = hd.utilities.transform_to_unit_circle(pds[:, perm])
    res_permuted_output = hd.utilities.transform_to_unit_circle(pds).numpy()[:, perm]
    np.testing.assert_almost_equal(res_permuted_output, res_permuted_input)

def test_transform_from_unit_circle_round_trip():
    rng = np.random.default_rng(2026)
    for n_dims in range(1, 6):
        pds = rng.uniform(0.0, 4.0, size=(32, n_dims))
        forward = hd.utilities.transform_to_unit_circle(pds)
        np.testing.assert_almost_equal(pds, hd.utilities.transform_from_unit_circle(forward))
        backward = hd.utilities.transform_from_unit_circle(pds)
        np.testing.assert_almost_equal(pds, hd.utilities.transform_to_unit_circle(backward))

def test_transform_from_unit_circle_origin():
    res = hd.utilities.transform_from_unit_circle(np.zeros((2, 3)))
    np.testing.assert_almost_equal(np.zeros((2, 3)), res)

def test_get_bases_2():
    exp = np.array([[1.0]])
    res = hd.utilities.get_bases(2)
    np.testing.assert_almost_equal(exp, res)

def test_get_bases_3():
    exp = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    res = hd.utilities.get_bases(3)
    np.testing.assert_almost_equal(exp, res)
