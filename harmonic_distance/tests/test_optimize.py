import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

def test_parabolic_loss_function_2d():
    log_pitches = np.log2([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    vs = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2)
    @tf.function
    def loss(x, c=0.01):
        return hd.optimize.parabolic_loss_function(vs.pds, vs.hds, x, curves=(0.01, 0.01))
    exp = np.log2([1.0, 6, 2, 3, 4])
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)

def test_loss_function_triad_2d():
    log_pitches = np.log2([[5/4, 3/2], [6/5, 3/2], [3/2, 3/2]])
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2)
    # Manually calculate the Tenney HDs for the combinatorial triadic relationships
    exp = np.array([11.813781191217037, 11.813781191217037, 5.169925001442312]) / 2.0
    res = hd.optimize.parabolic_loss_function(vs.pds, vs.hds, log_pitches, curves=(0.001, 0.001))
    np.testing.assert_almost_equal(exp, res)

def test_parabolic_loss_function_1d():
    log_pitches = np.log2([[[1.0], [1.5], [2.0], [3.0], [4.0]]])
    vs = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=1)
    @tf.function
    def loss(x):
        return hd.optimize.parabolic_loss_function(vs.pds, vs.hds, x, curves=[0.01])
    exp = np.log2([1.0, 6, 2, 3, 4])
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_loss_function_2d():
    minimizer = hd.optimize.Minimizer(dimensions=2, prime_limits=[3, 2, 1], c=0.001)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2]]))
    exp = np.array([11.813781191217037]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_loss_function_2d_b2():
    minimizer = hd.optimize.Minimizer(dimensions=2, prime_limits=[3, 2, 1], batch_size=2, c=0.001)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_vectorspace_batched_loss_matches_materialized():
    log_pitches = np.log2([[5/4, 3/2], [4/3, 3/2]])
    materialized = hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2, materialize="full")
    batched = hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1],
        dimensions=2,
        batch_size=13,
        materialize="none",
    )
    exp = hd.optimize.parabolic_loss_function(
        materialized.pds,
        materialized.hds,
        log_pitches,
        curves=(0.001, 0.001),
    )
    res = batched.loss(log_pitches, curves=(0.001, 0.001))
    np.testing.assert_allclose(exp, res)

def test_minimizer_uses_batched_vectorspace_loss():
    vs = hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1],
        dimensions=2,
        batch_size=13,
        materialize="none",
    )
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=0.001, vs=vs)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_1d():
    minimizer = hd.optimize.Minimizer(dimensions=1, prime_limits=[3, 2, 2, 1], convergence_threshold=1.0e-4)
    minimizer.log_pitches.assign([[4/12]])
    minimizer.minimize()
    exp = np.log2([[5/4]])
    np.testing.assert_almost_equal(exp, minimizer.log_pitches.numpy(), 3)

def test_minimizer_2d():
    minimizer = hd.optimize.Minimizer(dimensions=2, prime_limits=[3, 2, 2, 1], convergence_threshold=1.0e-4)
    minimizer.log_pitches.assign([[4/12, 7/12]])
    minimizer.minimize()
    exp = np.log2([[5/4, 3/2]])
    np.testing.assert_almost_equal(exp, minimizer.log_pitches.numpy(), 3)

def test_minimizer_2d_batched_active_rows():
    minimizer = hd.optimize.Minimizer(
        dimensions=2,
        prime_limits=[3, 2, 2, 1],
        batch_size=3,
        convergence_threshold=1.0e-4,
    )
    inactive = [11/12, 1/12]
    minimizer.log_pitches.assign([
        [4/12, 7/12],
        [5/12, 7/12],
        inactive,
    ])
    minimizer.set_active_count(2)
    minimizer.minimize()
    active_exp = np.log2([
        [5/4, 3/2],
        [4/3, 3/2],
    ])
    np.testing.assert_almost_equal(active_exp, minimizer.log_pitches.numpy()[:2], 3)
    np.testing.assert_almost_equal([inactive], minimizer.log_pitches.numpy()[2:], 12)

def test_minimizer_stopping_op_ignores_inactive_rows():
    minimizer = hd.optimize.Minimizer(
        dimensions=1,
        prime_limits=[3, 1],
        batch_size=2,
        convergence_threshold=1.0e-4,
    )
    minimizer.log_pitches.assign(np.log2([[1.0], [1.4]]))
    minimizer.set_active_count(1)
    assert not minimizer.stopping_op().numpy()
