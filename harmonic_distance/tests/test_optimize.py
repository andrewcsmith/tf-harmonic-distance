import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd
import numpy as np

class ExplodingVectorSpace:
    def loss(self, log_pitches, curves=None):
        raise AssertionError("loss should not be evaluated during Minimizer construction")

def test_parabolic_loss_function_2d(vs_31_2d):
    log_pitches = np.log2([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    @tf.function
    def loss(x, c=0.01):
        return hd.optimize.parabolic_loss_function(vs_31_2d.pds, vs_31_2d.hds, x, curves=(0.01, 0.01))
    exp = np.log2([1.0, 6, 2, 3, 4])
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)

def test_loss_function_triad_2d(vs_321_2d):
    log_pitches = np.log2([[5/4, 3/2], [6/5, 3/2], [3/2, 3/2]])
    # Manually calculate the Tenney HDs for the combinatorial triadic relationships
    exp = np.array([11.813781191217037, 11.813781191217037, 5.169925001442312]) / 2.0
    res = hd.optimize.parabolic_loss_function(vs_321_2d.pds, vs_321_2d.hds, log_pitches, curves=(0.001, 0.001))
    np.testing.assert_almost_equal(exp, res)

def test_parabolic_loss_function_1d(vs_31_1d):
    log_pitches = np.log2([[[1.0], [1.5], [2.0], [3.0], [4.0]]])
    @tf.function
    def loss(x):
        return hd.optimize.parabolic_loss_function(vs_31_1d.pds, vs_31_1d.hds, x, curves=[0.01])
    exp = np.log2([1.0, 6, 2, 3, 4])
    res = loss(log_pitches)
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_loss_function_2d(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, c=0.001, vs=vs_321_2d)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2]]))
    exp = np.array([11.813781191217037]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_init_does_not_evaluate_loss():
    hd.optimize.Minimizer(dimensions=1, batch_size=1, vs=ExplodingVectorSpace())

def test_minimizer_loss_function_2d_b2(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=0.001, vs=vs_321_2d)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_vectorspace_batched_loss_matches_materialized(vs_321_2d, vs_321_2d_batched):
    log_pitches = np.log2([[5/4, 3/2], [4/3, 3/2]])
    exp = hd.optimize.parabolic_loss_function(
        vs_321_2d.pds,
        vs_321_2d.hds,
        log_pitches,
        curves=(0.001, 0.001),
    )
    res = vs_321_2d_batched.loss(log_pitches, curves=(0.001, 0.001))
    np.testing.assert_allclose(exp, res)

def test_minimizer_uses_batched_vectorspace_loss(vs_321_2d_batched):
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=0.001, vs=vs_321_2d_batched)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

def test_minimizer_max_iters_caps_steps(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1, batch_size=1, max_iters=5, convergence_threshold=1.0e-300, vs=vs_31_1d
    )
    minimizer.log_pitches.assign([[np.log2(3/2) + 0.01]])
    minimizer.minimize()
    assert int(minimizer.step.numpy()) == 5

def test_minimizer_zero_max_iters_runs_until_convergence(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1, batch_size=1, max_iters=0, convergence_threshold=1.0e-5, vs=vs_31_1d
    )
    minimizer.log_pitches.assign([[np.log2(3/2) + 0.001]])
    minimizer.minimize()
    assert not bool(minimizer.stopping_op().numpy())  # converged, not capped
    assert int(minimizer.step.numpy()) > 0

def test_minimizer_none_max_iters_means_unlimited(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1, batch_size=1, max_iters=None, convergence_threshold=1.0e-5, vs=vs_31_1d
    )
    assert int(minimizer.max_iters.numpy()) == 0

def test_minimizer_set_max_iters_updates_after_tracing(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1, batch_size=1, max_iters=3, convergence_threshold=1.0e-300, vs=vs_31_1d
    )
    minimizer.log_pitches.assign([[np.log2(3/2) + 0.01]])
    minimizer.minimize()
    assert int(minimizer.step.numpy()) == 3
    minimizer.set_max_iters(6)
    minimizer.minimize()
    assert int(minimizer.step.numpy()) == 6

def test_minimizer_set_max_iters_rejects_negative(vs_31_1d):
    minimizer = hd.optimize.Minimizer(dimensions=1, batch_size=1, vs=vs_31_1d)
    with pytest.raises(ValueError, match="max_iters must be >= 0"):
        minimizer.set_max_iters(-1)

def test_minimizer_per_voice_curves_match_per_row_scalar_losses(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=1.0, vs=vs_321_2d)
    minimizer.set_curves([0.001, 0.01])
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp0 = vs_321_2d.loss(np.log2([[5/4, 3/2]]), curves=(0.001, 0.001))
    exp1 = vs_321_2d.loss(np.log2([[4/3, 3/2]]), curves=(0.01, 0.01))
    np.testing.assert_allclose([exp0[0], exp1[0]], minimizer.loss())

def test_minimizer_set_curves_scalar_matches_constructor_c(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=1.0, vs=vs_321_2d)
    minimizer.set_curves(0.001)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    np.testing.assert_almost_equal(exp, minimizer.loss())

def test_minimizer_set_curves_updates_after_tracing(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=1.0, vs=vs_321_2d)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    before = minimizer.loss()
    minimizer.set_curves([0.001, 0.01])
    after = minimizer.loss()
    assert not np.allclose(before, after)
    exp0 = vs_321_2d.loss(np.log2([[5/4, 3/2]]), curves=(0.001, 0.001))
    exp1 = vs_321_2d.loss(np.log2([[4/3, 3/2]]), curves=(0.01, 0.01))
    np.testing.assert_allclose([exp0[0], exp1[0]], after)

def test_minimizer_set_curves_rejects_bad_values(vs_31_1d):
    minimizer = hd.optimize.Minimizer(dimensions=1, batch_size=2, vs=vs_31_1d)
    with pytest.raises(ValueError, match="one value per voice"):
        minimizer.set_curves([0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="positive and finite"):
        minimizer.set_curves(0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        minimizer.set_curves([0.01, -0.01])

def test_vectorspace_batched_loss_per_voice_curves_matches_materialized(vs_321_2d, vs_321_2d_batched):
    log_pitches = np.log2([[5/4, 3/2], [4/3, 3/2]])
    curves = np.array([[0.001], [0.01]])
    exp = vs_321_2d.loss(log_pitches, curves=curves)
    res = vs_321_2d_batched.loss(log_pitches, curves=curves)
    np.testing.assert_allclose(exp, res)

def test_minimizer_uses_loaded_vectorspace_loss(tmp_path, vs_321_2d):
    path = tmp_path / "vs.npz"
    vs_321_2d.save(path)
    loaded = hd.vectors.VectorSpace.load(path)
    minimizer = hd.optimize.Minimizer(dimensions=2, batch_size=2, c=0.001, vs=loaded)
    minimizer.log_pitches.assign(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = np.array([11.813781191217037, 12.339850002884624]) / 2.0
    res = minimizer.loss()
    np.testing.assert_almost_equal(exp, res)

@pytest.mark.slow
def test_minimizer_1d(vs_321_1d):
    minimizer = hd.optimize.Minimizer(dimensions=1, convergence_threshold=1.0e-4, vs=vs_321_1d)
    minimizer.log_pitches.assign([[4/12]])
    minimizer.minimize()
    exp = np.log2([[5/4]])
    np.testing.assert_almost_equal(exp, minimizer.log_pitches.numpy(), 3)

@pytest.mark.slow
def test_minimizer_2d(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, convergence_threshold=1.0e-4, vs=vs_321_2d)
    minimizer.log_pitches.assign([[4/12, 7/12]])
    minimizer.minimize()
    exp = np.log2([[5/4, 3/2]])
    np.testing.assert_almost_equal(exp, minimizer.log_pitches.numpy(), 3)

@pytest.mark.slow
def test_minimizer_2d_batched_active_rows(vs_321_2d):
    minimizer = hd.optimize.Minimizer(
        dimensions=2,
        batch_size=3,
        convergence_threshold=1.0e-4,
        vs=vs_321_2d,
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

@pytest.mark.slow
def test_minimizer_active_mask_allows_holes(vs_321_2d):
    minimizer = hd.optimize.Minimizer(
        dimensions=2,
        batch_size=4,
        convergence_threshold=1.0e-4,
        vs=vs_321_2d,
    )
    inactive = [11/12, 1/12]
    minimizer.log_pitches.assign([
        [4/12, 7/12],
        [5/12, 7/12],
        inactive,
        [8/12, 7/12],
    ])
    minimizer.set_active_mask([1, 1, 0, 1])
    minimizer.minimize()
    contiguous_exp = np.log2([
        [5/4, 3/2],
        [4/3, 3/2],
    ])
    result = minimizer.log_pitches.numpy()
    np.testing.assert_almost_equal(contiguous_exp, result[[0, 1]], 3)
    np.testing.assert_almost_equal([inactive], result[2:3], 12)
    assert abs(result[3, 0] - 8/12) > 0.01

def test_minimizer_real_log_pitches_round_trip_polar(vs_321_2d_polar):
    minimizer = hd.optimize.Minimizer(dimensions=2, vs=vs_321_2d_polar)
    real = np.log2([[5/4, 3/2]])
    minimizer.set_real_log_pitches(real)
    transformed = hd.utilities.transform_to_unit_circle(real)
    np.testing.assert_almost_equal(transformed, minimizer.log_pitches.numpy())
    np.testing.assert_almost_equal(real, minimizer.real_log_pitches().numpy())

def test_minimizer_real_log_pitches_identity_without_polar(vs_321_2d):
    minimizer = hd.optimize.Minimizer(dimensions=2, vs=vs_321_2d)
    real = np.log2([[5/4, 3/2]])
    minimizer.set_real_log_pitches(real)
    np.testing.assert_almost_equal(real, minimizer.log_pitches.numpy())
    np.testing.assert_almost_equal(real, minimizer.real_log_pitches().numpy())

@pytest.mark.slow
def test_minimizer_polar_2d_converges_to_real_ratios(vs_321_2d_polar):
    minimizer = hd.optimize.Minimizer(
        dimensions=2,
        convergence_threshold=1.0e-4,
        vs=vs_321_2d_polar,
    )
    minimizer.set_real_log_pitches([[4/12, 7/12]])
    minimizer.minimize()
    exp = np.log2([[5/4, 3/2]])
    np.testing.assert_almost_equal(exp, minimizer.real_log_pitches().numpy(), 3)

def test_minimizer_stopping_op_ignores_inactive_rows(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1,
        batch_size=2,
        convergence_threshold=1.0e-4,
        vs=vs_31_1d,
    )
    minimizer.log_pitches.assign(np.log2([[1.0], [1.4]]))
    minimizer.set_active_count(1)
    assert not minimizer.stopping_op().numpy()

def test_minimizer_convergence_threshold_can_change_after_tracing(vs_31_1d):
    minimizer = hd.optimize.Minimizer(
        dimensions=1,
        convergence_threshold=1.0e9,
        vs=vs_31_1d,
    )
    minimizer.log_pitches.assign(np.log2([[1.4]]))
    assert not minimizer.stopping_op().numpy()

    minimizer.set_convergence_threshold(1.0e-30)
    assert minimizer.stopping_op().numpy()
