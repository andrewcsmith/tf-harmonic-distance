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

def test_closest_from_log_1d(vs_321_1d):
    log_pitches = np.array([[702.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs_321_1d.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_1d_2b(vs_321_1d):
    log_pitches = np.array([[702.], [386.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.]], [[-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs_321_1d.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d(vs_321_1d):
    log_pitches = np.array([[702., 386.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs_321_1d.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d_2b(vs_321_1d):
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs_321_1d.vectors)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_2d_2b(vs_321_2d):
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = vs_321_2d.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_1d(vs_321_1d):
    log_pitches = np.array([[702.]]) / 1200.0
    exp = np.array([[[-1., 1., 0.]]])
    res = vs_321_1d.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_batched_summaries_match_materialized(vs_31_2d):
    batched = hd.vectors.VectorSpace(
        prime_limits=[3, 1],
        dimensions=2,
        batch_size=7,
        materialize="none",
    )
    pds = []
    hds = []
    for start in range(0, batched.permutation_count, batched.batch_size):
        chunk_pds, chunk_hds = batched.summary_batch(start, batched.batch_size)
        pds.append(chunk_pds)
        hds.append(chunk_hds)
    np.testing.assert_allclose(vs_31_2d.pds, tf.concat(pds, axis=0))
    np.testing.assert_allclose(vs_31_2d.hds, tf.concat(hds, axis=0))

def test_vectorspace_summary_materialization_matches_full(vs_31_2d):
    summaries = hd.vectors.VectorSpace(
        prime_limits=[3, 1],
        dimensions=2,
        batch_size=7,
        materialize="summaries",
    )
    assert summaries.materialized
    assert summaries.materialize_mode == "summaries"
    assert not summaries.has_perms
    assert not hasattr(summaries, "perms")
    np.testing.assert_allclose(vs_31_2d.pds, summaries.pds)
    np.testing.assert_allclose(vs_31_2d.hds, summaries.hds)

def test_vectorspace_summary_materialization_batch_size_is_memory_capped():
    vs = hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1],
        dimensions=3,
        batch_size=1_000_000,
        materialize="none",
    )
    summary_bytes = vs.permutation_count * (vs.dimensions + 1) * np.dtype(np.float64).itemsize
    permutation_row_bytes = vs.dimensions * vs.n_primes * np.dtype(np.float64).itemsize
    expected = summary_bytes // permutation_row_bytes
    assert vs.summary_materialization_batch_size() == expected
    assert vs.summary_materialization_batch_size() < vs.batch_size

def test_vectorspace_batched_closest_from_log_matches_materialized(vs_321_2d, vs_321_2d_batched):
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    np.testing.assert_array_equal(
        vs_321_2d.closest_from_log(log_pitches),
        vs_321_2d_batched.closest_from_log(log_pitches),
    )

def test_vectorspace_auto_can_skip_materialization():
    vs = hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1],
        dimensions=2,
        materialize="auto",
        materialize_limit=1,
    )
    assert not vs.materialized
    assert vs.materialize_mode == "none"
    assert not hasattr(vs, "perms")
    assert vs.permutation_count > 1

def test_vectorspace_polar_transforms_pds(vs_321_2d, vs_321_2d_polar):
    np.testing.assert_allclose(
        hd.utilities.transform_to_unit_circle(vs_321_2d.pds), vs_321_2d_polar.pds
    )
    np.testing.assert_allclose(vs_321_2d.hds, vs_321_2d_polar.hds)

def test_vectorspace_polar_summaries_match_full():
    full = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2, materialize="full", polar=True)
    summaries = hd.vectors.VectorSpace(
        prime_limits=[3, 1],
        dimensions=2,
        batch_size=7,
        materialize="summaries",
        polar=True,
    )
    np.testing.assert_allclose(full.pds, summaries.pds)
    np.testing.assert_allclose(full.hds, summaries.hds)

def test_vectorspace_polar_batched_loss_matches_materialized(vs_321_2d_polar, vs_321_2d_polar_batched):
    log_pitches = hd.utilities.transform_to_unit_circle(np.log2([[5/4, 3/2], [4/3, 3/2]]))
    exp = vs_321_2d_polar.loss(log_pitches, curves=(0.001, 0.001))
    res = vs_321_2d_polar_batched.loss(log_pitches, curves=(0.001, 0.001))
    np.testing.assert_allclose(exp, res)

def test_vectorspace_polar_batched_gradient_matches_materialized(vs_321_2d_polar, vs_321_2d_polar_batched):
    log_pitches = tf.Variable(
        hd.utilities.transform_to_unit_circle(np.log2([[5/4 * 1.01, 3/2 * 0.99]]))
    )
    with tf.GradientTape() as tape:
        exp_loss = tf.reduce_sum(vs_321_2d_polar.loss(log_pitches, curves=(0.01, 0.01)))
    exp_grad = tape.gradient(exp_loss, log_pitches)
    with tf.GradientTape() as tape:
        res_loss = tf.reduce_sum(vs_321_2d_polar_batched.loss(log_pitches, curves=(0.01, 0.01)))
    res_grad = tape.gradient(res_loss, log_pitches)
    np.testing.assert_allclose(exp_grad, res_grad)

def test_vectorspace_rejects_boolean_materialize():
    with pytest.raises(ValueError, match="materialize must be"):
        hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=1, materialize=True)
