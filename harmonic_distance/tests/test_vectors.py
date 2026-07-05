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

def test_closest_from_log_1d():
    log_pitches = np.array([[702.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_1d_2b():
    log_pitches = np.array([[702.], [386.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]], [[-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d():
    log_pitches = np.array([[702., 386.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_closest_from_log_2d_2b():
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = hd.vectors.closest_from_log(log_pitches, vs.vectors)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_2d_2b():
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2)
    exp = np.array([[[-1., 1., 0.], [-2., 0., 1.]], [[2., -1., 0.], [1., 1., -1.]]])
    res = vs.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_closest_from_log_1d():
    log_pitches = np.array([[702.]]) / 1200.0
    vs = hd.vectors.VectorSpace(prime_limits=[3, 2, 1])
    exp = np.array([[[-1., 1., 0.]]])
    res = vs.closest_from_log(log_pitches)
    np.testing.assert_array_equal(exp, res)

def test_vectorspace_batched_summaries_match_materialized():
    materialized = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2, materialize="full")
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
    np.testing.assert_allclose(materialized.pds, tf.concat(pds, axis=0))
    np.testing.assert_allclose(materialized.hds, tf.concat(hds, axis=0))

def test_vectorspace_summary_materialization_matches_full():
    full = hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2, materialize="full")
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
    np.testing.assert_allclose(full.pds, summaries.pds)
    np.testing.assert_allclose(full.hds, summaries.hds)

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

def test_vectorspace_batched_closest_from_log_matches_materialized():
    log_pitches = np.array([[702., 386.], [498., 315.]]) / 1200.0
    materialized = hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2, materialize="full")
    batched = hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1],
        dimensions=2,
        batch_size=11,
        materialize="none",
    )
    np.testing.assert_array_equal(materialized.closest_from_log(log_pitches), batched.closest_from_log(log_pitches))

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

def test_vectorspace_rejects_boolean_materialize():
    with pytest.raises(ValueError, match="materialize must be"):
        hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=1, materialize=True)
