import os

# The test tensors are tiny, so GPU kernel-launch overhead dominates any
# speedup: VectorSpace construction benchmarks ~3x faster on CPU. Hide the
# GPU before TensorFlow is imported; set HD_TEST_GPU=1 to keep it visible.
if not os.environ.get("HD_TEST_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

devices = tf.config.experimental.get_visible_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

import harmonic_distance as hd


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end gradient-descent convergence tests"
    )


# Session-scoped spaces shared across test modules. VectorSpace is read-only
# after construction (loss/closest_from_log/summary_batch do not mutate), so
# sharing is safe; construction is the dominant cost of the suite.

@pytest.fixture(scope="session")
def vs_31_1d():
    return hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=1, materialize="full")


@pytest.fixture(scope="session")
def vs_31_2d():
    return hd.vectors.VectorSpace(prime_limits=[3, 1], dimensions=2, materialize="full")


@pytest.fixture(scope="session")
def vs_321_1d():
    return hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=1, materialize="full")


@pytest.fixture(scope="session")
def vs_321_2d():
    return hd.vectors.VectorSpace(prime_limits=[3, 2, 1], dimensions=2, materialize="full")


# batch_size=512 does not divide the permutation count, so the batched paths
# still cross several chunk boundaries and hit the short tail chunk, without
# tracing hundreds of unrolled loop iterations as a tiny batch size would.
@pytest.fixture(scope="session")
def vs_321_2d_batched():
    return hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1], dimensions=2, batch_size=512, materialize="none"
    )


@pytest.fixture(scope="session")
def vs_321_2d_polar():
    return hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1], dimensions=2, materialize="full", polar=True
    )


@pytest.fixture(scope="session")
def vs_321_2d_polar_batched():
    return hd.vectors.VectorSpace(
        prime_limits=[3, 2, 1], dimensions=2, batch_size=512, materialize="none", polar=True
    )
