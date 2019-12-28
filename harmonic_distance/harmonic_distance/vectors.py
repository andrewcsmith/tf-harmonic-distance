import tensorflow as tf
from . import PRIMES
from .utilities import *
from .cartesian import permutations

def pd_graph(vectors):
    """
    Calculate the pitch distance (log2 of ratio) of each
    vector, provided as a row of prime factor exponents.
    """
    prime_slice = PRIMES[:vectors.shape[-1]]
    float_ratio = tf.reduce_prod(tf.pow(tf.constant(prime_slice, dtype=tf.float64), vectors), axis=1)
    return log2_graph(float_ratio)

def space_graph(n_primes, n_degrees, bounds=None, name=None):
    vectors = permutations(tf.range(-n_degrees, n_degrees+1, dtype=tf.float64), times=n_primes, name=name)
    if bounds is not None:
        return restrict_bounds(tf.cast(vectors, tf.float64), bounds)
    else:
        return vectors

def restrict_bounds(vectors, bounds):
    pitch_distances = pd_graph(vectors)
    out_of_bounds_mask = tf.logical_and(tf.less_equal(pitch_distances, bounds[1]), tf.greater_equal(pitch_distances, bounds[0]))
    pitch_distances = tf.boolean_mask(pitch_distances, out_of_bounds_mask)
    return tf.boolean_mask(vectors, out_of_bounds_mask)

def space_graph_altered_permutations(limits, bounds=None, name=None):
    """
    This function is similar to space_graph, except it allows us to specify a
    prime limit for every dimension.
    """
    vectors = tf.meshgrid(*[list(range(-i, i+1)) for i in limits], indexing='ij')
    vectors = tf.stack(vectors, axis=-1)
    vectors = tf.reshape(vectors, (-1, tf.shape(limits)[0]))
    if bounds is not None:  
        return restrict_bounds(tf.cast(vectors, tf.float64), bounds)
    else:
        return vectors

def scales_graph(log_pitches, vectors, c=0.05, bounds=None, coeff=E):
    """
    Calculate the scale factor (between 0.0 - 1.0) for each of log_pitches, for
    each dimension.
    """
    pitch_distances = tf.expand_dims(pd_graph(vectors), -1)
    tiled_ones = tf.ones_like(pitch_distances) * -1.0
    bases = get_bases(log_pitches.shape[-1] + 1)
    combinatorial_log_pitches = tf.abs(tf.tensordot(log_pitches, bases, 1))
    combos = tf.tensordot(tiled_ones, combinatorial_log_pitches[None, :, :], 1)
    diffs = tf.abs(tf.add(tf.expand_dims(pitch_distances, 1), combos))
    return parabolic_scale(diffs, c, coeff=coeff)

def closest_from_log(log_pitches, vectors):
    log_vectors = pd_graph(vectors)
    mins = tf.argmin(tf.abs(log_vectors - log_pitches), axis=1)
    return tf.map_fn(lambda m: vectors[m, :], mins, dtype=tf.float64)

def sorted_from_log(log_pitches, vectors, n_returned=1):
    log_vectors = pd_graph(vectors)
    diffs = tf.abs(log_vectors - log_pitches)
    sorted_vectors = tf.contrib.framework.argsort(diffs, axis=1)
    return tf.gather(vectors, sorted_vectors[:, :n_returned])

def to_ratio(vector):
    primes = PRIMES[:vector.shape[0]]
    num = np.where(vector > 0, vector, np.zeros_like(primes))
    den = np.where(vector < 0, vector, np.zeros_like(primes))
    return (
        np.product(np.power(primes, num)),
        np.product(primes ** np.abs(den))
    )