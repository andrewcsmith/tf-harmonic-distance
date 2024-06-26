import tensorflow as tf
import numpy as np
from . import PRIMES
from . import tenney
from .utilities import log2_graph, E, get_bases, parabolic_scale, reduce_parabola
from .cartesian import permutations

# Create default prime limits for ease of use
PRIME_LIMITS = [5, 5, 3, 3, 2, 1]
PD_BOUNDS = (0.0, 4.0)
HD_LIMIT = 9.0
DIMS = 1

def pd_graph(vectors):
    """
    Calculate the pitch distance (log2 of ratio) of each vector, provided as a
    row of prime factor exponents.
    """
    prime_slice = PRIMES[:vectors.shape[-1]]
    float_ratio = tf.reduce_prod(tf.pow(tf.constant(prime_slice, dtype=tf.float64), vectors), axis=1)
    return log2_graph(float_ratio)

def space_graph(n_primes, n_degrees, bounds=None, name=None):
    return space_graph_altered_permutations(np.full([n_primes], n_degrees), bounds=bounds, name=name)

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
        return tf.cast(vectors, tf.float64)

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
    mins = tf.argmin(tf.abs(log_vectors[:, None, None] - log_pitches[None, :]), axis=0)
    return tf.gather(vectors, mins, axis=0)

def sorted_from_log(log_pitches, vectors, n_returned=1):
    log_vectors = pd_graph(vectors)
    diffs = tf.abs(log_vectors[:, None, None] - log_pitches[None, :])
    sorted_vectors = tf.argsort(diffs, axis=0)
    return tf.gather(vectors, sorted_vectors[:, :n_returned], axis=0)

def to_ratio(vector):
    primes = PRIMES[:vector.shape[-1]]
    num = np.where(vector > 0, vector, np.zeros_like(primes))
    den = np.where(vector < 0, vector, np.zeros_like(primes))
    return np.stack([
        np.product(np.power(primes, num), axis=-1),
        np.product(np.power(primes, np.abs(den)), axis=-1)
    ], -1)

class VectorSpace(tf.Module):
    def __init__(self, *args, **kwargs):
        self.perms = tf.Variable(self.get_perms(**kwargs))
        self.hds = tf.Variable(tenney.hd_aggregate_graph(self.perms))
        self.pds = tf.Variable(tenney.pd_aggregate_graph(self.perms))
        self.two_hds = tf.pow(2.0, self.hds)
    
    def closest_from_log(self, log_pitches):
        diffs = tf.abs(self.pds[:, None] - log_pitches[None, :])
        mins = tf.argmin(tf.reduce_sum(diffs, axis=-1), axis=0)
        return tf.gather(self.perms, mins, axis=0)

    def unique_ratios(self, log_pitches):
        ratios = to_ratio(self.closest_from_log(log_pitches))
        unique = np.log2(np.unique(ratios[:, :, 0] / ratios[:, :, 1]))
        return to_ratio(self.closest_from_log(unique))
    
    def get_perms(self, prime_limits=PRIME_LIMITS, pd_bounds=PD_BOUNDS, hd_limit=HD_LIMIT, dimensions=DIMS):
        vectors = space_graph_altered_permutations(prime_limits, bounds=pd_bounds)
        vectors_hds = tenney.hd_aggregate_graph(tf.cast(vectors[:, None, :], tf.float64))
        self.vectors = tf.boolean_mask(vectors, vectors_hds < hd_limit)
        return permutations(self.vectors, times=dimensions)
