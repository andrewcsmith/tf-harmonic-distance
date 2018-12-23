import tensorflow as tf
import itertools
import numpy as np
from .cartesian import * 
from . import PRIMES
from .utilities import *
from .vectors import *

from tensorflow.python import debug as tf_debug

def exploded_hd_graph(n_primes, vecs):
    hd = tf.pow(tf.constant(PRIMES[:n_primes], dtype=tf.float64), vecs)
    return log2_graph(hd)
    
def hd_graph(n_primes, vecs):
    """
    Gets the harmonic distance of each row in a series of vecs
    """
    hds = exploded_hd_graph(n_primes, vecs)
    return tf.reduce_sum(tf.abs(hds), axis=1)

def hd_aggregate_graph(n_primes, aggregates):
    bases = get_bases(aggregates.shape[1] + 1)
    def get_hd_sum(hd):
        hd = tf.transpose(hd)
        hd = tf.tensordot(hd, bases, 1)
        hd = tf.abs(hd)
        hd = tf.reduce_sum(hd, axis=0)
        hd = tf.reduce_sum(hd, axis=0)
        return hd
    hds = exploded_hd_graph(n_primes, aggregates)
    return tf.map_fn(get_hd_sum, hds)

def scaled_hd_graph(log_pitches, vectors, c=0.05, coeff=E):
    # scales_graph should take the euclidean distance in 2 dimensions
    # TODO: We need to take the scales_graph of each combo in the basis space,
    # not of the raw log_pitches, and then take the euclidean distance of that
    scales = scales_graph(log_pitches, vectors, c=c, coeff=coeff) + 2.0e-32
    n_primes = vectors.shape[-1]
    # returns the hd of every vector in the list: [vectors.shape[0]]
    # A: [n_vectors]
    hds = hd_graph(n_primes, vectors)
    hds = hds + 1.0
    # n_pairs = the number of combinatorial pairs 
    # T: [n_vectors, n_vectors, n_pairs]
    hds = tf.tile(hds[:, None, None], [1, scales.shape[1], scales.shape[2]])
    hds = hds * tf.reciprocal(scales)
    hds = tf.reduce_min(hds, axis=0)
    hds = tf.reduce_mean(hds, axis=1)
    hds = hds - 1.0
    return hds

