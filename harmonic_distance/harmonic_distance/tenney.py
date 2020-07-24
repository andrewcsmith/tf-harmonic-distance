import tensorflow as tf
import itertools
import numpy as np
from . import PRIMES
from .utilities import get_bases, log2_graph, E
from .vectors import scales_graph, pd_graph

def exploded_hd_graph(vecs):
    n_primes = vecs.shape[-1]
    hd = tf.pow(tf.constant(PRIMES[:n_primes], dtype=tf.float64), vecs)
    return log2_graph(hd)
    
def hd_graph(vecs):
    """
    Gets the harmonic distance of each row in a series of vecs
    """
    hds = exploded_hd_graph(vecs)
    return tf.reduce_sum(tf.abs(hds), axis=1)

def hd_root_valence(vecs):
    """
    Returns the "signed" pitch class harmonic root distance. Intuitively, this
    tells us how "powerful" the interval's valence is; more ambiguous intervals
    will give lower absolute values for this.
    """
    return tf.reduce_sum(exploded_hd_graph(vecs)[:, 1:], axis=1)

def hd_aggregate_graph(aggregates):
    bases = get_bases(aggregates.shape[1] + 1)
    def get_hd_sum(hd):
        hd = tf.transpose(hd)
        hd = tf.tensordot(hd, bases, 1)
        hd = tf.abs(hd)
        hd = tf.reduce_sum(hd, axis=0)
        hd = tf.reduce_sum(hd, axis=0)
        return hd
    hds = exploded_hd_graph(aggregates)
    return tf.map_fn(get_hd_sum, hds) / aggregates.shape[1]

def pd_aggregate_graph(aggregates):
    primes = tf.constant(PRIMES[:aggregates.shape[-1]], dtype=tf.float64)
    pds = tf.pow(primes, aggregates)
    pds = tf.reduce_prod(pds, 2)
    return log2_graph(pds)

def scaled_hd_graph(log_pitches, vectors, c=0.05, coeff=E):
    scales = scales_graph(log_pitches, vectors, c=c, coeff=coeff) + 2.0e-32
    hds = hd_graph(vectors)
    hds = hds + 1.0
    hds = tf.tile(hds[:, None, None], [1, scales.shape[1], scales.shape[2]])
    hds = hds * tf.math.reciprocal(scales)
    hds = tf.reduce_min(hds, axis=0)
    hds = tf.reduce_mean(hds, axis=1)
    hds = hds - 1.0
    return hds

def rationalize_within_tolerance(log_pitches, vectors, t=0.01):
    """
    Strict rationalization within a window of tolerance. Taken from "About
    Changes," and also from the article "Harmonic Series Aggregates" by Tenney.
    """
    log_vectors = pd_graph(vectors)
    diffs = tf.abs(log_vectors - log_pitches)
    out = []
    for d in diffs <= t:
        options = vectors[d]
        options_hds = hd_graph(options)
        winner_idx = tf.argmin(options_hds, axis=0)
        out.append(options[winner_idx])
    return out
