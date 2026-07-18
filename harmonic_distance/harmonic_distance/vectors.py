import tensorflow as tf
import numpy as np
from . import PRIMES
from . import tenney
from .utilities import log2_graph, E, get_bases, parabolic_scale, reduce_parabola, transform_to_unit_circle
from .cartesian import permutations

# Create default prime limits for ease of use
PRIME_LIMITS = [5, 5, 3, 3, 2, 1]
PD_BOUNDS = (0.0, 4.0)
HD_LIMIT = 9.0
DIMS = 1
BATCH_SIZE = 65536
MATERIALIZE_LIMIT = 1_000_000
MATERIALIZE_MODES = ("auto", "none", "summaries", "full")

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
        np.prod(np.power(primes, num), axis=-1),
        np.prod(np.power(primes, np.abs(den)), axis=-1)
    ], -1)

class VectorSpace(tf.Module):
    def __init__(
            self,
            *args,
            dimensions=DIMS,
            batch_size=BATCH_SIZE,
            materialize="auto",
            materialize_limit=MATERIALIZE_LIMIT,
            polar=False,
            progress_callback=None,
            device=None,
            **kwargs):
        self.dimensions = dimensions
        self.batch_size = int(batch_size)
        self.materialize_limit = int(materialize_limit)
        self.polar = bool(polar)
        self.progress_callback = progress_callback
        with tf.device(device) if device is not None else _null_context():
            self.vectors = tf.identity(self.get_vectors(**kwargs))
        self.num_vectors = int(self.vectors.shape[0])
        self.n_primes = int(self.vectors.shape[-1])
        self._permutation_count = self.num_vectors ** self.dimensions
        self.materialize_mode = self._resolve_materialize_mode(materialize)
        self.materialized = self.materialize_mode != "none"
        self.has_perms = self.materialize_mode == "full"
        if self.materialize_mode == "full":
            self.perms, self.pds, self.hds = self.materialize_full()
        elif self.materialize_mode == "summaries":
            self.pds, self.hds = self.materialize_summaries()

    def _resolve_materialize_mode(self, materialize):
        if isinstance(materialize, bool):
            raise ValueError("materialize must be 'auto', 'none', 'summaries', or 'full'")
        if materialize == "auto":
            if self._permutation_count <= self.materialize_limit:
                return "full"
            return "none"
        if materialize not in MATERIALIZE_MODES:
            raise ValueError("materialize must be 'auto', 'none', 'summaries', or 'full'")
        return materialize

    @property
    def permutation_count(self):
        return self._permutation_count

    def permutation_indices(self, start, count):
        stop = min(int(start) + int(count), self._permutation_count)
        flat = tf.range(int(start), stop, dtype=tf.int64)
        return self._flat_to_permutation_indices(flat)

    def perms_at(self, flat_indices):
        """
        Materialize only the permutations at the given flat cartesian-product
        indices, decoding each index into its per-dimension vector indices —
        the full perms tensor is never required (or consulted). Accepts any
        shape of indices and returns shape [..., dimensions, n_primes].
        """
        flat = tf.cast(tf.convert_to_tensor(flat_indices), tf.int64)
        tf.debugging.assert_non_negative(
            flat, message="permutation index must be non-negative"
        )
        tf.debugging.assert_less(
            flat,
            tf.constant(self._permutation_count, dtype=tf.int64),
            message="permutation index out of range",
        )
        return tf.gather(self.vectors, self._flat_to_permutation_indices(flat))

    def permutation_batch(self, start, count):
        indices = self.permutation_indices(start, count)
        return tf.gather(self.vectors, indices)

    def summary_batch(self, start, count):
        perms = self.permutation_batch(start, count)
        return self._maybe_polar(tenney.pd_aggregate_graph(perms)), tenney.hd_aggregate_graph(perms)

    def _maybe_polar(self, pds):
        # When polar, all pds (and any log_pitches evaluated against them) live
        # in the transformed space where magnitude equals chord span; convert
        # at the boundaries with utilities.transform_{to,from}_unit_circle.
        if self.polar:
            return transform_to_unit_circle(pds)
        return pds

    def full_batch(self, start, count):
        perms = self.permutation_batch(start, count)
        pds = self._maybe_polar(tenney.pd_aggregate_graph(perms))
        hds = tenney.hd_aggregate_graph(perms)
        return perms, pds, hds

    def materialize_full(self):
        # Computing hd_aggregate_graph/pd_aggregate_graph over the entire
        # permutation space in a single op (as opposed to summaries mode,
        # which was already chunked) can spike device memory well past what
        # the final perms/pds/hds variables need. Chunk by self.batch_size
        # instead; each row is computed independently, so the result is
        # identical to the unchunked computation.
        batch_size = self.batch_size
        total_batches = -(-self._permutation_count // batch_size)
        # permutation_count grows as num_vectors ** dimensions, so allocating
        # the full-size arrays below can itself be slow or memory-heavy before
        # any per-batch work (or progress_callback call) happens; announce the
        # plan first with the batch_index=0 sentinel.
        if self.progress_callback is not None:
            self.progress_callback(0, total_batches)
        perms = tf.Variable(
            tf.zeros([self._permutation_count, self.dimensions, self.n_primes], dtype=tf.float64),
            trainable=False,
        )
        pds = tf.Variable(
            tf.zeros([self._permutation_count, self.dimensions], dtype=tf.float64),
            trainable=False,
        )
        hds = tf.Variable(tf.zeros([self._permutation_count], dtype=tf.float64), trainable=False)
        starts = range(0, self._permutation_count, batch_size)
        for batch_index, start in enumerate(starts, start=1):
            count = min(batch_size, self._permutation_count - start)
            chunk_perms, chunk_pds, chunk_hds = self.full_batch(start, count)
            perms[start : start + count].assign(chunk_perms)
            pds[start : start + count].assign(chunk_pds)
            hds[start : start + count].assign(chunk_hds)
            if self.progress_callback is not None:
                self.progress_callback(batch_index, total_batches)
        return perms, pds, hds

    def materialize_summaries(self):
        batch_size = self.summary_materialization_batch_size()
        total_batches = -(-self._permutation_count // batch_size)
        # Same rationale as materialize_full: announce before the potentially
        # slow/large full-size allocation, not just once chunked work starts.
        if self.progress_callback is not None:
            self.progress_callback(0, total_batches)
        pds = tf.Variable(
            tf.zeros([self._permutation_count, self.dimensions], dtype=tf.float64),
            trainable=False,
        )
        hds = tf.Variable(tf.zeros([self._permutation_count], dtype=tf.float64), trainable=False)
        starts = range(0, self._permutation_count, batch_size)
        for batch_index, start in enumerate(starts, start=1):
            count = min(batch_size, self._permutation_count - start)
            chunk_pds, chunk_hds = self.summary_batch(start, count)
            pds[start : start + count].assign(chunk_pds)
            hds[start : start + count].assign(chunk_hds)
            if self.progress_callback is not None:
                self.progress_callback(batch_index, total_batches)
        return pds, hds

    def summary_materialization_batch_size(self):
        # Keep the transient permutation tensor no larger than the final pds+hds
        # summary cache. Concatenation can still briefly duplicate summaries.
        summary_row_bytes = (self.dimensions + 1) * np.dtype(np.float64).itemsize
        permutation_row_bytes = self.dimensions * self.n_primes * np.dtype(np.float64).itemsize
        summary_total_bytes = max(1, self._permutation_count * summary_row_bytes)
        memory_limited_batch_size = max(1, summary_total_bytes // permutation_row_bytes)
        return int(min(self.batch_size, memory_limited_batch_size))
    
    @tf.function
    def closest_from_log(self, log_pitches):
        if not self.materialized:
            return self._closest_from_log_batched(log_pitches)
        diffs = tf.abs(self.pds[:, None] - log_pitches[None, :])
        mins = tf.argmin(tf.reduce_sum(diffs, axis=-1), axis=0)
        if self.has_perms:
            return tf.gather(self.perms, mins, axis=0)
        return self.perms_at(mins)

    def _closest_from_log_batched(self, log_pitches):
        batch_count = int(log_pitches.shape[0])
        best_values = tf.fill([batch_count], tf.constant(np.inf, dtype=tf.float64))
        best_indices = tf.zeros([batch_count], dtype=tf.int64)
        for start in range(0, self._permutation_count, self.batch_size):
            pds, _ = self.summary_batch(start, self.batch_size)
            diffs = tf.reduce_sum(tf.abs(pds[:, None] - log_pitches[None, :]), axis=-1)
            chunk_values = tf.reduce_min(diffs, axis=0)
            chunk_indices = tf.argmin(diffs, axis=0, output_type=tf.int64) + start
            is_better = chunk_values < best_values
            best_values = tf.where(is_better, chunk_values, best_values)
            best_indices = tf.where(is_better, chunk_indices, best_indices)
        return self.perms_at(best_indices)

    def unique_ratios(self, log_pitches):
        ratios = to_ratio(self.closest_from_log(log_pitches))
        unique = np.log2(np.unique(ratios[:, :, 0] / ratios[:, :, 1]))
        return to_ratio(self.closest_from_log(unique))

    def loss(self, log_pitches, curves=None):
        if self.materialized:
            return self._materialized_loss(log_pitches, curves=curves)
        return self._batched_loss(log_pitches, curves=curves)

    def _materialized_loss(self, log_pitches, curves=None):
        distances = reduce_parabola(self.pds[:, None] - log_pitches, axis=-1, curves=curves)
        scaled = ((2.0 ** self.hds)[:, None] * distances) + self.hds[:, None]
        return tf.reduce_min(scaled, axis=0)

    def _batched_loss(self, log_pitches, curves=None):
        if curves is None:
            curves = tf.ones([self.dimensions], dtype=tf.float64)
        else:
            curves = tf.cast(curves, tf.float64)

        @tf.custom_gradient
        def loss_with_gradient(x):
            best_values, best_pds, best_hds = self._batched_loss_winners(x, curves)

            def grad(dy):
                scale = (2.0 ** best_hds)[:, None]
                gradients = -2.0 * scale * (best_pds - x) / curves
                return dy[:, None] * gradients

            return best_values, grad

        return loss_with_gradient(log_pitches)

    def _batched_loss_winners(self, log_pitches, curves):
        batch_count = int(log_pitches.shape[0])
        best_values = tf.fill([batch_count], tf.constant(np.inf, dtype=tf.float64))
        best_pds = tf.zeros([batch_count, self.dimensions], dtype=tf.float64)
        best_hds = tf.zeros([batch_count], dtype=tf.float64)
        for start in range(0, self._permutation_count, self.batch_size):
            pds, hds = self.summary_batch(start, self.batch_size)
            distances = reduce_parabola(pds[:, None] - log_pitches, axis=-1, curves=curves)
            scaled = ((2.0 ** hds)[:, None] * distances) + hds[:, None]
            chunk_values = tf.reduce_min(scaled, axis=0)
            chunk_indices = tf.argmin(scaled, axis=0, output_type=tf.int64)
            chunk_pds = tf.gather(pds, chunk_indices)
            chunk_hds = tf.gather(hds, chunk_indices)
            is_better = chunk_values < best_values
            best_values = tf.where(is_better, chunk_values, best_values)
            best_pds = tf.where(is_better[:, None], chunk_pds, best_pds)
            best_hds = tf.where(is_better, chunk_hds, best_hds)
        return best_values, best_pds, best_hds

    def _flat_to_permutation_indices(self, flat):
        indices = []
        for power in reversed(range(self.dimensions)):
            divisor = self.num_vectors ** power
            indices.append((flat // divisor) % self.num_vectors)
        return tf.stack(indices, axis=-1)
    
    def get_vectors(self, prime_limits=PRIME_LIMITS, pd_bounds=PD_BOUNDS, hd_limit=HD_LIMIT, **kwargs):
        vectors = space_graph_altered_permutations(prime_limits, bounds=pd_bounds)
        vectors_hds = tenney.hd_aggregate_graph(tf.cast(vectors[:, None, :], tf.float64))
        return tf.boolean_mask(vectors, vectors_hds < hd_limit)
    
    def get_perms(self, dimensions=DIMS, **kwargs):
        return permutations(self.vectors, times=dimensions)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
