# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

James Tenney's harmonic distance metrics implemented in TensorFlow. The core use case: given one or more pitches (as log2 frequency ratios), gradient-descend them toward nearby just-intonation ratios by minimizing a loss surface where every candidate ratio sits in a parabolic well scaled by its harmonic distance (simpler ratios = deeper/wider wells).

## Layout and commands

This is a uv workspace (Python 3.12, TensorFlow with CUDA):

- `harmonic_distance/` — the actual installable library (workspace member, editable). All real code lives in `harmonic_distance/harmonic_distance/`, tests in `harmonic_distance/tests/`.
- Repo root — exploratory notebooks and scratch scripts driving the library. `optimize_1d.py`, `optimize_2d_single.py`, `optimize_2d_batched.py`, and `plot_1d.py` are stale TF1-style code (`tf.Session`) and won't run against current TensorFlow.

Use the existing `.venv` directly — `uv run` currently fails because the `harmonic_distance` workspace member has only a `setup.py`, no `pyproject.toml`:

```sh
.venv/bin/pytest harmonic_distance/tests/            # all tests
.venv/bin/pytest harmonic_distance/tests/test_optimize.py -k test_name   # single test
.venv/bin/pytest harmonic_distance/tests/ -m "not slow"   # skip gradient-descent convergence tests
```

Tests force CPU (`conftest.py` hides the GPU — tiny tensors are ~3x faster without kernel-launch overhead; set `HD_TEST_GPU=1` to override) and share session-scoped `VectorSpace` fixtures from `conftest.py`; reuse those fixtures rather than constructing new spaces in tests, since construction dominates test runtime.

Everything uses `tf.float64` throughout.

## Theory (Dissertation-Theory.pdf, pp. 72–102)

The library implements the theory of interval/chord rationalization from the author's dissertation (chapter 4). The math that matters for the code:

- **Harmonic space (Tenney)**: a pitch is an integer vector of prime-factor exponents. `[-1, 1]` over primes `(2, 3)` means 2⁻¹·3¹ = 3/2. Each prime adds a dimension. **Harmonic distance** of a ratio a/b is log2(a·b), equivalently Σ|exponent|·log2(prime) — `tenney.hd_graph`.
- **Pitch aggregates**: an n-pitch chord is a point in (n−1)-dimensional space; each axis is the log2 distance from pitch 1 to another pitch (an implicit 1/1 root is always present). The chord's HD is the sum of harmonic distances over all combinatorial pairs, divided by n−1 so axes keep the same scale as the 1-D case — `tenney.hd_aggregate_graph`, using basis matrices from `utilities.get_bases` (combinatorial contours, after Polansky/Kant).
- **Tolerance and rationalization**: every rational interval gets a region of tolerance in log-pitch space. A parabola `P(d) = d²/c` is drawn around each candidate ratio (`utilities.reduce_parabola`; `c` is the "curve"/variance), and the **scaled distance** of a heard pitch x to candidate (p, h) is `SD(x, p, h) = 2ʰ · P(x − p) + h` where p is the candidate's pitch distance and h its harmonic distance. (The 2ʰ factor rather than plain h keeps the unison, h = 0, from degenerating.) This is exactly `optimize.parabolic_loss_function` / `VectorSpace.loss`. Taking the min of SD over all candidates yields a continuous, piecewise-parabolic surface whose local minima ("troughs") are the tuneable intervals — hence gradient descent for rationalization.
- **The `c` parameter** directly controls how many tuneable intervals exist: larger c ⇒ simpler ratios subsume their neighbors ⇒ fewer minima. c ∈ [0.0048, 0.0063] makes 8:7 the narrowest tuneable interval in an octave (matching the critical band and Marc Sabat's tuneable-intervals catalog).
- **Polar transform**: in the Cartesian projection the x=y diagonal (doubled voices) is √2 longer than the axes, giving it spuriously more minima. `utilities.transform_to_unit_circle` fixes this by rescaling each point so its Euclidean magnitude equals its Chebyshev (L∞) magnitude — `x · (‖x‖∞/‖x‖₂)` — which compresses every k-fold diagonal by 1/√k in any dimension; a transformed chord's magnitude equals its span. `transform_from_unit_circle` is the exact inverse (swap the norms). `VectorSpace(polar=True)` applies the transform to pds once at construction, so optimization runs on perfect parabolas in the transformed space and gradients never differentiate the transform (the non-smooth tie manifolds are only ever evaluated, not differentiated); `Minimizer.set_real_log_pitches()` / `real_log_pitches()` convert at the boundaries.
- **Slices**: fixing all but one dimension (e.g. SD(x) along <1:1, 3:2, x>) yields a scale of pitches that harmonize with a sounding chord — the generative use driving the batch-optimization notebooks and scripts.

## Architecture

The pipeline from ratios to optimized pitches spans four modules in `harmonic_distance/harmonic_distance/`:

1. **`vectors.py` — `VectorSpace`**: builds the candidate space. A "vector" is a row of prime-factor exponents (over `PRIMES = [2, 3, 5, 7, ...]`), so `[-1, 1]` means 3/2. `get_vectors()` enumerates all vectors within per-prime `prime_limits`, filters by pitch-distance `pd_bounds` and harmonic-distance `hd_limit`. For `dimensions > 1` (chords), the space is the Cartesian product of vectors with itself (`cartesian.permutations`), which explodes combinatorially — hence materialization modes.

2. **`tenney.py`**: the metrics. `hd_aggregate_graph` computes the mean pairwise harmonic distance of a chord (including the implicit 1/1 root, via `utilities.get_bases` combinatorial contours); `pd_aggregate_graph` computes log2 pitch distances.

3. **`optimize.py` — `Minimizer`**: owns a `tf.Variable` `log_pitches` of shape `[batch_size, dimensions]` and runs Adadelta against `VectorSpace.loss()`. Stopping condition is the L2 norm of the gradient falling below `convergence_threshold`. Constructing a `Minimizer` does NOT run any optimization (intentional — supports server startup); call `minimize()`. Live controls designed to work after `tf.function` tracing: `set_convergence_threshold()` (a non-trainable variable, not a Python float), `set_active_mask()` / `set_active_count()` (zero out gradients for masked-out batch elements; masks with holes like `[1,1,0,1]` are supported), `set_all_curves()` (parabola width per dimension). Optimizer state is snapshotted at construction and restored on each `minimize()` call.

4. **`utilities.py`**: `reduce_parabola` (the distance kernel used by the loss), `get_bases`, and the n-dimensional polar transform pair `transform_to_unit_circle` / `transform_from_unit_circle` (see Theory above; wired into `VectorSpace(polar=True)`).

### VectorSpace materialization modes

`VectorSpace(materialize=...)` takes a string, never a bool (bools raise):

- `"full"`: cache `perms`, `pds`, `hds` as variables, built in chunks of `batch_size` rows (each row's hd/pd is independent, so chunking doesn't change results — it only bounds peak memory during construction, which otherwise spikes because `hd_aggregate_graph`/`pd_aggregate_graph` would run over the entire permutation space in one op).
- `"summaries"`: cache only `pds`/`hds`, also built in memory-bounded chunks — for high dimensions where full perms don't fit but recomputing summaries every step is too slow.
- `"none"`: recompute everything batch-by-batch during loss evaluation; the batched loss path uses a `tf.custom_gradient` (`_batched_loss`).
- `"auto"` (default): `"full"` if `num_vectors ** dimensions <= materialize_limit` (1M), else `"none"`.

Both `"full"` and `"summaries"` accept `progress_callback(batch_index, total_batches)` (1-indexed). A `batch_index=0` call fires first — before the full-size `pds`/`hds`/`perms` arrays are allocated — since `permutation_count` grows as `num_vectors ** dimensions` and that allocation alone can be slow or memory-heavy at higher dimensions; the 0-call surfaces `total_batches` (hence scale) before that happens. One call per completed chunk follows. The OSC server passes a default printer so this shows at startup regardless of which mode `"auto"` resolves to.

`Minimizer.loss()` delegates to `VectorSpace.loss()`, which routes to the materialized or batched implementation.

## Branch state / conventions

- `LLM.md` contains detailed handoff notes for the current line of work (materialization modes, live minimizer controls, and a sibling OSC server package at `../harmonic_distance_osc/` that depends on this checkout). Read it when working on this branch.
- The worktree carries many unrelated modified notebooks, scratch scripts, and generated artifacts (`vectorspace/`, `*.egg-info/`). Keep them out of commits; review `git status --short` carefully and confirm with the user before staging anything outside `harmonic_distance/`.
