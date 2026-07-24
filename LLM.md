# LLM Handoff Notes

## Scope

This branch contains two related lines of work:

1. Core `harmonic_distance` improvements for higher-dimensional optimization.
2. A standalone `harmonic_distance_osc` package for controlling the optimizer over OpenSoundControl from SuperCollider.

The OSC package now lives as a sibling directory next to this checkout:

```text
workspaces/
  tf-harmonic-distance/
  harmonic_distance_osc/
```

## Core Library Changes

### VectorSpace materialization modes

`VectorSpace(materialize=...)` now expects a string mode:

- `"none"`: store base vectors only; compute permutation summaries batch-by-batch during evaluation.
- `"summaries"`: cache only `pds` and `hds`; do not cache full prime-exponent permutations.
- `"full"`: cache `perms`, `pds`, and `hds`.
- `"auto"`: use `"full"` under `materialize_limit`, otherwise `"none"`.

Boolean materialization values are rejected for consistency.

The new `"summaries"` mode is intended for larger dimensionalities where convergence should avoid recomputing summaries every step, but memory should not be spent on full `perms`. It preallocates final `pds` and `hds` tensors and fills them in bounded chunks. The chunk cap keeps the temporary permutation tensor no larger than the eventual `pds+hds` summary cache.

### VectorSpace save/restore and on-demand permutation recall

`VectorSpace.save(path)` writes the base vectors plus the cached `pds`/`hds`
summaries to a compressed npz (requires `"summaries"` or `"full"` mode; the
full `perms` tensor is never written). `VectorSpace.load(path)` is a
classmethod that restores a `"summaries"`-mode space directly from disk —
no vector enumeration, no summary materialization, no `perms` construction.
`dimensions` and `polar` are baked into the file; `load` accepts an optional
`batch_size` override and `device`. Constructed spaces record their
enumeration provenance (`prime_limits`, `pd_bounds`, `hd_limit` attributes;
`pd_bounds=None` means unbounded), and save/load carries it — files saved
before provenance existed load with those attributes as `None`.

`VectorSpace.perms_at(flat_indices)` recalls specific permutations by flat
cartesian-product index without the `perms` tensor: each index is decoded
mixed-radix (base `num_vectors`, one digit per dimension, matching the row
order of `cartesian.permutations`) and gathered from `self.vectors`. It
accepts any index shape, returns `[..., dimensions, n_primes]`, and asserts
indices are in `[0, permutation_count)`. `closest_from_log` now serves
`"summaries"`-mode (including loaded) spaces from the cached `pds` + `perms_at`
instead of falling back to the batched full recomputation.

### Minimizer startup and live controls

`Minimizer.__init__` no longer calls `opt_minimize()` eagerly. This avoids doing an expensive loss sweep during server startup or object construction.

`convergence_threshold` is now a non-trainable TensorFlow variable with `set_convergence_threshold(...)`, so changes after `tf.function` tracing are respected.

`max_iters` is likewise a non-trainable variable with `set_max_iters(...)`;
0 or None means unlimited (converge until the threshold is met). It is read
on every loop iteration, so assigning a small value from another thread —
`/hd/set_max_iters` over OSC — halts a running minimize after the current
step; that is the intended stop lever for unlimited runs.

`set_active_mask(...)` is now exposed in the core optimizer. `set_active_count(...)` remains as a contiguous-mask convenience wrapper. This supports masks with holes, e.g. `[1, 1, 0, 1]`.

Curves are per voice: `Minimizer.curves` has shape `[batch_size, 1]`
(broadcast across dimensions in the loss) and `set_curves(...)` accepts a
single number for every voice or one value per voice; `set_all_curves(...)`
survives as the scalar-only alias. Over OSC, `/hd/set_curve` takes one value
(all voices, backwards-compatible) or up to `max_batch_size` values (voices
`0..N-1`, rest unchanged) and `/hd/get_curves` reads them back; `--curve`
accepts the same as comma-separated values.

## OSC Package

The standalone package lives at:

```text
../harmonic_distance_osc/
```

It provides:

```sh
uv run hd-osc-server
```

Default OSC server behavior:

- `--materialize summaries`
- `--host 127.0.0.1`
- `--port 5656`
- `--convergence-threshold 1e-8`

Important OSC endpoints:

```text
/hd/ping
/hd/status
/hd/set_cents
/hd/get_cents
/hd/set_log_pitches
/hd/get_log_pitches
/hd/converge
/hd/set_active_count
/hd/set_active_mask
/hd/get_active_mask
/hd/set_curve
/hd/get_curves
/hd/set_convergence_threshold
/hd/set_max_iters
/hd/set_hd_limit
/hd/set_pd_bounds
/hd/get_perm
/hd/save_vectorspace
/hd/quit
```

Enumeration filters are settable at startup (`--hd-limit`, `--pd-bounds
MIN,MAX`) and at runtime: `/hd/set_hd_limit` and `/hd/set_pd_bounds` rebuild
the VectorSpace and swap in a fresh Minimizer asynchronously (reply
`rebuilding`, then `/hd/rebuilt <vectors> <perms> <hd_limit> <pd_min>
<pd_max>`), preserving pitches/mask/curves/threshold. Rebuild and converge
mutually exclude. Rebuilding a loaded space uses the provenance stored in the
npz and errors on pre-provenance files. `/hd/status` gained trailing fields:
`is_rebuilding`, `hd_limit`, `pd_min`, `pd_max` (nan when provenance unknown).

Vector-space persistence flags: `--save-vectorspace PATH` saves the
constructed space at startup (then keeps serving); `--load-vectorspace PATH`
restores one, skipping enumeration/materialization — the file is authoritative
for dimensions/polar, and `--prime-limits`/`--materialize` are ignored.
`/hd/get_perm <index>` replies `/hd/perm <index> <dims> <num> <den> ...`
(index echoed as a string; OSC ints are int32).

For WSL/Windows SuperCollider testing, run:

```sh
uv run hd-osc-server --host 0.0.0.0 --verbose
```

Then send `/hd/ping` from SuperCollider. If WSL logs the inbound packet but SC does not receive `/hd/pong`, use explicit reply routing:

```sh
uv run hd-osc-server --host 0.0.0.0 --verbose --reply-host WINDOWS_IP --reply-port 57120
```

`test.sc` contains a simple SuperCollider smoke test.

## Verification Run

The latest focused verification was:

```sh
pytest harmonic_distance/tests/test_vectors.py harmonic_distance/tests/test_optimize.py -q
```

Result:

```text
29 passed
```

The OSC server module also compiles:

```sh
python -m py_compile harmonic_distance_osc/src/hd_osc_server/server.py
```

## Packaging Boundary Note

`harmonic_distance_osc` is not needed for the core `harmonic_distance` library use case. It has been moved to a sibling layout:

```text
workspaces/
  tf-harmonic-distance/
  harmonic-distance-osc/
```

Ensure `../harmonic_distance_osc/pyproject.toml` points back to this checkout:

```toml
[tool.uv.sources]
harmonic-distance = { path = "../tf-harmonic-distance/harmonic_distance", editable = true }
```

or depend on a released `harmonic-distance` package once the core changes are published.

## Suggested Commit Series

Keep unrelated notebook and local script changes out of these commits.

1. `core: add explicit vector materialization modes`
   - `harmonic_distance/harmonic_distance/vectors.py`
   - `harmonic_distance/tests/test_vectors.py`
   - Covers `"none"`, `"summaries"`, `"full"`, `"auto"`, boolean rejection, and summary-only caching.

2. `core: expose live minimizer controls`
   - `harmonic_distance/harmonic_distance/optimize.py`
   - `harmonic_distance/tests/test_optimize.py`
   - Covers no eager constructor optimization, live convergence threshold updates, and active masks with holes.

3. `osc: add standalone harmonic distance OSC server`
   - `harmonic_distance_osc/pyproject.toml`
   - `harmonic_distance_osc/uv.lock`
   - `harmonic_distance_osc/src/hd_osc_server/__init__.py`
   - `harmonic_distance_osc/src/hd_osc_server/server.py`
   - `harmonic_distance_osc/README.md`
   - `harmonic_distance_osc/test.sc`

4. Optional separate commit in the OSC project:
   - `osc: track standalone OSC package`
   - Commit the sibling `harmonic_distance_osc/` project in its own repository if desired.

## Files To Avoid Accidentally Committing

The current worktree contains unrelated modified/untracked notebooks, scratch scripts, generated artifacts, and local environment files. Review `git status --short` carefully before staging.

Likely unrelated or generated examples include:

```text
*.ipynb
.python-version
callbacks.py
osc.py
vectorspace/
harmonic_distance/harmonic_distance.egg-info/
pyproject.toml
uv.lock
```

Confirm with the user before including any of those.
