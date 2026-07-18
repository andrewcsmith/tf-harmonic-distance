- We need to hold a third value corresponding to each value that has to do with
  the "freshness" of the pitch, and should gradually decay over time. This will
  correspond to the volume of the sound, or some other parameter. It could be
  tracked through a second vector that would be updated upon each iteration.
  - TODO: Look at the viability of having a second vector that would be updated
    in each callback, perhaps keeping track of the error rate and other
    variables that may be read asynchronously.

- in hd-server
  - set_log_pitches needs to be able to only set log pitches in a single index

- questions:
  - does "converge" actually use the polar transform, or is it still the
    space that favors more pitches along the x=y axis?
  - the pds (pitch distances) themselves have to be transformed, as do the log_pitches
  - in p. 101 plot, it looks as though the top axis y=1 and the left axis x=0
    are mirror images of each other, and likewise for y=0 and x=1. verify this.

explain these

adapt hd.utilities.transform_to_unit_circle to higher dimensions


TODOS:
- [x] Save and restore the necessary parts of a VectorSpace so that it does not
  need to be fully computed at startup. This should ideally also entirely avoid
  the construction of `perms` if the `hds` and `pds` are simply being restored
  rather than computed.
  - Done: `VectorSpace.save(path)` / `VectorSpace.load(path)` (npz of vectors +
    pds/hds summaries; load restores a `"summaries"`-mode space with no
    enumeration, no materialization, and no `perms`). Exposed in the OSC server
    as `--save-vectorspace` / `--load-vectorspace` and `/hd/save_vectorspace`.
- [x] Be able to recall a specific `perms` value by index (from the `hds` and
  `pds`), even if the `perms` vector itself has never been generated. This
  should take into account indexing of specific cartesian permutations so that a
  permutation at a given index can be calculated on-the-fly.
  - Done: `VectorSpace.perms_at(flat_indices)` decodes flat cartesian-product
    indices (mixed-radix, base `num_vectors`, one digit per dimension) and
    gathers from the base vectors; `closest_from_log` now uses it in
    `"summaries"` mode. Exposed in the OSC server as `/hd/get_perm <index>`.