"""Collapse OmniGibson's O(N) full-scene fabric syncs during a bulk pose restore.

THE PROBLEM. `scene.reset()` restores every object's pose. Each restore goes through
`XFormPrim.set_position_orientation`, which does two things that interact badly:

    # omnigibson/prims/xform_prim.py
    position, orientation = PoseAPI.convert_world_pose_to_local(self._prim, ...)  # READS
    ...
    PoseAPI.invalidate()                                                          # DIRTIES

and the read is not cheap, because `convert_world_pose_to_local` needs the parent's
world pose:

    # omnigibson/utils/usd_utils.py  PoseAPI._refresh
    if og.sim._physx_fabric_interface:
        og.sim._physx_fabric_interface.update(...)   # a FULL-SCENE fabric sync

So object k's write dirties the cache, and object k+1's read rebuilds it by syncing the
*entire scene*. N objects => N full-scene syncs => quadratic. On
`house_double_floor_lower` this made one `session/start` exceed **90 minutes** at ~10
cores busy, with py-spy landing in `set_attribute` every single time. It is not a
deadlock and no amount of waiting fixes it.

THE FIX. During a bulk restore, nothing is physically moving — we are writing poses,
not simulating. The parent world poses that `convert_world_pose_to_local` needs are
therefore constant for the whole batch, so one sync at the start is enough. Refresh
once, suppress the intermediate invalidations, and invalidate once at the end.

WHY THIS IS SOUND. The assumption is precisely:

    no prim written inside the block is an ancestor of another prim written inside it.

In an OmniGibson scene restore that holds: objects are siblings under the scene prim,
not USD-parented to each other (an object "inside" a container is a spatial relation,
not a USD hierarchy one). If it were ever violated, the descendant would be positioned
against its ancestor's stale pose. `assert_no_moved_ancestors` makes that checkable
rather than assumed, and the end-to-end gate is stronger still: replay a harvested
expert episode and confirm it still succeeds, since `place_inside` is exquisitely
sensitive to wrong poses (a wrong pose leaves Inside=False and the object held).

Declaring the cache clean is an existing OmniGibson idiom, not one invented here —
`Simulator.render()` ends with `PoseAPI.mark_valid()` for the same reason.

PHYSICS STEPS STILL INVALIDATE. `scene.reset()` takes a physics step internally —
`Simulator.removing_objects` moves each doomed object to a graveyard, steps physics, then
restores the dumped state — so blanket suppression would let a post-step read see stale
poses. The context manager therefore wraps `og.sim.step_physics` to invalidate for real,
and suppresses only the pose-write churn that is actually quadratic. A restore takes a
handful of steps, so this costs a handful of syncs rather than N.

Even so, do not wrap a *caller's* physics loop in this: it would work, but batching buys
nothing there (the steps dominate) while widening the window in which a subtle staleness
bug could hide. The 25-step `keep_still()` loop in `load_activity_instance_tro_state` is
left unbatched for that reason.
"""

from __future__ import annotations

import contextlib
import time


class PoseBatchStats:
    """Counters from the most recent batch, for logging and for tests."""

    suppressed: int = 0
    elapsed_s: float = 0.0

    @classmethod
    def summary(cls) -> str:
        return (
            f"pose-batch: suppressed {cls.suppressed} full-scene fabric syncs "
            f"in {cls.elapsed_s:.1f}s"
        )


@contextlib.contextmanager
def batched_pose_writes(label: str = "restore"):
    """Make a bulk pose restore linear instead of quadratic.

    Args:
        label: shows up in the log line, so multiple batches are distinguishable.

    Usage:
        with batched_pose_writes("scene.reset"):
            env.scene.reset()
    """
    import omnigibson as og
    from omnigibson.utils.usd_utils import PoseAPI

    # One sync up front: after this, every parent world pose is valid, and parents do
    # not move for the rest of the block.
    PoseAPI._refresh()

    suppressed = 0

    def _suppressed_invalidate(*_args, **_kwargs):
        nonlocal suppressed
        suppressed += 1

    # PoseAPI.invalidate is a classmethod; grab the descriptor itself so it can be put
    # back exactly as it was rather than rebound as a plain function.
    original = PoseAPI.__dict__["invalidate"]
    PoseAPI.invalidate = staticmethod(_suppressed_invalidate)

    # A physics step DOES legitimately invalidate the cache, and `scene.reset()` takes
    # one internally: `Simulator.removing_objects` moves each doomed object to the
    # graveyard and then calls `step_physics()` before restoring the dumped state. So
    # suppressing invalidation blindly across the whole block would let a post-step
    # read see stale poses. Keep the real invalidation for physics motion and suppress
    # only the pose-write churn, which is what is actually quadratic. There are a
    # handful of steps per restore, so this costs a handful of syncs, not N.
    sim = og.sim
    orig_step_physics = sim.step_physics

    def _step_physics_then_invalidate(*a, **k):
        result = orig_step_physics(*a, **k)
        PoseAPI.VALID = False          # what the real invalidate() does
        return result

    sim.step_physics = _step_physics_then_invalidate

    t0 = time.time()
    try:
        yield
    finally:
        sim.step_physics = orig_step_physics
        PoseAPI.invalidate = original
        # The batch really did dirty the cache — mark it so, once.
        PoseAPI.invalidate()
        PoseBatchStats.suppressed = suppressed
        PoseBatchStats.elapsed_s = time.time() - t0
        print(
            f"[pose_batch] {label}: collapsed {suppressed} full-scene fabric syncs "
            f"into 1 ({PoseBatchStats.elapsed_s:.1f}s)",
            flush=True,
        )


def assert_no_moved_ancestors(prim_paths) -> None:
    """Check the one assumption `batched_pose_writes` rests on.

    Raises if any path in ``prim_paths`` is a strict USD ancestor of another. Cheap
    (a prefix check per pair via sorting), so it is worth running once on a new scene
    or activity rather than trusting that the sibling layout holds everywhere.
    """
    paths = sorted(set(prim_paths))
    for i, parent in enumerate(paths):
        prefix = parent.rstrip("/") + "/"
        for child in paths[i + 1:]:
            if not child.startswith(prefix):
                break
            raise AssertionError(
                f"batched_pose_writes is unsafe here: {parent!r} is an ancestor of "
                f"{child!r}, so the descendant would be placed against a stale parent "
                f"pose. Restore these in separate batches."
            )
