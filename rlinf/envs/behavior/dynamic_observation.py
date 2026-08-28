"""Session-local post-action geometry for BEHAVIOR-TextWorld observations.

The sampled :class:`layout.Layout` is immutable evidence about the reset scene.  BDDL
tools mutate symbolic state, however, so projecting that reset snapshot after grasp,
place, or transform produces stale boxes.  This module stores only the deterministic
delta from the reset layout.  It is observation state, not physics: every override is
labelled ``symbolic_transition`` and must never be reported as measured geometry.

Mutation methods intentionally do not advance ``revision``.  A caller first completes
the matching BDDL transition and then calls :meth:`commit` exactly once.  Failed tools
therefore leave both geometry and the observation revision unchanged.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicPose:
    """A root-link pose introduced by a symbolic transition."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    pose_source: str = "symbolic_transition"


class DynamicObservationState:
    """Overlay of hidden and moved entities on top of an immutable reset layout."""

    def __init__(self, initial_xyz: dict, initial_orientation: dict):
        self._initial_xyz = {
            str(name): tuple(float(value) for value in position)
            for name, position in initial_xyz.items()
        }
        self._initial_orientation = {
            str(name): tuple(float(value) for value in orientation)
            for name, orientation in initial_orientation.items()
        }
        self._overrides: dict[str, DynamicPose] = {}
        self._hidden: set[str] = set()
        self._placement_counts: dict[tuple[str, str], int] = {}
        self.revision = 0

    @property
    def has_symbolic_geometry(self) -> bool:
        """Whether a committed transition has changed observable geometry."""
        return self.revision > 0

    def _raw_position(self, name: str) -> tuple[float, float, float] | None:
        pose = self._overrides.get(name)
        return pose.position if pose is not None else self._initial_xyz.get(name)

    def position(self, name: str) -> tuple[float, float, float] | None:
        """Current visible root-link position, or ``None`` when absent/held."""
        return None if name in self._hidden else self._raw_position(name)

    def last_position(self, name: str) -> tuple[float, float, float] | None:
        """Last root-link position even if the entity is currently hidden."""
        return self._raw_position(name)

    def orientation(self, name: str) -> tuple[float, float, float, float]:
        pose = self._overrides.get(name)
        if pose is not None:
            return pose.orientation
        return self._initial_orientation.get(name, (0.0, 0.0, 0.0, 1.0))

    def pose_source(self, name: str, *, sampled_initial: bool) -> str | None:
        """Provenance for trace metadata; never expose it in policy detections."""
        if name in self._hidden or self._raw_position(name) is None:
            return None
        if name in self._overrides:
            return self._overrides[name].pose_source
        return "sampled_initial" if sampled_initial else "generated_initial"

    def hide(self, name: str) -> None:
        """Remove an entity from projection while retaining its last pose."""
        self._hidden.add(name)

    def show_at(
        self,
        name: str,
        position,
        *,
        orientation=None,
    ) -> None:
        """Expose @name at a deterministic transition-created root-link pose."""
        current_orientation = (
            self.orientation(name) if orientation is None else tuple(orientation)
        )
        self._overrides[name] = DynamicPose(
            position=tuple(float(value) for value in position),
            orientation=tuple(float(value) for value in current_orientation),
        )
        self._hidden.discard(name)

    @staticmethod
    def _phase(name: str, target: str, relation: str) -> float:
        payload = f"{name}\0{target}\0{relation}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        return 2.0 * math.pi * value / 2**32

    def target_relative_center(
        self,
        name: str,
        target: str,
        relation: str,
        *,
        target_center,
        target_extent,
        object_extent,
        slot: int = 0,
    ) -> tuple[float, float, float]:
        """Deterministic bbox centre for an analytic on/in placement.

        Multiple placements spread within the target footprint.  The calculation is
        deliberately conservative and deterministic; it is not a fit or collision
        claim.  BDDL remains the authoritative success state.
        """
        phase = self._phase(name, target, relation) + slot * (math.pi / 3.0)
        radius = 0.22 * min(float(target_extent[0]), float(target_extent[1]))
        dx = radius * math.cos(phase) if slot or radius else 0.0
        dy = radius * math.sin(phase) if slot or radius else 0.0
        if relation == "inside":
            z = float(target_center[2])
        elif relation == "ontop":
            z = (
                float(target_center[2])
                + float(target_extent[2])
                + float(object_extent[2])
                + 0.02
            )
        else:
            raise ValueError(f"unsupported placement relation: {relation!r}")
        return (
            float(target_center[0]) + dx,
            float(target_center[1]) + dy,
            z,
        )

    def next_placement_slot(self, target: str, relation: str) -> int:
        """Return the next slot without mutating state, for fail-closed previews."""
        return self._placement_counts.get((target, relation), 0)

    def record_placement(self, target: str, relation: str) -> None:
        """Consume one target-relative slot after the BDDL transition succeeds."""
        key = (target, relation)
        self._placement_counts[key] = self._placement_counts.get(key, 0) + 1

    def commit(self) -> int:
        """Publish one successful visual transition and return its revision."""
        self.revision += 1
        return self.revision


__all__ = ["DynamicObservationState", "DynamicPose"]
