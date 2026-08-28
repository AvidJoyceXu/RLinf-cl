"""Simulator-free constants and metrics shared by BEHAVIOR policy adapters."""

from __future__ import annotations

import os

MAX_NUDGES = 3
NUDGE = (
    "Continue working on the task. Respond with a tool call, not prose. "
    "If the goal is already achieved, call end_task."
)

DETECT_OBS_MODES = frozenset(
    {
        "detect",
        "detect_scope",
        "detect_scope_scan",
        "detect_scope_memory",
        "fov_distract",
    }
)
DETECT_RUNTIME_PATHS = {
    "BEHAVIOR_BDDL_DEFINITION_ROOT": "directory",
    "BEHAVIOR_NATIVE_BBOX_PATH": "file",
    "BEHAVIOR_ASSET_SCENE_ROOT": "directory",
}


def validate_detect_runtime(obs_mode: str) -> None:
    """Fail before model loading when a detect run is not release-pinned."""
    if obs_mode not in DETECT_OBS_MODES:
        return
    sources = os.environ.get("BEHAVIOR_INSTANCE_SOURCES", "").strip()
    if not sources:
        raise ValueError(
            "detect evaluation requires explicit BEHAVIOR_INSTANCE_SOURCES"
        )
    errors = []
    for variable, kind in DETECT_RUNTIME_PATHS.items():
        path = os.environ.get(variable, "").strip()
        valid = os.path.isfile(path) if kind == "file" else os.path.isdir(path)
        if not valid:
            errors.append(f"{variable}={path!r} is not a {kind}")
    if errors:
        raise ValueError("invalid pinned detect runtime: " + "; ".join(errors))


def validate_geometry_instance(layout, requested_key: str | None) -> str | None:
    """Return the actual instance basename and reject frozen-row drift."""
    path = getattr(layout, "instance_path", None) if layout is not None else None
    actual_key = os.path.basename(path) if path else None
    if requested_key and actual_key != requested_key:
        raise ValueError(
            f"manifest requested geometry_instance_key={requested_key!r}, but "
            f"the environment resolved {actual_key!r} from {path!r}"
        )
    return actual_key


def outcome_from_goal_status(goal_status: dict, ended: bool) -> dict:
    """Return the benchmark's goal and proper-completion endpoints."""
    satisfied = list(goal_status.get("satisfied") or [])
    unsatisfied = list(goal_status.get("unsatisfied") or [])
    num_goal = len(satisfied) + len(unsatisfied)
    goal_satisfied = bool(num_goal and satisfied and not unsatisfied)
    return {
        "num_satisfied": len(satisfied),
        "num_goal": num_goal,
        "coverage": len(satisfied) / num_goal if num_goal else 0.0,
        "success": goal_satisfied,
        "proper_completion": bool(goal_satisfied and ended),
    }


__all__ = [
    "DETECT_OBS_MODES",
    "MAX_NUDGES",
    "NUDGE",
    "outcome_from_goal_status",
    "validate_detect_runtime",
    "validate_geometry_instance",
]
