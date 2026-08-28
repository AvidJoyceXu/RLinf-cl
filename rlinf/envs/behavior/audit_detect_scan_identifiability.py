"""Certify that scan-reference route and selector labels are public functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMITIVE_CAMERA_TOOLS = frozenset(
    {
        "turn_left",
        "turn_right",
        "move_ahead",
        "move_back",
        "strafe_left",
        "strafe_right",
        "look_up",
        "look_down",
    }
)


def _handle_order(handle: str) -> int:
    return int(handle[1:]) if handle.startswith("d") and handle[1:].isdigit() else 10**9


def _action_compatible(tool: str, detection: dict[str, Any]) -> bool:
    states = detection.get("states") or {}
    if tool == "open":
        return states.get("Open") is False
    if tool == "close":
        return states.get("Open") is True
    if tool == "place_inside" and "Open" in states:
        return states["Open"] is True
    return True


def audit_document(document: dict[str, Any], *, source: str = "") -> dict[str, Any]:
    """Audit one completed reference using only public trace plus audit mappings."""
    summary = document["summary"]
    trace = document["trace"]
    resolutions = sorted(
        summary.get("handle_resolutions", []), key=lambda row: row["turn"]
    )

    track_to_scope: dict[str, str] = {}
    scope_to_track: dict[str, str] = {}
    track_scope_conflicts = 0
    missing_public_tracks = 0
    for step in trace:
        detections = (step.get("result") or {}).get("detections", [])
        handle_to_scope = {
            str(item.get("handle")): str(item.get("scope"))
            for item in (step.get("audit") or {}).get("scope_detections", [])
        }
        for detection in detections:
            handle = str(detection.get("det", ""))
            scope = handle_to_scope.get(handle)
            track = detection.get("track")
            if scope is None:
                continue
            if not track:
                missing_public_tracks += 1
                continue
            if track in track_to_scope and track_to_scope[track] != scope:
                track_scope_conflicts += 1
            if scope in scope_to_track and scope_to_track[scope] != track:
                track_scope_conflicts += 1
            track_to_scope[str(track)] = scope
            scope_to_track[scope] = str(track)

    role_to_track: dict[str, str] = {}
    bound_tracks: set[str] = set()
    missing_selected_handles = 0
    resolution_scope_mismatches = 0
    resolution_track_mismatches = 0
    canonical_binding_mismatches = 0
    bound_reacquisition_mismatches = 0
    for resolution in resolutions:
        turn = int(resolution["turn"])
        preceding = trace[turn - 2] if turn >= 2 else {}
        detections = (preceding.get("result") or {}).get("detections", [])
        handle_to_scope = {
            str(item.get("handle")): str(item.get("scope"))
            for item in (preceding.get("audit") or {}).get("scope_detections", [])
        }
        selected = next(
            (
                detection
                for detection in detections
                if detection.get("det") == resolution.get("handle")
            ),
            None,
        )
        if selected is None:
            missing_selected_handles += 1
            continue
        handle = str(resolution["handle"])
        track = str(selected.get("track") or "")
        scope = handle_to_scope.get(handle)
        if scope != resolution.get("scope"):
            resolution_scope_mismatches += 1
        if not track or track != resolution.get("track"):
            resolution_track_mismatches += 1

        role = str(resolution.get("planned_scope"))
        if resolution.get("binding_created"):
            available = [
                detection
                for detection in detections
                if detection.get("category") == selected.get("category")
                and detection.get("track") not in bound_tracks
                and _action_compatible(str(resolution["tool"]), detection)
            ]
            expected = min(
                available,
                key=lambda detection: _handle_order(str(detection.get("det", ""))),
                default=None,
            )
            if (
                expected is None
                or expected.get("det") != handle
                or role in role_to_track
            ):
                canonical_binding_mismatches += 1
            role_to_track[role] = track
            bound_tracks.add(track)
        elif role_to_track.get(role) != track:
            bound_reacquisition_mismatches += 1

    primitive_camera_actions = sum(
        step.get("name") in PRIMITIVE_CAMERA_TOOLS for step in trace
    )
    scan_steps = [step for step in trace if step.get("name") == "scan_next"]
    scan_argument_violations = sum(bool(step.get("arguments")) for step in scan_steps)
    scan_cursor_violations = sum(
        not isinstance(((step.get("result") or {}).get("view") or {}).get("scan"), dict)
        for step in scan_steps
    )
    violations = {
        "primitive_camera_actions": primitive_camera_actions,
        "scan_argument_violations": scan_argument_violations,
        "scan_cursor_violations": scan_cursor_violations,
        "missing_public_tracks": missing_public_tracks,
        "track_scope_conflicts": track_scope_conflicts,
        "missing_selected_handles": missing_selected_handles,
        "resolution_scope_mismatches": resolution_scope_mismatches,
        "resolution_track_mismatches": resolution_track_mismatches,
        "canonical_binding_mismatches": canonical_binding_mismatches,
        "bound_reacquisition_mismatches": bound_reacquisition_mismatches,
    }
    return {
        "source": source,
        "activity": summary["activity"],
        "split": summary["split"],
        "proper_completion": bool(summary.get("proper_completion")),
        "scan_actions": len(scan_steps),
        "selectors": len(resolutions),
        **violations,
        "policy_identifiable": bool(summary.get("proper_completion"))
        and not any(violations.values()),
    }


def audit_directory(path: Path) -> dict[str, Any]:
    episodes = []
    for trace_path in sorted(path.glob("*.json")):
        with trace_path.open() as stream:
            episodes.append(audit_document(json.load(stream), source=trace_path.name))
    if not episodes:
        raise ValueError(f"no reference trace JSON files found under {path}")

    numeric_keys = (
        "scan_actions",
        "selectors",
        "primitive_camera_actions",
        "scan_argument_violations",
        "scan_cursor_violations",
        "missing_public_tracks",
        "track_scope_conflicts",
        "missing_selected_handles",
        "resolution_scope_mismatches",
        "resolution_track_mismatches",
        "canonical_binding_mismatches",
        "bound_reacquisition_mismatches",
    )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "episodes": len(rows),
            "proper_completion": sum(row["proper_completion"] for row in rows),
            "policy_identifiable": sum(row["policy_identifiable"] for row in rows),
            **{key: sum(row[key] for row in rows) for key in numeric_keys},
        }

    eligible = [row for row in episodes if row["proper_completion"]]
    return {
        "schema": "behavior-detect-scan-identifiability-v1",
        "contract": {
            "route": "no-argument scan_next with public cursor",
            "identity": "stable public sN track across view-local dN handles",
            "binding": "first compatible unbound public handle per goal role",
        },
        "all": aggregate(episodes),
        "eligible": aggregate(eligible),
        "by_split": {
            split: aggregate([row for row in eligible if row["split"] == split])
            for split in sorted({row["split"] for row in eligible})
        },
        "excluded": [row for row in episodes if not row["proper_completion"]],
        "episodes": sorted(episodes, key=lambda row: (row["split"], row["activity"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit_directory(args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("eligible", "by_split")}, indent=2))


if __name__ == "__main__":
    main()
