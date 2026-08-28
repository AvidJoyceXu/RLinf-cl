"""Audit whether detect-reference camera labels are identifiable by the policy.

The sequential reference certificate proves that a trace uses public tools and
finishes the task.  It does not prove that the expert's next camera action is a
function of the policy observation.  ``DetectReference`` currently chooses
translations from hidden target geometry.  This audit quantifies how much of a
frozen reference corpus is generated while the target is absent, and checks that
the hidden camera position used by the planner is not present in the public payload.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

CAMERA_TOOLS = {
    "turn_left",
    "turn_right",
    "move_ahead",
    "move_back",
    "strafe_left",
    "strafe_right",
    "look_up",
    "look_down",
}
TRANSLATION_TOOLS = {"move_ahead", "move_back", "strafe_left", "strafe_right"}


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.floor(fraction * (len(ordered) - 1))]


def _target_visible(observation: dict[str, Any], target: str) -> bool:
    audit = observation.get("audit") or {}
    return any(
        item.get("scope") == target for item in audit.get("scope_detections", [])
    )


def audit_trace(document: dict[str, Any], *, source: str = "") -> dict[str, Any]:
    """Return one episode's public/privileged camera-label diagnostics."""
    summary = document["summary"]
    trace = document["trace"]
    resolutions = summary.get("handle_resolutions", [])
    tool_counts = collections.Counter(step["name"] for step in trace)

    observations = [step for step in trace if step["name"] == "observe"]
    hidden_xy = 0
    payload_xy = 0
    handle_payloads_with_geometry = 0
    for step in observations:
        public = step.get("result") or {}
        audit = step.get("audit") or {}
        if "view_xy" in audit:
            hidden_xy += 1
        view = public.get("view") or {}
        if any(key in view for key in ("x", "y", "xy", "position")):
            payload_xy += 1
        for detection in public.get("detections", []):
            if any(key in detection for key in ("bbox", "where", "position", "depth")):
                handle_payloads_with_geometry += 1

    acquisitions = []
    ambiguous_selectors = 0
    ambiguous_selectors_with_unselected_peer = 0
    missing_selected_handles = 0
    selected_scopes = {str(resolution["scope"]) for resolution in resolutions}
    previous_selected_turn = 0
    for resolution in resolutions:
        selected_turn = int(resolution["turn"])
        # Turns are one-indexed. The slice begins immediately after the previous
        # selector and ends immediately before the current selector.
        segment = trace[previous_selected_turn : selected_turn - 1]
        first_observation = next(
            (step for step in segment if step["name"] == "observe"), None
        )
        camera = [step["name"] for step in segment if step["name"] in CAMERA_TOOLS]
        translations = [name for name in camera if name in TRANSLATION_TOOLS]
        initially_visible = bool(
            first_observation
            and _target_visible(first_observation, str(resolution["scope"]))
        )
        preceding = trace[selected_turn - 2] if selected_turn >= 2 else None
        detections = (
            (preceding.get("result") or {}).get("detections", [])
            if preceding and preceding.get("name") == "observe"
            else []
        )
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
            same_category_candidates = 0
        else:
            same_category_handles = {
                str(detection.get("det"))
                for detection in detections
                if detection.get("category") == selected.get("category")
            }
            same_category_candidates = len(same_category_handles)
            ambiguous_selectors += same_category_candidates > 1
            handle_to_scope = {
                str(item.get("handle")): str(item.get("scope"))
                for item in (preceding.get("audit") or {}).get("scope_detections", [])
            }
            same_category_scopes = {
                handle_to_scope[handle]
                for handle in same_category_handles
                if handle in handle_to_scope
            }
            ambiguous_selectors_with_unselected_peer += (
                same_category_candidates > 1
                and bool(same_category_scopes - selected_scopes)
            )
        acquisitions.append(
            {
                "turn": selected_turn,
                "tool": resolution["tool"],
                "target": resolution["scope"],
                "initially_visible": initially_visible,
                "camera_actions": len(camera),
                "translation_actions": len(translations),
                "same_category_candidates": same_category_candidates,
            }
        )
        previous_selected_turn = selected_turn

    absent = [item for item in acquisitions if not item["initially_visible"]]
    hidden_routes = [item for item in absent if item["translation_actions"]]
    return {
        "source": source,
        "activity": summary["activity"],
        "split": summary["split"],
        "policy_legal": bool(summary.get("policy_legal")),
        "proper_completion": bool(summary.get("proper_completion")),
        "turns": len(trace),
        "tool_counts": dict(sorted(tool_counts.items())),
        "camera_actions": sum(tool_counts[name] for name in CAMERA_TOOLS),
        "translation_actions": sum(tool_counts[name] for name in TRANSLATION_TOOLS),
        "observations": len(observations),
        "observations_with_hidden_view_xy": hidden_xy,
        "observations_with_public_view_xy": payload_xy,
        "handle_detections_with_public_geometry": handle_payloads_with_geometry,
        "acquisitions": len(acquisitions),
        "same_category_ambiguous_selectors": ambiguous_selectors,
        "ambiguous_selectors_with_unselected_peer": (
            ambiguous_selectors_with_unselected_peer
        ),
        "missing_selected_handles": missing_selected_handles,
        "target_absent_at_acquisition_start": len(absent),
        "absent_target_acquisitions_with_translation": len(hidden_routes),
        "camera_actions_for_absent_targets": sum(
            item["camera_actions"] for item in absent
        ),
        "translations_for_absent_targets": sum(
            item["translation_actions"] for item in absent
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [row["turns"] for row in rows]
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "turns",
            "camera_actions",
            "translation_actions",
            "observations",
            "observations_with_hidden_view_xy",
            "observations_with_public_view_xy",
            "handle_detections_with_public_geometry",
            "acquisitions",
            "same_category_ambiguous_selectors",
            "ambiguous_selectors_with_unselected_peer",
            "missing_selected_handles",
            "target_absent_at_acquisition_start",
            "absent_target_acquisitions_with_translation",
            "camera_actions_for_absent_targets",
            "translations_for_absent_targets",
        )
    }
    acquisitions = totals["acquisitions"]
    absent = totals["target_absent_at_acquisition_start"]
    return {
        "episodes": len(rows),
        **totals,
        "turns_distribution": {
            "min": min(turns),
            "median": _percentile(turns, 0.5),
            "p90": _percentile(turns, 0.9),
            "max": max(turns),
        },
        "camera_action_rate": totals["camera_actions"] / totals["turns"],
        "target_absent_acquisition_rate": absent / acquisitions if acquisitions else 0,
        "same_category_ambiguous_selector_rate": (
            totals["same_category_ambiguous_selectors"] / acquisitions
            if acquisitions
            else 0
        ),
        "ambiguous_with_unselected_peer_rate": (
            totals["ambiguous_selectors_with_unselected_peer"] / acquisitions
            if acquisitions
            else 0
        ),
        "absent_target_translation_rate": (
            totals["absent_target_acquisitions_with_translation"] / absent
            if absent
            else 0
        ),
    }


def audit_directory(path: Path) -> dict[str, Any]:
    rows = []
    for trace_path in sorted(path.glob("*.json")):
        with trace_path.open() as stream:
            rows.append(audit_trace(json.load(stream), source=trace_path.name))
    if not rows:
        raise ValueError(f"no reference trace JSON files found under {path}")
    eligible = [row for row in rows if row["policy_legal"] and row["proper_completion"]]
    by_split = {
        split: _aggregate([row for row in eligible if row["split"] == split])
        for split in sorted({row["split"] for row in eligible})
    }
    return {
        "schema": "behavior-detect-reference-identifiability-v1",
        "interpretation": {
            "certificate": "public-action sequential solvability",
            "not_certified": "policy-observation identifiability of expert route labels",
            "hidden_signal": "planner view_xy and target bbox used to choose translations",
            "policy_signal": "facing/pitch plus handle/category/score detections",
        },
        "eligible": _aggregate(eligible),
        "by_split": by_split,
        "excluded": [
            {
                "source": row["source"],
                "activity": row["activity"],
                "policy_legal": row["policy_legal"],
                "proper_completion": row["proper_completion"],
            }
            for row in rows
            if row not in eligible
        ],
        "episodes": sorted(rows, key=lambda row: (row["split"], row["activity"])),
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
    print(
        json.dumps(
            {key: report[key] for key in ("schema", "eligible", "by_split")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
