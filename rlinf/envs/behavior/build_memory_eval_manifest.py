"""Freeze the matched TextWorld memory evaluation denominator.

The target panel contains five held-out activities from one scene. Every target is
evaluated at every virtual session position (1..5). Same-scene history comes from a
cyclic permutation of the other panel activities; different-scene history is matched
one prior at a time by oracle turns while using distinct donor scenes. This file builds
the cases only. It never calls a model and never treats context-only donor outcomes as
target scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "activity": row["activity"],
        "split": row["split"],
        "scene": row["scene"],
        "instance_source": row["instance_source"],
        "geometry_instance_key": row["geometry_instance_key"],
        "oracle_turns": int(row["oracle_turns"]),
    }


def _matched_donors(
    same_scene_prior: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    target_scene: str,
) -> list[dict[str, Any]]:
    chosen = []
    used_scenes = {target_scene}
    for source in same_scene_prior:
        candidates = [row for row in donor_pool if row["scene"] not in used_scenes]
        if not candidates:
            raise ValueError(
                "not enough distinct donor scenes for diff-scene context: "
                f"need {len(same_scene_prior)}, used={sorted(used_scenes)}"
            )
        donor = min(
            candidates,
            key=lambda row: (
                abs(int(row["oracle_turns"]) - int(source["oracle_turns"])),
                row["split"] != "s2",
                row["activity"],
            ),
        )
        chosen.append(_compact(donor))
        used_scenes.add(donor["scene"])
    return chosen


def build_manifest(
    source: dict[str, Any],
    *,
    panel_scene: str = "house_single_floor",
    panel_size: int = 5,
) -> dict[str, Any]:
    """Build counterbalanced memory cases from a certified reference manifest."""
    if source.get("obs_mode") != "detect_scope_scan":
        raise ValueError("source manifest must certify obs_mode=detect_scope_scan")
    rows = list(source.get("rows") or [])
    panel = sorted(
        (
            row
            for row in rows
            if row.get("split") == "s2" and row.get("scene") == panel_scene
        ),
        key=lambda row: row["activity"],
    )
    if len(panel) != panel_size:
        raise ValueError(
            f"expected exactly {panel_size} S2 rows in {panel_scene}, got {len(panel)}"
        )
    donor_pool = sorted(
        (row for row in rows if row.get("scene") != panel_scene),
        key=lambda row: (row["scene"], row["activity"]),
    )
    donor_scenes = {row["scene"] for row in donor_pool}
    if len(donor_scenes) < panel_size - 1:
        raise ValueError(
            f"need at least {panel_size - 1} donor scenes, got {len(donor_scenes)}"
        )

    cases = []
    for target_index, target in enumerate(panel):
        for position_index in range(panel_size):
            session = [
                panel[(target_index - position_index + offset) % panel_size]
                for offset in range(panel_size)
            ]
            if session[position_index]["activity"] != target["activity"]:
                raise AssertionError("cyclic counterbalance did not place target")
            same_scene_prior = [_compact(row) for row in session[:position_index]]
            diff_scene_prior = _matched_donors(
                session[:position_index], donor_pool, panel_scene
            )
            cases.append(
                {
                    "case_id": f"{target['activity']}__p{position_index + 1}",
                    "position": position_index + 1,
                    "target": _compact(target),
                    "fresh_key": target["activity"],
                    "same_scene_prior": same_scene_prior,
                    "diff_scene_prior": diff_scene_prior,
                    "summary_source": "same_scene_prior",
                }
            )

    document = {
        "schema": "behavior-text-memory-v1",
        "source_manifest_content_sha256": source.get("content_sha256"),
        "obs_mode": "detect_scope_memory",
        "panel": {
            "scene": panel_scene,
            "split": "s2",
            "size": panel_size,
            "activities": [row["activity"] for row in panel],
        },
        "arms": ["fresh", "same-scene", "diff-scene", "summary"],
        "contract": {
            "primary_score": "target activity only; prior activities are context",
            "counterbalance": "every target appears once at each position 1..5",
            "fresh": "one context-free result per target, reused across positions",
            "same_scene": "completed prior activities from the target scene",
            "diff_scene": (
                "oracle-turn-matched priors from mutually distinct non-target scenes"
            ),
            "summary": (
                "deterministic compression of public scene_id/lN/category associations"
            ),
            "context_match": (
                "match same/diff target-request token counts by complete-turn compaction"
            ),
            "identity": (
                "scene_id and lN persist within a scene; sN resets per activity"
            ),
        },
        "counts": {
            "cases": len(cases),
            "targets": len(panel),
            "positions": panel_size,
            "donor_scenes": len(donor_scenes),
        },
        "cases": cases,
    }
    document["content_sha256"] = _canonical_hash(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panel-scene", default="house_single_floor")
    parser.add_argument("--panel-size", type=int, default=5)
    args = parser.parse_args()

    with args.source.open() as stream:
        source = json.load(stream)
    document = build_manifest(
        source,
        panel_scene=args.panel_scene,
        panel_size=args.panel_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(document["counts"], sort_keys=True))
    print(f"content_sha256={document['content_sha256']}")


if __name__ == "__main__":
    main()
