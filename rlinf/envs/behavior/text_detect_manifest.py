"""Freeze the geometry-bearing input rows for the text detect benchmark.

This is a simulator-free provenance gate: it reads BDDL, sampled task snapshots,
paired templates, and the native-bbox artifact, but never imports Isaac Sim or boots
a renderer.  The output contains one deterministic compatible instance per activity
from the already-frozen TextWorld split.  It does not certify sequential solvability;
that is the next gate and must consume this manifest without redrawing membership.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "text-detect-v1"
EXPECTED_SEED = 0
EXPECTED_TRAIN = 592
EXPECTED_S2 = 148
DEFAULT_SOURCES = "2026-v3.9.1,2025-official"


@functools.lru_cache(maxsize=None)
def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_frozen_split(path: str, *, allow_nonfrozen: bool = False) -> tuple[dict, str]:
    with open(path) as stream:
        split = json.load(stream)
    train = split.get("train_activities")
    s2 = split.get("s2_heldout_activities")
    if not isinstance(train, list) or not isinstance(s2, list):
        raise ValueError(
            "split must contain train_activities and s2_heldout_activities"
        )
    overlap = sorted(set(train) & set(s2))
    if overlap:
        raise ValueError(f"train/S2 activity overlap: {overlap[:5]}")
    if len(train) != len(set(train)) or len(s2) != len(set(s2)):
        raise ValueError("split contains duplicate activities")
    if not allow_nonfrozen:
        actual = (split.get("seed"), len(train), len(s2))
        expected = (EXPECTED_SEED, EXPECTED_TRAIN, EXPECTED_S2)
        if actual != expected:
            raise ValueError(
                f"refusing a redrawn split: expected seed/train/S2={expected}, got {actual}"
            )
    return split, _sha256(path)


def _layout_payload(layout) -> dict:
    return {
        "scene": layout.scene,
        "robot_xy": list(layout.robot_xy),
        "robot_yaw": layout.robot_yaw,
        "xyz": {name: list(layout.xyz[name]) for name in sorted(layout.xyz)},
        "orientation": {
            name: list(layout.orientation[name]) for name in sorted(layout.orientation)
        },
    }


def _geometry_instance_id(path: Path) -> int:
    match = re.search(r"_(\d+)_(\d+)_template-tro_state\.json$", path.name)
    if match is None:
        raise ValueError(f"cannot parse geometry instance id from {path.name!r}")
    return int(match.group(2))


def _build_row(activity: str, split_name: str, source, bddl_root: str, bbox_path: str):
    from rlinf.envs.behavior.detect import scope_assets
    from rlinf.envs.behavior.instance_compatibility import template_path_for
    from rlinf.envs.behavior.layout import build_layout
    from rlinf.envs.behavior.symbolic_world import (
        SymbolicACI,
        SymbolicWorld,
        properties_of,
    )

    world = SymbolicWorld(
        activity,
        instance_source_selection=source.name,
    )
    layout = build_layout(
        activity,
        world,
        prefer="sampled",
        source_selection=source.name,
        require_detect_models=True,
    )
    if not layout.is_real or layout.instance_source != source.name:
        raise ValueError("no compatible sampled instance")

    # Construct the exact policy environment, not merely a file-level match.  This
    # catches missing scene data, model binding, or bbox extents before a row freezes.
    aci = SymbolicACI(
        world,
        obs_mode="detect_scope",
        selection_mode="handle",
        layout_source_selection=source.name,
    )
    visual_targets = sorted(
        name
        for name in world.scope_names
        if name != world.agent
        and world.is_real(name)
        and "substance" not in properties_of(name)
    )
    assets = scope_assets(layout.instance_path)
    missing_pose = [name for name in visual_targets if aci._pos(name) is None]
    missing_model = [name for name in visual_targets if name not in assets]
    missing_bbox = [
        name for name in visual_targets if aci._observation_bbox(name) is None
    ]
    if missing_pose or missing_model or missing_bbox:
        raise ValueError(
            "incomplete detect geometry: "
            f"pose={missing_pose[:3]} model={missing_model[:3]} "
            f"bbox={missing_bbox[:3]}"
        )

    instance_path = Path(layout.instance_path)
    template_path = template_path_for(layout.instance_path)
    problem_path = Path(bddl_root) / activity / "problem0.bddl"
    return {
        "activity": activity,
        "split": split_name,
        "instance_source": source.name,
        "scene": layout.scene,
        "instance_file": str(instance_path.relative_to(source.root)),
        # Symbolic BDDL has one definition; the sampled geometry has its own ID.
        "instance_id": 0,
        "geometry_instance_id": _geometry_instance_id(instance_path),
        "geometry_instance_key": instance_path.name,
        "instance_sha256": _sha256(instance_path),
        "template_file": str(template_path.relative_to(source.root)),
        "template_sha256": _sha256(template_path),
        "bddl_file": str(problem_path.relative_to(bddl_root)),
        "bddl_sha256": _sha256(problem_path),
        "bddl_release": layout.bddl_release,
        "asset_release": layout.asset_release,
        "native_bbox_file": Path(bbox_path).name,
        "native_bbox_sha256": _sha256(bbox_path),
        "initial_layout_sha256": _canonical_sha256(_layout_payload(layout)),
        "initial_visual_targets": len(visual_targets),
        "supported_obs_modes": ["detect_scope", "detect"],
        "supported_selection_modes": ["handle", "bbox"],
        "geometry_contract": "sampled_initial+deterministic_symbolic_transition",
    }


def build_manifest(
    split_path: str,
    sources_spec: str,
    bddl_root: str,
    bbox_path: str,
    *,
    allow_nonfrozen_split: bool = False,
) -> dict:
    from rlinf.envs.behavior.instance_sources import selected_layout_sources

    split, split_hash = _load_frozen_split(
        split_path,
        allow_nonfrozen=allow_nonfrozen_split,
    )
    sources = selected_layout_sources(sources_spec)
    if any(not source.detect_eligible for source in sources):
        rejected = [source.name for source in sources if not source.detect_eligible]
        raise ValueError(f"detect-ineligible source(s) requested: {rejected}")
    if not Path(bddl_root).is_dir():
        raise ValueError(f"BDDL root does not exist: {bddl_root}")
    if not Path(bbox_path).is_file():
        raise ValueError(f"native bbox artifact does not exist: {bbox_path}")

    # SymbolicWorld and detect also read these variables.  Pin them inside this
    # process so the hashes below and the geometry we validated cannot disagree.
    os.environ["BEHAVIOR_BDDL_DEFINITION_ROOT"] = str(Path(bddl_root).resolve())
    os.environ["BEHAVIOR_NATIVE_BBOX_PATH"] = str(Path(bbox_path).resolve())
    os.environ["BEHAVIOR_INSTANCE_SOURCES"] = ",".join(
        source.name for source in sources
    )

    rows = []
    excluded = []
    for split_name, key in (
        ("train", "train_activities"),
        ("s2", "s2_heldout_activities"),
    ):
        for activity in sorted(split[key]):
            attempts = []
            for source in sources:
                try:
                    row = _build_row(activity, split_name, source, bddl_root, bbox_path)
                except Exception as error:  # noqa: BLE001 - exclusion is benchmark data
                    attempts.append(
                        {
                            "source": source.name,
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                rows.append(row)
                break
            else:
                excluded.append(
                    {"activity": activity, "split": split_name, "attempts": attempts}
                )

    source_counts = collections.Counter(row["instance_source"] for row in rows)
    split_counts = collections.Counter(row["split"] for row in rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "contract": {
            "benchmark": "BEHAVIOR-TextWorld S1.5 simulated perception",
            "primary": {
                "goal_format": "nl",
                "obs_mode": "detect_scope",
                "selection_mode": "handle",
            },
            "not_claimed": [
                "rendered RGB",
                "physical post-action geometry",
                "multimodal S2",
            ],
        },
        "frozen_split": {
            "file": os.path.relpath(
                Path(split_path),
                os.environ.get("BEHAVIOR_DATA_ROOT", "/data/behavior-data"),
            ),
            "sha256": split_hash,
            "seed": split.get("seed"),
            "input_train": len(split["train_activities"]),
            "input_s2": len(split["s2_heldout_activities"]),
        },
        "source_priority": [source.name for source in sources],
        "counts": {
            "included": len(rows),
            "excluded": len(excluded),
            "by_split": {name: split_counts.get(name, 0) for name in ("train", "s2")},
            "by_source": {
                source.name: source_counts.get(source.name, 0) for source in sources
            },
        },
        "rows": rows,
        "excluded": excluded,
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="/data/behavior-data/tw/split.json")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument(
        "--bddl-root",
        default=os.environ.get("BEHAVIOR_BDDL_DEFINITION_ROOT"),
        required=not bool(os.environ.get("BEHAVIOR_BDDL_DEFINITION_ROOT")),
    )
    parser.add_argument(
        "--bbox",
        default=os.environ.get("BEHAVIOR_NATIVE_BBOX_PATH"),
        required=not bool(os.environ.get("BEHAVIOR_NATIVE_BBOX_PATH")),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-nonfrozen-split",
        action="store_true",
        help="development only; production refuses anything other than seed 0 / 592 / 148",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        args.split,
        args.sources,
        args.bddl_root,
        args.bbox,
        allow_nonfrozen_split=args.allow_nonfrozen_split,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"content_sha256={manifest['content_sha256']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
