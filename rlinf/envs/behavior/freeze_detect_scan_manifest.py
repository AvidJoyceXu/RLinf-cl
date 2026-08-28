"""Freeze a compact claim-facing manifest from scan reference certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def build_manifest(
    input_manifest: dict[str, Any],
    reference: dict[str, Any],
    identifiability: dict[str, Any],
    *,
    reference_file_sha256: str,
    identifiability_file_sha256: str,
) -> dict[str, Any]:
    """Return a compact frozen denominator after cross-certificate validation."""
    if reference.get("obs_mode") != "detect_scope_scan":
        raise ValueError("reference is not detect_scope_scan")
    if reference.get("input_manifest_content_sha256") != input_manifest.get(
        "content_sha256"
    ):
        raise ValueError("input/reference content hash mismatch")
    if identifiability.get("schema") != "behavior-detect-scan-identifiability-v1":
        raise ValueError("unexpected identifiability schema")

    identifiable = {
        row["activity"]
        for row in identifiability["episodes"]
        if row.get("policy_identifiable")
    }
    accepted_results = [
        row
        for row in reference["results"]
        if row.get("proper_completion")
        and row.get("policy_legal")
        and row["activity"] in identifiable
    ]
    if len(accepted_results) != identifiability["eligible"]["policy_identifiable"]:
        raise ValueError("reference/identifiability accepted counts disagree")
    by_activity = {row["activity"]: row for row in accepted_results}
    rows = []
    for source_row in input_manifest["rows"]:
        result = by_activity.get(source_row["activity"])
        if result is None:
            continue
        rows.append(
            {
                **source_row,
                "supported_obs_modes": sorted(
                    set(source_row.get("supported_obs_modes", []))
                    | {"detect_scope_scan"}
                ),
                "oracle_turns": int(result["turns"]),
                "trace_file": result["trace_file"],
            }
        )
    if len(rows) != len(accepted_results):
        raise ValueError("accepted activities are not a subset of the input manifest")

    exclusions = []
    for result in reference["results"]:
        if result["activity"] in by_activity:
            continue
        audit_row = next(
            (
                row
                for row in identifiability["episodes"]
                if row["activity"] == result["activity"]
            ),
            None,
        )
        exclusions.append(
            {
                "activity": result["activity"],
                "split": result["split"],
                "failure": result.get("failure"),
                "proper_completion": bool(result.get("proper_completion")),
                "policy_identifiable": bool(
                    audit_row and audit_row.get("policy_identifiable")
                ),
            }
        )

    manifest = {
        "schema": "text-detect-public-scan-reference-v2",
        "input_manifest_content_sha256": input_manifest["content_sha256"],
        "obs_mode": "detect_scope_scan",
        "selection_mode": "handle",
        "oracle_controller": {
            "search": "target-independent no-argument scan_next with public cursor",
            "identity": "stable public sN track; actions use current view-local dN",
            "binding": "first compatible unbound public handle per goal role",
        },
        "reference": {
            "content_sha256": reference["content_sha256"],
            "file_sha256": reference_file_sha256,
        },
        "identifiability": {
            "schema": identifiability["schema"],
            "file_sha256": identifiability_file_sha256,
            "violation_counts": {
                key: identifiability["eligible"][key]
                for key in (
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
            },
        },
        "counts": {
            "input": len(input_manifest["rows"]),
            "certified": len(rows),
            "by_split": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "s2")
            },
            "excluded": len(exclusions),
        },
        "rows": rows,
        "exclusions": exclusions,
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--identifiability", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with args.input_manifest.open() as stream:
        input_manifest = json.load(stream)
    with args.reference.open() as stream:
        reference = json.load(stream)
    with args.identifiability.open() as stream:
        identifiability = json.load(stream)
    manifest = build_manifest(
        input_manifest,
        reference,
        identifiability,
        reference_file_sha256=_file_sha256(args.reference),
        identifiability_file_sha256=_file_sha256(args.identifiability),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"content_sha256={manifest['content_sha256']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
