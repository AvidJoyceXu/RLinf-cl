"""Build RL prompt rows from the frozen policy-legal detect certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_row(result: dict) -> dict:
    spec = {
        "activity": result["activity"],
        "instance_id": 0,
        "instance_source": result["instance_source"],
        "geometry_instance_key": result["geometry_instance_key"],
        "oracle_turns": int(result["turns"]),
        "benchmark_split": result["split"],
    }
    return {
        "prompt": result["activity"].replace("_", " "),
        "solutions": json.dumps(spec, ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    certificate_path = Path(args.certificate)
    with certificate_path.open() as stream:
        certificate = json.load(stream)
    obs_mode = certificate.get("obs_mode", "detect_scope")
    rows = [
        result
        for result in certificate["results"]
        if result.get("proper_completion") and result.get("policy_legal")
    ]
    if len(rows) != certificate["counts"]["proper_completion"]:
        raise ValueError("certificate counts disagree with policy-legal result rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, filename in (("train", "train.jsonl"), ("s2", "val_s2.jsonl")):
        path = out_dir / filename
        selected = sorted(
            (result for result in rows if result["split"] == split),
            key=lambda result: result["activity"],
        )
        with path.open("w") as stream:
            for result in selected:
                stream.write(
                    json.dumps(_dataset_row(result), ensure_ascii=False) + "\n"
                )
        outputs[split] = {
            "path": filename,
            "rows": len(selected),
            "activities": [result["activity"] for result in selected],
            "sha256": _sha256(path),
        }

    manifest = {
        "schema": (
            "behavior-detect-scan-rl-dataset-v2"
            if obs_mode == "detect_scope_scan"
            else "behavior-detect-rl-dataset-v1"
        ),
        "reference_content_sha256": certificate["content_sha256"],
        "reference_file_sha256": _sha256(certificate_path),
        "goal_format": "nl",
        "obs_mode": obs_mode,
        "selection_mode": "handle",
        "budget": "ceil(K * oracle_turns), selected at run time",
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
