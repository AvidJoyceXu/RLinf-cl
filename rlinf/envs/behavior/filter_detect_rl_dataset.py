"""Filter a frozen detect RL JSONL to an explicit reproducible smoke subset."""

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


def filter_rows(input_path: Path, output: Path, requested: list[str]) -> dict:
    """Write requested rows in declared order and return their provenance."""
    if len(requested) != len(set(requested)):
        raise ValueError("activity list contains duplicates")
    rows = {}
    with input_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            spec = json.loads(row["solutions"])
            activity = str(spec["activity"])
            if activity in rows:
                raise ValueError(f"duplicate input activity: {activity}")
            rows[activity] = row
    missing = sorted(set(requested) - rows.keys())
    if missing:
        raise ValueError(f"requested activities are absent from input: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for activity in requested:
            stream.write(json.dumps(rows[activity], ensure_ascii=False) + "\n")
    return {
        "input": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "activities": requested,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--activities", nargs="+", required=True)
    args = parser.parse_args()
    result = filter_rows(Path(args.input), Path(args.output), list(args.activities))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
