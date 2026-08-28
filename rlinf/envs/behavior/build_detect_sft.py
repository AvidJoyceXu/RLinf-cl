"""Build matched NL/handle SFT rows from policy-legal detect reference traces.

Only certified training activities are emitted.  Audit-only trace fields (scope
identity, view coordinates, epochs) are deliberately discarded; each tool result is
exactly the payload visible to the policy.  The output is a raw trajectory dataset:
token-length audit and any loss-windowing happen as a separate, recorded step rather
than silently dropping middle turns in this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spatialcode.sft_format import format_tool_response

from rlinf.envs.behavior.nl_goal import render_goal_nl
from rlinf.envs.behavior.sft_build import (
    all_schemas,
    build_prompt_messages,
)
from rlinf.envs.behavior.symbolic_world import SymbolicWorld


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_row(result: dict, trace_path: Path) -> dict:
    with trace_path.open() as stream:
        trace_artifact = json.load(stream)
    trace = trace_artifact["trace"]
    summary = trace_artifact["summary"]
    if not summary.get("proper_completion") or not summary.get("policy_legal"):
        raise ValueError(f"uncertified trace: {trace_path}")
    if summary["activity"] != result["activity"]:
        raise ValueError(
            f"trace/certificate activity mismatch: {summary['activity']} != "
            f"{result['activity']}"
        )
    if len(trace) != result["turns"]:
        raise ValueError(
            f"trace length mismatch for {result['activity']}: "
            f"{len(trace)} != {result['turns']}"
        )
    obs_mode = result.get("obs_mode", "detect_scope")

    world = SymbolicWorld(
        result["activity"],
        instance_source_selection=result["instance_source"],
    )
    goal_lines = []
    for atom in world.ground_goal_atoms():
        body = list(getattr(atom, "body", atom))
        negative = bool(body and body[0] == "not")
        if negative:
            body = list(body[1])
        goal_lines.append(
            ("not " if negative else "")
            + f"{body[0]}({', '.join(str(value) for value in body[1:])})"
        )
    goal_nl, fallbacks = render_goal_nl(result["activity"])
    if not goal_nl:
        raise ValueError(f"no NL goal for {result['activity']}")

    tool_steps = [
        {
            "name": step["name"],
            "arguments": dict(step.get("arguments") or {}),
            # Only `result`, never step.audit, enters the training context.
            "result_text": format_tool_response(step["result"]),
        }
        for step in trace
    ]
    return {
        "activity": result["activity"],
        "task": result["activity"].replace("_", " "),
        "split": result["split"],
        "instance_source": result["instance_source"],
        "geometry_instance_key": result["geometry_instance_key"],
        "success": True,
        "num_turns": len(tool_steps),
        "prompt_messages": build_prompt_messages(
            result["activity"],
            goal_lines,
            obs_mode,
            goal_nl=goal_nl,
            selection_mode="handle",
        ),
        "tool_steps": tool_steps,
        "goal_nl_fallbacks": fallbacks,
        "reference_certificate_turns": result["turns"],
        "obs_mode": obs_mode,
        "answer": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train", choices=["train", "s2"])
    args = parser.parse_args()

    certificate_path = Path(args.certificate)
    with certificate_path.open() as stream:
        certificate = json.load(stream)
    obs_mode = certificate.get("obs_mode", "detect_scope")
    trace_dir = Path(args.trace_dir)
    rows = []
    for result in certificate["results"]:
        if result["split"] != args.split or not result["proper_completion"]:
            continue
        trace_file = result.get("trace_file")
        if not trace_file:
            raise ValueError(
                "certificate has no trace_file entries; rerun detect_reference with "
                "--trace-dir"
            )
        rows.append(build_row(result, trace_dir / trace_file))
    if not rows:
        raise ValueError(f"no certified {args.split} rows")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    schemas = all_schemas(obs_mode, selection_mode="handle")
    schema_path = Path(str(output) + ".tools.json")
    with schema_path.open("w") as stream:
        json.dump(schemas, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    turns = sorted(row["num_turns"] for row in rows)
    manifest = {
        "schema": (
            "behavior-detect-scan-sft-raw-v2"
            if obs_mode == "detect_scope_scan"
            else "behavior-detect-sft-raw-v1"
        ),
        "input_certificate_content_sha256": certificate["content_sha256"],
        "split": args.split,
        "goal_format": "nl",
        "obs_mode": obs_mode,
        "selection_mode": "handle",
        "rows": len(rows),
        "activities": [row["activity"] for row in rows],
        "turns": {
            "min": min(turns),
            "median": turns[len(turns) // 2],
            "p90": turns[int(0.9 * (len(turns) - 1))],
            "max": max(turns),
        },
        "dataset_sha256": _sha256(output),
        "tool_schemas_sha256": _sha256(schema_path),
        "training_ready": False,
        "training_ready_blocker": (
            "token-length audit and explicit contiguous loss windows are required; "
            "the legacy loader drops middle turns"
        ),
    }
    manifest_path = Path(str(output) + ".manifest.json")
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
