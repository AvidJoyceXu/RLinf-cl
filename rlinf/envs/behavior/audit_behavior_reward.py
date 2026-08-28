"""Replay certified TextWorld traces through grounded-atom reward v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rlinf.algorithms.rewards.behavior import compute_staged_rewards
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld


def _content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def replay_trace(document: dict[str, Any]) -> dict[str, Any]:
    """Replay one public trace and return reward evidence plus violations."""
    summary = document["summary"]
    env = BehaviorTextWorld(
        summary["activity"],
        obs_mode=summary["obs_mode"],
        selection_mode="handle",
        instance_source_selection=summary.get("instance_source"),
    )
    start = env.start(
        instance_source=summary.get("instance_source"),
        geometry_instance_key=summary.get("geometry_instance_key"),
    )
    tool_trace = []
    action_mismatches = 0
    refused_calls = 0
    for expected in document["trace"]:
        name = expected["name"]
        result = env.call(name, expected.get("arguments") or {})
        payload = result["payload"]
        tool_trace.append(
            {
                "name": name,
                "arguments": expected.get("arguments") or {},
                "result": payload,
                "meta": result["meta"],
            }
        )
        expected_ok = expected.get("ok")
        actual_ok = payload.get("ok") if isinstance(payload, dict) else None
        if (
            expected_ok is not None
            and actual_ok is not None
            and bool(expected_ok) != bool(actual_ok)
        ):
            action_mismatches += 1
        if (
            name != "end_task"
            and isinstance(payload, dict)
            and payload.get("ok") is False
        ):
            refused_calls += 1

    final_meta = tool_trace[-1]["meta"] if tool_trace else {}
    reward = float(sum(compute_staged_rewards(tool_trace)))
    initial = float(start["initial_goal_progress"]["atom_coverage"])
    final = float(final_meta.get("atom_coverage", 0.0))
    expected_reward = 1.0 + 0.3 * (1.0 - initial)
    violations = {
        "action_mismatches": action_mismatches,
        "refused_calls": refused_calls,
        "initial_atom_saturated": int(not start.get("contaminated") and initial >= 1.0),
        "final_atom_incomplete": int(final != 1.0),
        "not_proper_completion": int(
            not tool_trace
            or tool_trace[-1]["name"] != "end_task"
            or not final_meta.get("is_success")
        ),
        "reward_formula_mismatch": int(abs(reward - expected_reward) > 1e-9),
    }
    return {
        "activity": summary["activity"],
        "split": summary["split"],
        "initial_atom_coverage": initial,
        "final_atom_coverage": final,
        "atom_goal": int(final_meta.get("atom_goal", 0)),
        "atom_options": int(final_meta.get("atom_options", 0)),
        "reward": reward,
        **violations,
        "passed": not any(violations.values()),
    }


def audit_directory(path: Path) -> dict[str, Any]:
    episodes = []
    excluded = []
    for trace_path in sorted(path.glob("*.json")):
        with trace_path.open() as stream:
            document = json.load(stream)
        if not document["summary"].get("proper_completion"):
            excluded.append(document["summary"]["activity"])
            continue
        episodes.append(replay_trace(document))
    if not episodes:
        raise ValueError(f"no traces under {path}")
    rewards = [row["reward"] for row in episodes]
    numeric_violations = (
        "action_mismatches",
        "refused_calls",
        "initial_atom_saturated",
        "final_atom_incomplete",
        "not_proper_completion",
        "reward_formula_mismatch",
    )
    document = {
        "schema": "behavior-grounded-reward-audit-v1",
        "contract": (
            "terminal + 0.30*(C_final-C_initial) - 0.05*N_format - 0.02*N_refused"
        ),
        "counts": {
            "episodes": len(episodes),
            "excluded_reference_failures": len(excluded),
            "passed": sum(row["passed"] for row in episodes),
            **{
                key: sum(int(row[key]) for row in episodes)
                for key in numeric_violations
            },
        },
        "reward": {
            "min": min(rewards),
            "mean": sum(rewards) / len(rewards),
            "max": max(rewards),
        },
        "excluded_reference_failures": sorted(excluded),
        "episodes": episodes,
    }
    document["content_sha256"] = _content_hash(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    document = audit_directory(args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(document["counts"], sort_keys=True))
    print(json.dumps(document["reward"], sort_keys=True))
    print(f"content_sha256={document['content_sha256']}")
    if document["counts"]["passed"] != document["counts"]["episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
