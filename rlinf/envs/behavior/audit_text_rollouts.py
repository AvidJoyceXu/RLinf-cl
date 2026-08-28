"""Audit matched HF/API rollout JSONL without loading a model or environment."""

from __future__ import annotations

import argparse
import json
from collections import Counter

INTERFACE_FAILURE_PATTERNS = {
    "stale_handle": ("last observation is stale",),
    "non_handle_selection": ("selection_mode=handle requires",),
    "missing_current_handle": ("not in the current view",),
    "held_misuse": ("held is only valid", "not holding anything"),
}
CLAIM_PROVENANCE_FIELDS = (
    "adapter",
    "model_id",
    "activities_sha256",
    "replicate_id",
    "rlinf_revision",
    "outer_revision",
    "container_image",
    "prompt_mode",
    "reason_mode",
    "decoding",
    "instance_source",
    "geometry_instance_key",
    "obs_mode",
    "goal_format",
    "selection_mode",
    "oracle_steps",
    "budget_k",
    "max_turns",
)


def audit_records(records: list[dict]) -> dict:
    """Aggregate claim-facing metrics and interface violations."""
    model_turns = sum(int(row.get("model_turns", 0)) for row in records)
    actions = sum(int(row.get("n_steps", 0)) for row in records)
    interface_failures: Counter[str] = Counter()
    refused_calls = 0
    tool_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    rejection_tools: Counter[str] = Counter()
    repeated_action_calls = 0
    repeated_non_observe_action_calls = 0
    consecutive_duplicate_calls = 0
    episode_diagnostics = []
    missing_provenance: Counter[str] = Counter()
    for row in records:
        episode_refused = 0
        episode_tool_counts: Counter[str] = Counter()
        for field in CLAIM_PROVENANCE_FIELDS:
            if row.get(field) is None:
                missing_provenance[field] += 1
        steps = row.get("tool_trace") or []
        signatures = []
        non_observe_signatures = []
        for step in steps:
            name = str(step.get("name", ""))
            tool_counts[name] += 1
            episode_tool_counts[name] += 1
            signature = json.dumps(
                [name, step.get("arguments") or {}],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            signatures.append(signature)
            if name != "observe":
                non_observe_signatures.append(signature)
            result = step.get("result", step.get("payload", {})) or {}
            if result.get("ok") is not False:
                continue
            if name == "end_task":
                continue
            refused_calls += 1
            episode_refused += 1
            reason = str(result.get("reason", "")).lower()
            rejection_reasons[reason] += 1
            rejection_tools[name] += 1
            for label, patterns in INTERFACE_FAILURE_PATTERNS.items():
                if any(pattern in reason for pattern in patterns):
                    interface_failures[label] += 1
        episode_repeated = len(signatures) - len(set(signatures))
        episode_repeated_non_observe = len(non_observe_signatures) - len(
            set(non_observe_signatures)
        )
        episode_consecutive = sum(
            current == previous for previous, current in zip(signatures, signatures[1:])
        )
        repeated_action_calls += episode_repeated
        repeated_non_observe_action_calls += episode_repeated_non_observe
        consecutive_duplicate_calls += episode_consecutive
        episode_diagnostics.append(
            {
                "activity": row.get("activity"),
                "actions": int(row.get("n_steps", 0)),
                "coverage": float(row.get("coverage", 0.0)),
                "stop_reason": row.get("stop_reason"),
                "refused_calls": episode_refused,
                "repeated_action_calls": episode_repeated,
                "repeated_non_observe_action_calls": episode_repeated_non_observe,
                "consecutive_duplicate_calls": episode_consecutive,
                "tool_counts": dict(sorted(episode_tool_counts.items())),
            }
        )

    return {
        "episodes": len(records),
        "activities": len({row.get("activity") for row in records}),
        "model_turns": model_turns,
        "action_turns": actions,
        "tool_call_rate": actions / model_turns if model_turns else 0.0,
        "nudges": sum(int(row.get("nudge_count", 0)) for row in records),
        "proper_completions": sum(
            bool(row.get("proper_completion")) for row in records
        ),
        "goal_satisfied": sum(bool(row.get("success")) for row in records),
        "mean_goal_coverage": sum(float(row.get("coverage", 0.0)) for row in records)
        / max(len(records), 1),
        "end_tasks": sum(bool(row.get("ended")) for row in records),
        "unsuccessful_end_tasks": sum(
            bool(row.get("ended")) and not bool(row.get("success")) for row in records
        ),
        "harness_failures": sum(
            str(row.get("stop_reason", "")).startswith("harness_error:")
            for row in records
        ),
        "parse_failures": sum(
            row.get("stop_reason") == "parse_error" for row in records
        ),
        "refused_calls": refused_calls,
        "tool_counts": dict(sorted(tool_counts.items())),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "rejection_tools": dict(rejection_tools.most_common()),
        "repeated_action_calls": repeated_action_calls,
        "repeated_non_observe_action_calls": repeated_non_observe_action_calls,
        "consecutive_duplicate_calls": consecutive_duplicate_calls,
        "episode_diagnostics": episode_diagnostics,
        "interface_violations": dict(sorted(interface_failures.items())),
        "interface_violation_count": sum(interface_failures.values()),
        "missing_provenance": dict(sorted(missing_provenance.items())),
        "missing_provenance_count": sum(missing_provenance.values()),
        "stop_reasons": dict(
            sorted(Counter(str(row.get("stop_reason", "")) for row in records).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts")
    parser.add_argument("--min-tool-call-rate", type=float, default=0.99)
    parser.add_argument("--require-zero-interface-violations", action="store_true")
    parser.add_argument("--require-provenance", action="store_true")
    args = parser.parse_args()
    with open(args.rollouts) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    summary = audit_records(records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failures = []
    if summary["tool_call_rate"] < args.min_tool_call_rate:
        failures.append(
            f"tool_call_rate={summary['tool_call_rate']:.6f} "
            f"< {args.min_tool_call_rate:.6f}"
        )
    if args.require_zero_interface_violations and summary["interface_violation_count"]:
        failures.append(
            f"interface_violation_count={summary['interface_violation_count']}"
        )
    if args.require_provenance and summary["missing_provenance_count"]:
        failures.append(
            f"missing_provenance_count={summary['missing_provenance_count']}"
        )
    if failures:
        raise SystemExit("rollout audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
