"""Paired-bootstrap comparison for matched BEHAVIOR text rollout JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict

MATCH_FIELDS = (
    "activities_sha256",
    "obs_mode",
    "goal_format",
    "selection_mode",
    "prompt_mode",
    "reason_mode",
    "decoding",
)


def load_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path) as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def arm_activity_means(
    records: list[dict], *, min_replicates: int, budget_k: float
) -> dict[str, float]:
    if not records:
        raise ValueError("rollout arm is empty")
    by_activity: dict[str, list[bool]] = defaultdict(list)
    seen = set()
    for row in records:
        activity = row.get("activity")
        replicate = row.get("replicate_id")
        if not activity or replicate is None:
            raise ValueError("every row must carry activity and replicate_id")
        key = (str(activity), str(replicate))
        if key in seen:
            raise ValueError(f"duplicate activity/replicate row: {key}")
        seen.add(key)
        if float(row.get("budget_k", -1)) != budget_k:
            raise ValueError(
                f"{activity}/{replicate} has budget_k={row.get('budget_k')}, "
                f"expected {budget_k}"
            )
        by_activity[str(activity)].append(bool(row.get("proper_completion")))
    short = {
        activity: len(values)
        for activity, values in by_activity.items()
        if len(values) < min_replicates
    }
    if short:
        raise ValueError(f"insufficient replicates: {short}")
    return {
        activity: sum(values) / len(values) for activity, values in by_activity.items()
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_declared_denominator(records: list[dict]) -> None:
    paths = {row.get("activities_path") for row in records}
    digests = {row.get("activities_sha256") for row in records}
    if len(paths) != 1 or None in paths or len(digests) != 1 or None in digests:
        raise ValueError("arm must carry one activities_path and activities_sha256")
    path = str(next(iter(paths)))
    expected_digest = str(next(iter(digests)))
    actual_digest = _sha256(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"activity input hash drift: recorded={expected_digest}, actual={actual_digest}"
        )
    expected = set()
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            spec = json.loads(row["solutions"]) if "solutions" in row else row
            expected.add(str(spec["activity"]))
    observed = {str(row.get("activity")) for row in records if row.get("activity")}
    if observed != expected:
        raise ValueError(
            "rollout denominator does not equal declared activity input: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def validate_matched_contract(candidate: list[dict], baseline: list[dict]) -> None:
    validate_declared_denominator(candidate)
    validate_declared_denominator(baseline)
    for field in MATCH_FIELDS:
        candidate_values = {
            json.dumps(row.get(field), sort_keys=True) for row in candidate
        }
        baseline_values = {
            json.dumps(row.get(field), sort_keys=True) for row in baseline
        }
        if len(candidate_values) != 1 or len(baseline_values) != 1:
            raise ValueError(f"{field} varies within an arm")
        if candidate_values != baseline_values:
            raise ValueError(
                f"unmatched {field}: candidate={candidate_values}, "
                f"baseline={baseline_values}"
            )
    candidate_budgets = _activity_budgets(candidate)
    baseline_budgets = _activity_budgets(baseline)
    if candidate_budgets != baseline_budgets:
        raise ValueError("candidate and baseline episode budgets do not match exactly")


def _activity_budgets(records: list[dict]) -> dict[str, tuple]:
    fields = (
        "budget_source",
        "oracle_steps",
        "budget_k",
        "budget_cap",
        "requested_max_turns",
        "max_turns",
        "capped",
    )
    budgets: dict[str, tuple] = {}
    for row in records:
        activity = str(row["activity"])
        budget = tuple(row.get(field) for field in fields)
        previous = budgets.setdefault(activity, budget)
        if previous != budget:
            raise ValueError(f"budget varies across {activity} replicates")
    return budgets


def paired_bootstrap(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict:
    if candidate.keys() != baseline.keys():
        only_candidate = sorted(candidate.keys() - baseline.keys())
        only_baseline = sorted(baseline.keys() - candidate.keys())
        raise ValueError(
            "activity denominators differ: "
            f"candidate_only={only_candidate}, baseline_only={only_baseline}"
        )
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    activities = sorted(candidate)
    if not activities:
        raise ValueError("no paired activities")
    differences = [candidate[a] - baseline[a] for a in activities]
    point = sum(differences) / len(differences)
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(differences) for _ in activities) / len(activities)
        for _ in range(samples)
    )
    lower = boot[int(0.025 * (samples - 1))]
    upper = boot[int(0.975 * (samples - 1))]
    return {
        "activities": len(activities),
        "candidate_rate": sum(candidate.values()) / len(activities),
        "baseline_rate": sum(baseline.values()) / len(activities),
        "paired_difference": point,
        "ci95": [lower, upper],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "criterion_passed": lower > 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--min-replicates", type=int, default=3)
    parser.add_argument("--budget-k", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out")
    parser.add_argument("--require-positive-lower-bound", action="store_true")
    args = parser.parse_args()

    candidate_records = load_records(args.candidate)
    baseline_records = load_records(args.baseline)
    validate_matched_contract(candidate_records, baseline_records)
    candidate = arm_activity_means(
        candidate_records,
        min_replicates=args.min_replicates,
        budget_k=args.budget_k,
    )
    baseline = arm_activity_means(
        baseline_records,
        min_replicates=args.min_replicates,
        budget_k=args.budget_k,
    )
    result = paired_bootstrap(
        candidate,
        baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(payload + "\n")
    if args.require_positive_lower_bound and not result["criterion_passed"]:
        raise SystemExit("paired-bootstrap lower bound is not above zero")


if __name__ == "__main__":
    main()
