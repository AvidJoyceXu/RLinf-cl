"""Replay and classify persisted BEHAVIOR GRPO training traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rlinf.algorithms.rewards.behavior import reward_breakdown


def _signature(turn: dict[str, Any]) -> str:
    return json.dumps(
        [turn.get("name"), turn.get("arguments") or {}],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _classify(row: dict[str, Any], breakdown: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("tool_trace") or []
    signatures = [_signature(turn) for turn in trace]
    last_tool = str(trace[-1].get("name")) if trace else "EMPTY"
    premature_end = last_tool == "end_task" and not bool(row.get("success"))
    partial_progress = float(breakdown["coverage_delta"]) > 0.0
    repeated_calls = len(signatures) - len(set(signatures))
    consecutive_duplicates = sum(
        left_signature == right_signature and left_turn.get("name") != "scan_next"
        for left_turn, left_signature, right_signature in zip(
            trace, signatures, signatures[1:]
        )
    )
    if row.get("proper_completion"):
        primary = "proper_success"
    elif premature_end and int(row.get("num_turns", 0)) <= 2:
        primary = "immediate_give_up"
    elif premature_end:
        primary = "premature_end"
    elif int(breakdown["format_errors"]) > 0:
        primary = "format_failure"
    elif row.get("stop_reason") == "max_turns":
        primary = "budget_exhaustion"
    elif partial_progress:
        primary = "partial_progress"
    else:
        primary = "zero_progress_failure"
    return {
        "primary_outcome": primary,
        "last_tool": last_tool,
        "premature_end": premature_end,
        "immediate_give_up": premature_end and int(row.get("num_turns", 0)) <= 2,
        "partial_progress": partial_progress,
        "zero_progress_premature_end": premature_end and not partial_progress,
        "repeated_action_calls": repeated_calls,
        "consecutive_duplicate_calls": consecutive_duplicates,
        "possible_tool_loop": consecutive_duplicates >= 2,
        "high_reward_failure": not bool(row.get("success"))
        and float(breakdown["total"]) >= 0.0,
    }


def load_trace_streams(
    stream_specs: list[str], *, group_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ``START_STEP:PATH`` streams and infer one group per step per file."""
    if group_size <= 1:
        raise ValueError("group_size must be greater than one")
    episodes = []
    groups = []
    for stream_index, spec in enumerate(stream_specs):
        stream_episode_start = len(episodes)
        try:
            start_text, path_text = spec.split(":", 1)
            start_step = int(start_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"invalid trace stream {spec!r}; expected START_STEP:PATH"
            ) from exc
        path = Path(path_text).resolve()
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or len(rows) % group_size:
            raise ValueError(
                f"{path} has {len(rows)} rows, not a positive multiple of "
                f"group_size={group_size}"
            )
        stream_name = f"stream{stream_index}:{path.name}"
        for line_index, row in enumerate(rows):
            step = start_step + line_index // group_size
            member = line_index % group_size
            breakdown = reward_breakdown(row.get("tool_trace") or [])
            fixed_breakdown = reward_breakdown(
                row.get("tool_trace") or [], premature_end_penalty=-0.5
            )
            logged_reward = float(row["reward"])
            if not math.isclose(
                logged_reward, float(breakdown["total"]), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"reward replay mismatch in {path}:{line_index + 1}: "
                    f"logged={logged_reward}, replayed={breakdown['total']}"
                )
            classification = _classify(row, breakdown)
            episodes.append(
                {
                    "step": step,
                    "group_id": f"{stream_name}:step{step}",
                    "group_member": member,
                    "source_path": str(path),
                    "source_line": line_index + 1,
                    "episode_id": row.get("episode_id"),
                    "activity": row.get("activity"),
                    "stop_reason": row.get("stop_reason"),
                    "success": bool(row.get("success")),
                    "proper_completion": bool(row.get("proper_completion")),
                    "coverage": float(row.get("coverage", 0.0)),
                    "initial_coverage": float(row.get("initial_atom_coverage", 0.0)),
                    "num_turns": int(row.get("num_turns", 0)),
                    "model_turns": int(row.get("model_turns", 0)),
                    "reward": logged_reward,
                    "reward_breakdown": breakdown,
                    "fixed_reward_breakdown": fixed_breakdown,
                    "tool_counts": dict(
                        sorted(
                            Counter(
                                str(turn.get("name"))
                                for turn in (row.get("tool_trace") or [])
                            ).items()
                        )
                    ),
                    **classification,
                }
            )
        for offset in range(0, len(rows), group_size):
            group_start = stream_episode_start + offset
            group_rows = episodes[group_start : group_start + group_size]
            rewards = [float(row["reward"]) for row in group_rows]
            fixed_rewards = [
                float(row["fixed_reward_breakdown"]["total"]) for row in group_rows
            ]
            step = start_step + offset // group_size
            groups.append(
                {
                    "step": step,
                    "group_id": f"{stream_name}:step{step}",
                    "activity": group_rows[0]["activity"],
                    "rewards": rewards,
                    "reward_mean": statistics.fmean(rewards),
                    "reward_std": statistics.stdev(rewards),
                    "zero_reward_variance": max(rewards) == min(rewards),
                    "fixed_rewards": fixed_rewards,
                    "fixed_reward_mean": statistics.fmean(fixed_rewards),
                    "fixed_reward_std": statistics.stdev(fixed_rewards),
                    "fixed_zero_reward_variance": max(fixed_rewards)
                    == min(fixed_rewards),
                    "successes": sum(row["success"] for row in group_rows),
                    "premature_ends": sum(row["premature_end"] for row in group_rows),
                }
            )
    return episodes, groups


def summarize(
    episodes: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    window_size: int,
) -> dict[str, Any]:
    """Return overall, per-window, and per-step training diagnostics."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "episodes": count,
            "success_rate": sum(row["success"] for row in rows) / count,
            "proper_completion_rate": sum(row["proper_completion"] for row in rows)
            / count,
            "premature_end_rate": sum(row["premature_end"] for row in rows) / count,
            "immediate_give_up_rate": sum(row["immediate_give_up"] for row in rows)
            / count,
            "zero_progress_premature_end_rate": sum(
                row["zero_progress_premature_end"] for row in rows
            )
            / count,
            "partial_progress_rate": sum(row["partial_progress"] for row in rows)
            / count,
            "mean_coverage": statistics.fmean(row["coverage"] for row in rows),
            "mean_coverage_delta": statistics.fmean(
                row["reward_breakdown"]["coverage_delta"] for row in rows
            ),
            "mean_turns": statistics.fmean(row["num_turns"] for row in rows),
            "mean_reward": statistics.fmean(row["reward"] for row in rows),
            "mean_fixed_reward": statistics.fmean(
                row["fixed_reward_breakdown"]["total"] for row in rows
            ),
            "mean_format_errors": statistics.fmean(
                row["reward_breakdown"]["format_errors"] for row in rows
            ),
            "mean_refused_calls": statistics.fmean(
                row["reward_breakdown"]["refused_calls"] for row in rows
            ),
            "outcomes": dict(
                sorted(Counter(row["primary_outcome"] for row in rows).items())
            ),
            "stop_reasons": dict(
                sorted(Counter(str(row["stop_reason"]) for row in rows).items())
            ),
        }

    first_step = min(row["step"] for row in episodes)
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    groups_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_window[(row["step"] - first_step) // window_size].append(row)
        by_step[row["step"]].append(row)
    for group in groups:
        groups_by_window[(group["step"] - first_step) // window_size].append(group)

    def aggregate_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
        zero_variance = sum(row["zero_reward_variance"] for row in rows)
        fixed_zero_variance = sum(row["fixed_zero_reward_variance"] for row in rows)
        return {
            "count": len(rows),
            "zero_reward_variance": zero_variance,
            "zero_reward_variance_rate": zero_variance / len(rows),
            "mean_reward_std": statistics.fmean(row["reward_std"] for row in rows),
            "fixed_zero_reward_variance": fixed_zero_variance,
            "fixed_zero_reward_variance_rate": fixed_zero_variance / len(rows),
            "mean_fixed_reward_std": statistics.fmean(
                row["fixed_reward_std"] for row in rows
            ),
        }

    return {
        "schema": "behavior-grpo-trace-analysis-v1",
        "step_range": [first_step, max(row["step"] for row in episodes)],
        "overall": aggregate(episodes),
        "windows": [
            {
                "step_range": [
                    min(row["step"] for row in rows),
                    max(row["step"] for row in rows),
                ],
                **aggregate(rows),
            }
            for _, rows in sorted(by_window.items())
        ],
        "steps": [
            {"step": step, **aggregate(rows)} for step, rows in sorted(by_step.items())
        ],
        "groups": {
            **aggregate_groups(groups),
            "windows": [
                {
                    "step_range": [
                        min(row["step"] for row in rows),
                        max(row["step"] for row in rows),
                    ],
                    **aggregate_groups(rows),
                }
                for _, rows in sorted(groups_by_window.items())
            ],
        },
    }


def write_analysis(
    out_dir: Path,
    episodes: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write the compact, machine-readable analysis without copying raw traces."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "episodes.jsonl": "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in episodes
        )
        + "\n",
        "groups.jsonl": "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in groups
        )
        + "\n",
        "summary.json": json.dumps(
            summary, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    }
    for name, payload in outputs.items():
        path = out_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis artifact: {path}")
        path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-stream",
        action="append",
        required=True,
        help="START_STEP:PATH; one file is one GRPO group stream",
    )
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    episodes, groups = load_trace_streams(args.trace_stream, group_size=args.group_size)
    summary = summarize(episodes, groups, window_size=args.window_size)
    write_analysis(args.out_dir, episodes, groups, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
