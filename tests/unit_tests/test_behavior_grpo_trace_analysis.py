"""Simulator-free tests for persisted BEHAVIOR GRPO trace analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlinf.algorithms.rewards.behavior import reward_breakdown
from rlinf.envs.behavior.analyze_grpo_traces import load_trace_streams, summarize


def _row(*, coverage: float, end: bool, success: bool = False) -> dict:
    name = "end_task" if end else "observe"
    ok = success if end else True
    trace = [
        {
            "name": name,
            "arguments": {},
            "result": {"ok": ok},
            "meta": {
                "initial_atom_coverage": 0.0,
                "atom_coverage": coverage,
                "is_success": success,
            },
        }
    ]
    return {
        "episode_id": f"episode-{coverage}-{end}",
        "activity": "activity",
        "stop_reason": "end_task" if end else "max_turns",
        "success": success,
        "proper_completion": success and end,
        "coverage": coverage,
        "initial_atom_coverage": 0.0,
        "num_turns": 1,
        "model_turns": 1,
        "tool_trace": trace,
        "reward": reward_breakdown(trace)["total"],
    }


class GrpoTraceAnalysisTest(unittest.TestCase):
    def test_maps_stream_rows_to_steps_and_replays_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [
                _row(coverage=0.0, end=True),
                _row(coverage=0.5, end=False),
                _row(coverage=1.0, end=True, success=True),
                _row(coverage=0.0, end=False),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            episodes, groups = load_trace_streams([f"25:{path}"], group_size=2)
            summary = summarize(episodes, groups, window_size=2)

        self.assertEqual([row["step"] for row in episodes], [25, 25, 26, 26])
        self.assertEqual(len(groups), 2)
        self.assertEqual(summary["step_range"], [25, 26])
        self.assertEqual(summary["overall"]["proper_completion_rate"], 0.25)
        self.assertEqual(episodes[0]["primary_outcome"], "immediate_give_up")
        self.assertEqual(episodes[1]["primary_outcome"], "budget_exhaustion")
        self.assertEqual(episodes[0]["reward"], -0.02)
        self.assertEqual(episodes[0]["fixed_reward_breakdown"]["total"], -0.52)
        self.assertGreater(
            episodes[1]["fixed_reward_breakdown"]["total"],
            episodes[0]["fixed_reward_breakdown"]["total"],
        )

    def test_rejects_reward_replay_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            row = _row(coverage=0.0, end=True)
            row["reward"] = 123.0
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "reward replay mismatch"):
                load_trace_streams([f"1:{path}"], group_size=2)


if __name__ == "__main__":
    unittest.main()
