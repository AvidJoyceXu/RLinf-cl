"""Integration regression for crash-safe standalone-HF rollout resume."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlinf.envs.behavior import textworld_api_rollout as api_rollout
from rlinf.envs.behavior import textworld_rollout as hf_rollout


def _record(spec, budget, provenance):
    return {
        **provenance,
        "activity": spec["activity"],
        "instance_source": spec.get("instance_source"),
        "geometry_instance_key": spec.get("geometry_instance_key"),
        "obs_mode": "detect_scope",
        "goal_format": "nl",
        "selection_mode": "handle",
        **budget.record(),
        "n_steps": 1,
        "model_turns": 1,
        "tool_call_rate": 1.0,
        "nudge_count": 0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "seconds": 0.1,
        "context_compactions": 0,
        "ended": True,
        "stop_reason": "end_task",
        "error": "",
        "num_satisfied": 1,
        "num_goal": 1,
        "coverage": 1.0,
        "success": True,
        "proper_completion": True,
        "tools": ["end_task"],
        "tool_trace": [],
        "rejections": [],
    }


class StandaloneHFRolloutResumeTest(unittest.TestCase):
    def test_generation_stops_on_complete_tool_call_close_tokens(self):
        import torch

        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [7, 8] if text == "</tool_call>" else []

        criterion = hf_rollout._tool_call_stopping_criteria(Tokenizer())[0]
        self.assertFalse(criterion(torch.tensor([[1, 7]]), None))
        self.assertTrue(criterion(torch.tensor([[1, 7, 8]]), None))

    def test_interrupt_then_resume_skips_committed_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activities = root / "activities.jsonl"
            activities.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": activity,
                            "solutions": json.dumps(
                                {
                                    "activity": activity,
                                    "oracle_turns": 1,
                                    "instance_source": "source",
                                    "geometry_instance_key": f"{activity}.json",
                                }
                            ),
                        }
                    )
                    + "\n"
                    for activity in ("a", "b")
                )
            )
            model = root / "model"
            model.mkdir()
            output = root / "k2.jsonl"
            argv = [
                "textworld_rollout",
                "--model",
                str(model),
                "--activities",
                f"@{activities}",
                "--budget-k",
                "2",
                "--replicate-id",
                "rep0",
                "--rlinf-revision",
                "nested",
                "--outer-revision",
                "outer",
                "--container-image",
                "image",
                "--resume",
                "--out",
                str(output),
            ]

            first_calls = []

            def interrupted(model_arg, tokenizer, spec, args, budget, provenance):
                del model_arg, tokenizer, args
                first_calls.append(spec["activity"])
                if spec["activity"] == "b":
                    raise KeyboardInterrupt
                return _record(spec, budget, provenance)

            with (
                patch.object(sys, "argv", argv),
                patch.object(hf_rollout, "validate_detect_runtime"),
                patch.object(
                    hf_rollout, "load_model", return_value=(object(), object())
                ),
                patch.object(hf_rollout, "run_episode", side_effect=interrupted),
                self.assertRaises(KeyboardInterrupt),
            ):
                hf_rollout.main()

            self.assertEqual(first_calls, ["a", "b"])
            self.assertFalse(output.exists())
            self.assertTrue(Path(f"{output}.partial/000000.json").is_file())
            self.assertFalse(Path(f"{output}.partial/000001.json").exists())

            resumed_calls = []

            def resumed(model_arg, tokenizer, spec, args, budget, provenance):
                del model_arg, tokenizer, args
                resumed_calls.append(spec["activity"])
                return _record(spec, budget, provenance)

            with (
                patch.object(sys, "argv", argv),
                patch.object(hf_rollout, "validate_detect_runtime"),
                patch.object(
                    hf_rollout, "load_model", return_value=(object(), object())
                ),
                patch.object(hf_rollout, "run_episode", side_effect=resumed),
            ):
                hf_rollout.main()

            self.assertEqual(resumed_calls, ["b"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["activity"] for row in rows], ["a", "b"])
            self.assertEqual([row["episode_index"] for row in rows], [0, 1])
            self.assertTrue((root / "k2.summary.json").is_file())


def _api_record(activity, obs_mode, goal_format, budget_meta, provenance):
    return {
        **provenance,
        "activity": activity,
        "instance_source": "source",
        "geometry_instance_key": f"{activity}.json",
        "obs_mode": obs_mode,
        "goal_format": goal_format,
        "selection_mode": "handle",
        **budget_meta,
        "n_steps": 1,
        "model_turns": 1,
        "tool_call_rate": 1.0,
        "nudge_count": 0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "api_calls": 1,
        "api_errors": 0,
        "seconds": 0.1,
        "context_compactions": 0,
        "ended": True,
        "stop_reason": "end_task",
        "api_error": "",
        "coverage": 1.0,
        "success": True,
        "proper_completion": True,
        "tools": ["end_task"],
        "tool_trace": [],
        "rejections": [],
    }


class APIRolloutResumeTest(unittest.TestCase):
    def test_interrupt_then_resume_skips_api_episode_and_rebuilds_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activities = root / "activities.jsonl"
            activities.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": activity,
                            "solutions": json.dumps(
                                {
                                    "activity": activity,
                                    "oracle_turns": 1,
                                    "instance_source": "source",
                                    "geometry_instance_key": f"{activity}.json",
                                }
                            ),
                        }
                    )
                    + "\n"
                    for activity in ("a", "b", "c")
                )
            )
            output = root / "k2.jsonl"
            argv = [
                "textworld_api_rollout",
                "--model",
                "provider-model",
                "--activities",
                f"@{activities}",
                "--budget-k",
                "2",
                "--replicate-id",
                "rep0",
                "--rlinf-revision",
                "nested",
                "--outer-revision",
                "outer",
                "--container-image",
                "image",
                "--context-tokenizer",
                "tokenizer",
                "--concurrency",
                "1",
                "--resume",
                "--out",
                str(output),
            ]
            first_calls = []

            def interrupted(*args, **kwargs):
                del kwargs
                activity, obs_mode, goal_format = args[2:5]
                first_calls.append(activity)
                if activity == "b":
                    raise KeyboardInterrupt
                return _api_record(activity, obs_mode, goal_format, args[12], args[18])

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"ROBOPARTY_API_KEY": "test-key"}),
                patch("openai.OpenAI", return_value=object()),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=object(),
                ),
                patch.object(api_rollout, "validate_detect_runtime"),
                patch.object(api_rollout, "run_episode", side_effect=interrupted),
                self.assertRaises(KeyboardInterrupt),
            ):
                api_rollout.main()

            self.assertEqual(first_calls, ["a", "b", "c"])
            self.assertFalse(output.exists())
            self.assertTrue(Path(f"{output}.partial/000000.json").is_file())

            resumed_calls = []

            def resumed(*args, **kwargs):
                del kwargs
                activity, obs_mode, goal_format = args[2:5]
                resumed_calls.append(activity)
                return _api_record(activity, obs_mode, goal_format, args[12], args[18])

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"ROBOPARTY_API_KEY": "test-key"}),
                patch("openai.OpenAI", return_value=object()),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=object(),
                ),
                patch.object(api_rollout, "validate_detect_runtime"),
                patch.object(api_rollout, "run_episode", side_effect=resumed),
            ):
                api_rollout.main()

            self.assertEqual(resumed_calls, ["b", "c"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["activity"] for row in rows], ["a", "b", "c"])
            summary = json.loads((root / "k2_summary.json").read_text())
            self.assertEqual(summary["summaries"][0]["prompt_tokens"], 30)
            self.assertEqual(summary["summaries"][0]["api_calls"], 3)


if __name__ == "__main__":
    unittest.main()
