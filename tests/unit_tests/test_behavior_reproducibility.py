"""Sim-free regression gates for the BEHAVIOR tool-call benchmark."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_pure_module(name: str, filename: str):
    """Load a sim-free module without importing ``rlinf`` (which imports torch)."""
    root = Path(__file__).resolve().parents[2]
    path = root / "rlinf" / "envs" / "behavior" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sft_build = _load_pure_module("_behavior_sft_build_test", "sft_build.py")
_turn_budget = _load_pure_module("_behavior_turn_budget_test", "turn_budget.py")
CAMERA_OBS_MODES = _sft_build.CAMERA_OBS_MODES
all_schemas = _sft_build.all_schemas
tool_schemas = _sft_build.tool_schemas
turn_budget_for = _turn_budget.turn_budget_for


class ToolSchemaTest(unittest.TestCase):
    def test_non_camera_modes_keep_the_core_action_space(self):
        core_names = [tool["name"] for tool in tool_schemas()]
        self.assertEqual(len(core_names), 17)
        self.assertEqual(len(core_names), len(set(core_names)))
        for mode in ("full", "partial", "object"):
            self.assertEqual(
                [tool["name"] for tool in all_schemas(mode)],
                core_names,
                msg=mode,
            )

    def test_camera_modes_add_the_same_eight_unique_actions(self):
        core_names = [tool["name"] for tool in tool_schemas()]
        expected_camera = {
            "turn_left",
            "turn_right",
            "move_ahead",
            "move_back",
            "strafe_left",
            "strafe_right",
            "look_down",
            "look_up",
        }
        for mode in CAMERA_OBS_MODES:
            names = [tool["name"] for tool in all_schemas(mode)]
            self.assertEqual(len(names), 25, msg=mode)
            self.assertEqual(len(names), len(set(names)), msg=mode)
            self.assertEqual(set(names) - set(core_names), expected_camera, msg=mode)


class TurnBudgetTest(unittest.TestCase):
    def test_flat_budget_is_fully_recorded(self):
        budget = turn_budget_for("task_a", flat_max_turns=40)
        self.assertEqual(budget.max_turns, 40)
        self.assertEqual(budget.budget_source, "flat")
        self.assertFalse(budget.capped)
        self.assertEqual(budget.record()["requested_max_turns"], 40)

    def test_oracle_relative_budget_and_cap_are_recorded(self):
        uncapped = turn_budget_for(
            "task_a",
            flat_max_turns=40,
            oracle_steps={"task_a": 56},
            budget_k=2,
            budget_cap=250,
        )
        self.assertEqual(uncapped.max_turns, 112)
        self.assertFalse(uncapped.capped)

        capped = turn_budget_for(
            "task_a",
            flat_max_turns=40,
            oracle_steps={"task_a": 200},
            budget_k=2,
            budget_cap=250,
        )
        self.assertEqual(capped.max_turns, 250)
        self.assertEqual(capped.requested_max_turns, 400)
        self.assertTrue(capped.capped)

    def test_missing_oracle_fails_closed_unless_explicit(self):
        with self.assertRaises(KeyError):
            turn_budget_for(
                "task_b",
                flat_max_turns=40,
                oracle_steps={"task_a": 56},
            )

        fallback = turn_budget_for(
            "task_b",
            flat_max_turns=40,
            oracle_steps={"task_a": 56},
            allow_missing_oracle=True,
        )
        self.assertEqual(fallback.max_turns, 40)
        self.assertEqual(fallback.budget_source, "flat_fallback")

    def test_invalid_budget_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            turn_budget_for("task_a", flat_max_turns=0)
        with self.assertRaises(ValueError):
            turn_budget_for("task_a", flat_max_turns=40, budget_k=0)
        with self.assertRaises(ValueError):
            turn_budget_for("task_a", flat_max_turns=40, budget_cap=-1)


if __name__ == "__main__":
    unittest.main()
