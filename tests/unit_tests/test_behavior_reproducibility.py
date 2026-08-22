"""Sim-free regression gates for the BEHAVIOR tool-call benchmark."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
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
_certificate = _load_pure_module(
    "_behavior_solvability_certificate_test", "solvability_certificate.py"
)
_compatibility = _load_pure_module(
    "_behavior_instance_compatibility_test", "instance_compatibility.py"
)
_sources = _load_pure_module("_behavior_instance_sources_test", "instance_sources.py")
_capabilities = _load_pure_module(
    "_behavior_harness_capabilities_test", "harness_capabilities.py"
)
_trajectory_video = _load_pure_module(
    "_behavior_trajectory_video_test", "trajectory_video.py"
)
CAMERA_OBS_MODES = _sft_build.CAMERA_OBS_MODES
all_schemas = _sft_build.all_schemas
tool_schemas = _sft_build.tool_schemas
turn_budget_for = _turn_budget.turn_budget_for
build_activity_certificate = _certificate.build_activity_certificate
classify_object = _certificate.classify_object
instance_scope = _compatibility.instance_scope
scope_mismatch = _compatibility.scope_mismatch
expand_problem_wildcards = _compatibility.expand_problem_wildcards
assert_detect_eligible = _sources.assert_detect_eligible
resolve_activity_location = _sources.resolve_activity_location
selected_layout_sources = _sources.selected_layout_sources
source_catalog = _sources.source_catalog
validate_harness = _capabilities.validate_harness
TrajectoryVideoRecorder = _trajectory_video.TrajectoryVideoRecorder


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


class InstanceSourcePolicyTest(unittest.TestCase):
    def test_default_layout_sources_exclude_local_samples(self):
        names = [source.name for source in selected_layout_sources(data_root="/tmp/x")]
        self.assertEqual(names, ["2026-v3.9.1", "2025-official"])
        self.assertNotIn("local-v3.7", names)

    def test_unknown_and_duplicate_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            selected_layout_sources("invented", data_root="/tmp/x")
        with self.assertRaises(ValueError):
            selected_layout_sources(
                "2025-official,2025-official", data_root="/tmp/x"
            )

    def test_local_samples_are_not_detect_eligible(self):
        catalog = source_catalog("/tmp/x")
        with self.assertRaisesRegex(ValueError, "paired .*template"):
            assert_detect_eligible(catalog["local-v3.7"])
        assert_detect_eligible(catalog["2025-official"])
        assert_detect_eligible(catalog["2026-v3.9.1"])

    def test_simulator_resolution_never_crosses_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = "house_single_floor"
            rel = Path("2025-challenge-task-instances/scenes") / scene / "json"
            activity_dir = rel / f"{scene}_task_task_a_instances"
            (root / activity_dir).mkdir(parents=True)
            location = resolve_activity_location(
                "task_a", source_name="2025-official", data_root=str(root)
            )
            self.assertEqual(location.source.name, "2025-official")
            self.assertEqual(Path(location.directory), root / activity_dir)
            with self.assertRaisesRegex(ValueError, "source='2026-v3.9.1'"):
                resolve_activity_location(
                    "task_a", source_name="2026-v3.9.1", data_root=str(root)
                )

    def test_scene_wildcard_is_removed_when_minimum_already_fills_room(self):
        problem = """(:objects
  sink.n.01_1 sink.n.01_* - sink.n.01
)
(:init
  (inroom sink.n.01_* bathroom)
)
"""
        expanded = expand_problem_wildcards(problem, {"sink.n.01_1"})
        self.assertNotIn("*", expanded)
        self.assertNotIn("(inroom sink.n.01_", expanded)
        self.assertIn("sink.n.01_1", expanded)

    def test_scene_wildcard_expands_extra_template_bindings(self):
        problem = """(:objects
  cabinet.n.01_1 cabinet.n.01_* - cabinet.n.01
)
(:init
  (inroom cabinet.n.01_* kitchen)
)
"""
        expanded = expand_problem_wildcards(
            problem,
            {"cabinet.n.01_1", "cabinet.n.01_2", "cabinet.n.01_3"},
        )
        self.assertNotIn("*", expanded)
        self.assertIn("cabinet.n.01_2 cabinet.n.01_3 - cabinet.n.01", expanded)
        self.assertIn("(inroom cabinet.n.01_2 kitchen)", expanded)
        self.assertIn("(inroom cabinet.n.01_3 kitchen)", expanded)


class HarnessCapabilityTest(unittest.TestCase):
    def test_omnigibson_fov_schema_has_dispatch_methods(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse(
            (root / "rlinf/envs/behavior/semantic_tools.py").read_text()
        )
        semantic_aci = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SemanticACI"
        )
        methods = {
            node.name
            for node in semantic_aci.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        schema_names = {tool["name"] for tool in all_schemas("fov")}
        self.assertFalse(schema_names - methods, msg=sorted(schema_names - methods))

    def test_schema_only_omnigibson_modes_fail_before_launch(self):
        validate_harness("omnigibson", "fov")
        for mode in ("detect", "detect_scope", "fov_distract", "object"):
            with self.assertRaisesRegex(ValueError, "Schema registration alone"):
                validate_harness("omnigibson", mode)

    def test_policy_rgb_transport_is_not_claimed(self):
        with self.assertRaisesRegex(ValueError, "tokenizes JSON tool text only"):
            validate_harness("omnigibson", "fov", policy_rgb=True)

    def test_textworld_keeps_all_symbolic_observation_modes(self):
        for mode in (
            "full",
            "partial",
            "object",
            "fov",
            "fov_distract",
            "detect",
            "detect_scope",
        ):
            validate_harness("textworld", mode)


class TrajectoryVideoTest(unittest.TestCase):
    def test_one_video_and_aligned_sidecar_per_complete_session(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryVideoRecorder(directory, fps=2)
            video = recorder.begin(
                {"activity": "task_a", "instance_id": 7, "session_id": "abc"}
            )
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            recorder.append(frame, tool="session_start", ok=True)
            recorder.append(frame, tool="turn_left", ok=True)
            record = recorder.finish(complete=True, reason="session_end")
            self.assertTrue(Path(video).is_file())
            self.assertGreater(Path(video).stat().st_size, 0)
            sidecars = list(Path(directory).glob("*.json"))
            videos = list(Path(directory).glob("*.mp4"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(len(videos), 1)
            self.assertTrue(record["complete"])
            self.assertEqual(record["frames"], 2)
            self.assertEqual(
                [event["tool"] for event in record["events"]],
                ["session_start", "turn_left"],
            )


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


class SolvabilityCertificateTest(unittest.TestCase):
    def test_miss_taxonomy_is_mutually_exclusive(self):
        cases = {
            "nonvisual_substance": ([], ["nonvisual_substance"]),
            "missing_pose": ([], ["missing_pose"]),
            "missing_extent": ([], ["missing_extent"]),
            "found_by_primitive": (["visible"], ["visible"]),
            "primitive_search_miss": ([], ["visible"]),
            "outside_tour_range": ([], ["target"]),
            "never_projectable": ([], ["within_range"]),
            "near_total_occlusion": ([], ["within_range", "projectable", "occluded"]),
            "tour_miss_unclassified": ([], ["within_range", "projectable"]),
        }
        for expected, (primitive, tour) in cases.items():
            self.assertEqual(classify_object(primitive, tour), expected)

    def test_activity_certificate_conserves_target_denominator(self):
        primitive = {
            "target_objects": ["a", "b"],
            "object_events": {"a": ["visible"], "b": ["target"]},
            "source": "instance.json",
            "instance_path": "/data/instance.json",
            "steps": 40,
            "complete": False,
        }
        tour = {
            "target_objects": ["a", "b", "c"],
            "object_events": {
                "a": ["visible"],
                "b": ["visible"],
                "c": ["missing_pose"],
            },
            "source": "instance.json",
            "instance_path": "/data/instance.json",
            "steps": 400,
            "complete": False,
        }
        cert = build_activity_certificate("task", primitive, tour)
        self.assertEqual(cert["targets"], 3)
        self.assertEqual(cert["instance_path"], "/data/instance.json")
        self.assertEqual(sum(cert["outcome_counts"].values()), 3)
        self.assertEqual(cert["outcome_counts"]["found_by_primitive"], 1)
        self.assertEqual(cert["outcome_counts"]["primitive_search_miss"], 1)
        self.assertEqual(cert["outcome_counts"]["missing_pose"], 1)


class InstanceCompatibilityTest(unittest.TestCase):
    def _write_pair(self, root: Path, template_scope, tro_scope) -> Path:
        tro = root / "example_template-tro_state.json"
        template = root / "example_template.json"
        tro.write_text(json.dumps({name: {} for name in tro_scope}))
        template.write_text(
            json.dumps(
                {
                    "metadata": {
                        "task": {
                            "inst_to_name": {
                                name: f"scene_{name}" for name in template_scope
                            }
                        }
                    }
                }
            )
        )
        return tro

    def test_exact_template_scope_is_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = self._write_pair(Path(directory), ["agent", "plate"], ["plate"])
            self.assertEqual(instance_scope(str(tro)), {"agent", "plate"})
            self.assertIsNone(scope_mismatch(["agent", "plate"], str(tro)))

    def test_partial_overlap_reports_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = self._write_pair(
                Path(directory), ["agent", "taco", "countertop"], ["taco"]
            )
            self.assertEqual(
                scope_mismatch(["agent", "kabob", "breakfast_table"], str(tro)),
                {
                    "missing": ["breakfast_table", "kabob"],
                    "extra": ["countertop", "taco"],
                },
            )

    def test_scene_wildcards_match_numbered_template_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = self._write_pair(
                Path(directory), ["cabinet.n.01_2", "cabinet.n.01_3"], []
            )
            self.assertIsNone(scope_mismatch(["cabinet.n.01_*"], str(tro)))

    def test_future_objects_are_ignored_but_other_missing_objects_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = self._write_pair(Path(directory), ["onion.n.01_1"], [])
            expected = ["onion.n.01_1", "diced__onion.n.01_1", "knife.n.01_1"]
            self.assertEqual(
                scope_mismatch(
                    expected,
                    str(tro),
                    ignored_scope=["diced__onion.n.01_1"],
                ),
                {"missing": ["knife.n.01_1"], "extra": []},
            )

    def test_scene_objects_can_be_absent_from_task_instance_state(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = self._write_pair(
                Path(directory), ["grocery.n.01_1", "refrigerator.n.01_1"], []
            )
            self.assertIsNone(
                scope_mismatch(
                    ["grocery.n.01_1", "refrigerator.n.01_1"],
                    str(tro),
                    ignored_scope=["refrigerator.n.01_1"],
                )
            )


if __name__ == "__main__":
    unittest.main()
