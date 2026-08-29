"""Sim-free regression gates for the BEHAVIOR tool-call benchmark."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from spatialcode.sft_format import (
    assemble_flat_sequence,
    compact_complete_turns,
    include_toolcall_targets,
)
from torch.utils.data import SequentialSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.agents.behavior.behavior_agent_loop import _native_feedback_ids
from rlinf.algorithms.advantages import compute_grpo_dynamic_advantages
from rlinf.algorithms.toolcall_parsers import EQAQwenToolCallParser
from rlinf.envs.behavior.textworld_api_rollout import load_episode_specs
from rlinf.utils.distributed import _dynamic_ending_rates
from rlinf.utils.runner_utils import check_progress


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
_official_scene = _load_pure_module(
    "_behavior_official_scene_test", "official_scene.py"
)
_rgb_manipulation = _load_pure_module(
    "_behavior_rgb_manipulation_test", "rgb_manipulation.py"
)
_dynamic_observation = _load_pure_module(
    "_behavior_dynamic_observation_test", "dynamic_observation.py"
)
_text_detect_manifest = _load_pure_module(
    "_behavior_text_detect_manifest_test", "text_detect_manifest.py"
)
_episode_contract = _load_pure_module(
    "_behavior_episode_contract_test", "episode_contract.py"
)
_rollout_audit = _load_pure_module(
    "_behavior_rollout_audit_test", "audit_text_rollouts.py"
)
_rollout_compare = _load_pure_module(
    "_behavior_rollout_compare_test", "compare_text_rollouts.py"
)
_rollout_store = _load_pure_module("_behavior_rollout_store_test", "rollout_store.py")
_filter_detect_rl = _load_pure_module(
    "_behavior_filter_detect_rl_test", "filter_detect_rl_dataset.py"
)
_memory_manifest = _load_pure_module(
    "_behavior_memory_manifest_test", "build_memory_eval_manifest.py"
)
_memory_context = _load_pure_module(
    "_behavior_memory_context_test", "memory_context.py"
)
_verify_detect_export = _load_pure_module(
    "_behavior_verify_detect_export_test", "verify_detect_sft_export.py"
)
_behavior_reward = _load_pure_module(
    "_behavior_reward_test", "../../algorithms/rewards/behavior.py"
)
_detect = _load_pure_module("_behavior_detect_test", "detect.py")
_layout_source = (
    Path(__file__).resolve().parents[2] / "rlinf/envs/behavior/layout.py"
).read_text()
_env_server_source = (
    Path(__file__).resolve().parents[2] / "rlinf/envs/behavior/env_server.py"
).read_text()
_expert_planner_source = (
    Path(__file__).resolve().parents[2] / "rlinf/envs/behavior/expert_planner.py"
).read_text()
_semantic_tools_source = (
    Path(__file__).resolve().parents[2] / "rlinf/envs/behavior/semantic_tools.py"
).read_text()
_agent_runner_source = (
    Path(__file__).resolve().parents[2] / "rlinf/runners/agent_runner.py"
).read_text()
_api_eval_launcher_source = (
    Path(__file__).resolve().parents[2]
    / "examples/agent/behavior_qwen/eval_text_detect_api.sh"
).read_text()
_reasoning_runner_source = (
    Path(__file__).resolve().parents[2] / "rlinf/runners/reasoning_runner.py"
).read_text()
_sglang_worker_source = (
    Path(__file__).resolve().parents[2]
    / "rlinf/workers/rollout/sglang/sglang_worker.py"
).read_text()
_megatron_worker_source = (
    Path(__file__).resolve().parents[2] / "rlinf/workers/megatron_worker.py"
).read_text()
CAMERA_OBS_MODES = _sft_build.CAMERA_OBS_MODES
PRIMITIVE_CAMERA_OBS_MODES = _sft_build.PRIMITIVE_CAMERA_OBS_MODES
all_schemas = _sft_build.all_schemas
tool_schemas = _sft_build.tool_schemas
build_memory_manifest = _memory_manifest.build_manifest
compact_memory_context = _memory_context.compact_context_blocks
extract_public_memory = _memory_context.extract_public_memory
merge_public_memory = _memory_context.merge_public_memory
render_public_memory = _memory_context.render_public_memory
rendered_context_tokens = _memory_context.rendered_context_tokens
compute_behavior_score = _behavior_reward.compute_score
compute_behavior_staged_rewards = _behavior_reward.compute_staged_rewards
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
DynamicObservationState = _dynamic_observation.DynamicObservationState
scope_scene_names = _detect.scope_scene_names
scope_assets = _detect.scope_assets
project_detections = _detect.detect
project_bbox = _detect._project
world_bbox = _detect.world_bbox


class AgentRunnerClosureTest(unittest.TestCase):
    def test_megatron_update_emits_explicit_parameter_delta_evidence(self):
        required = {
            "param_probe_before",
            "param_probe_after",
            "param_probe_changed",
            "param_probe_l1_delta",
            "param_probe_max_delta",
            "master_param_probe",
            "optimizer_master=True",
        }
        for metric in required:
            self.assertIn(metric, _megatron_worker_source)

    def test_eqa_parser_recovers_multiple_json_objects_in_one_block(self):
        parser = EQAQwenToolCallParser()
        text = (
            '<tool_call>{"name":"observe","arguments":{}}\n'
            '{"name":"scan_next","arguments":{}}</tool_call>'
        )
        content, calls = asyncio.run(parser(text))
        self.assertEqual(content, "")
        self.assertEqual([call.name for call in calls], ["observe", "scan_next"])

    def test_eqa_parser_rejects_trailing_non_json_payload(self):
        parser = EQAQwenToolCallParser()
        text = '<tool_call>{"name":"observe","arguments":{}}oops</tool_call>'
        _, calls = asyncio.run(parser(text))
        self.assertEqual(calls, [])

    def test_rl_trace_exposes_shared_auditor_fields(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "rlinf/agents/behavior/behavior_agent_loop.py"
        ).read_text()
        self.assertIn('"n_steps": generate_context["turn"]', source)
        self.assertIn('**generate_context["budget"]', source)

    def test_hf_evaluator_uses_native_feedback_boundaries(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/behavior/textworld_rollout.py"
        ).read_text()
        self.assertIn('_native_feedback_ids(tokenizer, "tool", response_text)', source)
        self.assertIn('_native_feedback_ids(tokenizer, "user", NUDGE)', source)

    def test_final_step_metrics_and_validation_precede_stop(self):
        tree = ast.parse(_agent_runner_source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentRunner"
        )
        run = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        final_stop = next(
            node
            for node in ast.walk(run)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "is_train_end"
        )
        metric_logs = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log"
        ]
        validation_calls = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_validation"
        ]
        self.assertTrue(metric_logs)
        self.assertEqual(len(validation_calls), 1)
        self.assertLess(max(node.lineno for node in metric_logs), final_stop.lineno)
        self.assertLess(validation_calls[0].lineno, final_stop.lineno)

    def test_runner_allows_epoch_boundary_resume_reset_pass(self):
        tree = ast.parse(_agent_runner_source)
        epoch_ranges = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and any(
                isinstance(arg, ast.BinOp)
                and isinstance(arg.op, ast.Add)
                and isinstance(arg.right, ast.Constant)
                and arg.right.value == 1
                for arg in node.args
            )
        ]
        self.assertTrue(epoch_ranges)
        self.assertIn("self.global_steps >= self.max_steps", _agent_runner_source)

    def test_stateful_dataloader_boundary_resume_reaches_exact_step_limit(self):
        dataset = TensorDataset(torch.arange(4))

        def loader():
            return StatefulDataLoader(
                dataset,
                batch_size=2,
                drop_last=True,
                sampler=SequentialSampler(dataset),
            )

        original = loader()
        iterator = iter(original)
        next(iterator)
        next(iterator)
        boundary_state = original.state_dict()

        resumed = loader()
        resumed.load_state_dict(boundary_state)
        self.assertEqual(list(resumed), [])

        global_steps = 2
        max_steps = 10
        max_epochs = 5
        for _ in range(global_steps // len(resumed), max_epochs + 1):
            for _batch in resumed:
                global_steps += 1
                if global_steps == max_steps:
                    break
            if global_steps == max_steps:
                break
        self.assertEqual(global_steps, max_steps)

    def test_final_step_is_saved_and_declared_terminal(self):
        self.assertEqual(
            check_progress(
                step=23,
                max_steps=24,
                val_check_interval=-1,
                save_interval=24,
                limit_val_batches=1.0,
            ),
            (False, False, False),
        )
        self.assertEqual(
            check_progress(
                step=24,
                max_steps=24,
                val_check_interval=-1,
                save_interval=24,
                limit_val_batches=1.0,
            ),
            (False, True, True),
        )

    def test_sglang_engine_seed_is_explicit(self):
        self.assertIn(
            'random_seed=int(self._cfg_rollout.get("seed", self._cfg.actor.seed))',
            _sglang_worker_source,
        )

    def test_validation_has_a_dedicated_drained_rollout_channel(self):
        tree = ast.parse(_agent_runner_source)
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertIn("eval_rollout_channel", attributes)
        self.assertIn("_run_validation", attributes)
        self.assertIn("get", attributes)

    def test_reasoning_runner_honors_explicit_validation_batch_size(self):
        tree = ast.parse(_reasoning_runner_source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ReasoningRunner"
        )
        build = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_dataloader"
        )
        loader = next(
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StatefulDataLoader"
            and any(
                keyword.arg == "dataset"
                and isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "val_dataset"
                for keyword in node.keywords
            )
        )
        batch_size = next(
            keyword.value for keyword in loader.keywords if keyword.arg == "batch_size"
        )
        self.assertIsInstance(batch_size, ast.Call)
        self.assertIsInstance(batch_size.func, ast.Name)
        self.assertEqual(batch_size.func.id, "int")
        self.assertIsInstance(batch_size.args[0], ast.Name)
        self.assertEqual(batch_size.args[0].id, "val_batch_size")


class EpisodeContractTest(unittest.TestCase):
    def test_detect_runtime_requires_explicit_existing_release_paths(self):
        variables = [
            "BEHAVIOR_INSTANCE_SOURCES",
            "BEHAVIOR_BDDL_DEFINITION_ROOT",
            "BEHAVIOR_NATIVE_BBOX_PATH",
            "BEHAVIOR_ASSET_SCENE_ROOT",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for variable in variables:
                os.environ.pop(variable, None)
            with self.assertRaisesRegex(ValueError, "INSTANCE_SOURCES"):
                _episode_contract.validate_detect_runtime("detect_scope")
            _episode_contract.validate_detect_runtime("full")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definitions = root / "definitions"
            scenes = root / "scenes"
            definitions.mkdir()
            scenes.mkdir()
            bbox = root / "bbox.json"
            bbox.write_text("{}")
            env = {
                "BEHAVIOR_INSTANCE_SOURCES": "2026-v3.9.1",
                "BEHAVIOR_BDDL_DEFINITION_ROOT": str(definitions),
                "BEHAVIOR_NATIVE_BBOX_PATH": str(bbox),
                "BEHAVIOR_ASSET_SCENE_ROOT": str(scenes),
            }
            with patch.dict(os.environ, env, clear=False):
                _episode_contract.validate_detect_runtime("detect_scope")

    def test_frozen_geometry_key_must_match_resolved_instance(self):
        layout = SimpleNamespace(instance_path="/tmp/activity_0_7_template.json")
        self.assertEqual(
            _episode_contract.validate_geometry_instance(
                layout, "activity_0_7_template.json"
            ),
            "activity_0_7_template.json",
        )
        with self.assertRaisesRegex(ValueError, "manifest requested"):
            _episode_contract.validate_geometry_instance(
                layout, "activity_0_8_template.json"
            )

    def test_proper_completion_requires_goal_and_end_task(self):
        goal = {"satisfied": ["g1"], "unsatisfied": []}
        self.assertTrue(
            _episode_contract.outcome_from_goal_status(goal, True)["proper_completion"]
        )
        self.assertFalse(
            _episode_contract.outcome_from_goal_status(goal, False)["proper_completion"]
        )
        incomplete = {"satisfied": ["g1"], "unsatisfied": ["g2"]}
        self.assertFalse(
            _episode_contract.outcome_from_goal_status(incomplete, True)["success"]
        )

    def test_empty_goal_is_not_a_free_success(self):
        outcome = _episode_contract.outcome_from_goal_status(
            {"satisfied": [], "unsatisfied": []}, True
        )
        self.assertFalse(outcome["success"])
        self.assertFalse(outcome["proper_completion"])
        self.assertEqual(_episode_contract.MAX_NUDGES, 3)


class RolloutAuditTest(unittest.TestCase):
    def test_audit_reports_tool_rate_and_interface_failure_classes(self):
        records = [
            {
                "activity": "a",
                "model_turns": 3,
                "n_steps": 2,
                "nudge_count": 1,
                "proper_completion": False,
                "success": False,
                "coverage": 0.25,
                "ended": True,
                "stop_reason": "max_turns",
                "tool_trace": [
                    {
                        "name": "go_to",
                        "arguments": {"object": "h1"},
                        "result": {"ok": True},
                    },
                    {
                        "name": "go_to",
                        "arguments": {"object": "h1"},
                        "result": {
                            "ok": False,
                            "reason": "your last observation is stale; observe again",
                        },
                    },
                ],
            }
        ]
        audit = _rollout_audit.audit_records(records)
        self.assertAlmostEqual(audit["tool_call_rate"], 2 / 3)
        self.assertEqual(audit["nudges"], 1)
        self.assertEqual(audit["refused_calls"], 1)
        self.assertEqual(audit["tool_counts"], {"go_to": 2})
        self.assertEqual(audit["repeated_action_calls"], 1)
        self.assertEqual(audit["repeated_non_observe_action_calls"], 1)
        self.assertEqual(audit["consecutive_duplicate_calls"], 1)
        self.assertEqual(audit["mean_goal_coverage"], 0.25)
        self.assertEqual(audit["unsuccessful_end_tasks"], 1)
        self.assertEqual(audit["interface_violations"], {"stale_handle": 1})
        self.assertEqual(audit["missing_provenance"]["model_id"], 1)
        self.assertGreater(audit["missing_provenance_count"], 0)

    def test_complete_claim_provenance_has_no_missing_fields(self):
        row = dict.fromkeys(_rollout_audit.CLAIM_PROVENANCE_FIELDS, "pinned")
        row.update(
            {
                "activity": "a",
                "model_turns": 1,
                "n_steps": 1,
                "tool_trace": [],
            }
        )
        audit = _rollout_audit.audit_records([row])
        self.assertEqual(audit["missing_provenance"], {})
        self.assertEqual(audit["missing_provenance_count"], 0)


class RolloutComparisonTest(unittest.TestCase):
    def test_paired_bootstrap_uses_activity_level_differences(self):
        result = _rollout_compare.paired_bootstrap(
            {"a": 1.0, "b": 1.0, "c": 1.0},
            {"a": 0.0, "b": 0.0, "c": 0.0},
            samples=1000,
            seed=7,
        )
        self.assertEqual(result["paired_difference"], 1.0)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertTrue(result["criterion_passed"])

    def test_comparison_rejects_denominator_and_replicate_drift(self):
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            _rollout_compare.paired_bootstrap(
                {"a": 1.0}, {"b": 0.0}, samples=100, seed=1
            )
        rows = [
            {
                "activity": "a",
                "replicate_id": "r0",
                "budget_k": 2.0,
                "proper_completion": True,
            }
        ]
        with self.assertRaisesRegex(ValueError, "insufficient replicates"):
            _rollout_compare.arm_activity_means(rows, min_replicates=3, budget_k=2.0)

    def test_comparison_rejects_unmatched_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"activity": "a"}) + "\n")
            digest = _text_detect_manifest._sha256(path)
            candidate = [
                {
                    "activity": "a",
                    "activities_path": str(path),
                    "activities_sha256": digest,
                    "obs_mode": "detect_scope",
                    "goal_format": "nl",
                    "selection_mode": "handle",
                }
            ]
            baseline = [{**candidate[0], "selection_mode": "bbox"}]
            with self.assertRaisesRegex(ValueError, "unmatched selection_mode"):
                _rollout_compare.validate_matched_contract(candidate, baseline)

            matched = [{**candidate[0], "selection_mode": "handle", "max_turns": 4}]
            with self.assertRaisesRegex(ValueError, "budgets do not match"):
                _rollout_compare.validate_matched_contract(
                    [{**candidate[0], "max_turns": 3}], matched
                )

            missing = [{**candidate[0], "activity": "other"}]
            with self.assertRaisesRegex(ValueError, "denominator"):
                _rollout_compare.validate_declared_denominator(missing)


class APIActivitySpecContractTest(unittest.TestCase):
    def test_activity_file_requires_at_prefix(self):
        with self.assertRaisesRegex(ValueError, "must be prefixed with '@'"):
            load_episode_specs("/data/eval/val_s2.jsonl", 1)

    def test_prefixed_activity_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activities.jsonl"
            path.write_text("activity_a\nactivity_b\n")
            self.assertEqual(
                load_episode_specs(f"@{path}", 1),
                [{"activity": "activity_a"}],
            )

    def test_formal_launcher_prefixes_activity_manifest(self):
        self.assertIn(
            "--activities @/data/behavior-data/text_detect_scan_v5/rl/val_s2.jsonl",
            _api_eval_launcher_source,
        )


class RolloutEpisodeStoreTest(unittest.TestCase):
    def test_interrupted_rows_resume_and_finalize_in_declared_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "k2.jsonl"
            keys = [
                _rollout_store.episode_key(activity="a", sample_index=0),
                _rollout_store.episode_key(activity="b", sample_index=0),
            ]
            contract = {"model": "m", "budget_k": 2, "seed": 7}
            first = _rollout_store.EpisodeStore(
                output,
                contract=contract,
                expected_keys=keys,
                resume=True,
            )
            first.commit(1, keys[1], {"activity": "b", "coverage": 0.5})
            self.assertFalse(output.exists())
            self.assertTrue(Path(f"{output}.partial/000001.json").is_file())

            resumed = _rollout_store.EpisodeStore(
                output,
                contract=contract,
                expected_keys=keys,
                resume=True,
            )
            self.assertEqual(resumed.completed_count, 1)
            self.assertTrue(resumed.is_complete(1))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resumed.finalize()
            resumed.commit(0, keys[0], {"activity": "a", "coverage": 1.0})
            records = resumed.finalize()
            self.assertEqual([row["activity"] for row in records], ["a", "b"])
            written = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["activity"] for row in written], ["a", "b"])
            self.assertTrue(all(row["run_contract_sha256"] for row in written))

    def test_resume_rejects_contract_drift_and_unapproved_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curve.jsonl"
            keys = [_rollout_store.episode_key(activity="a")]
            _rollout_store.EpisodeStore(
                output,
                contract={"budget_k": 1},
                expected_keys=keys,
                resume=True,
            )
            with self.assertRaisesRegex(FileExistsError, "pass --resume"):
                _rollout_store.EpisodeStore(
                    output,
                    contract={"budget_k": 1},
                    expected_keys=keys,
                    resume=False,
                )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                _rollout_store.EpisodeStore(
                    output,
                    contract={"budget_k": 2},
                    expected_keys=keys,
                    resume=True,
                )

    def test_store_rejects_duplicate_denominator_and_record_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            key = _rollout_store.episode_key(activity="a")
            with self.assertRaisesRegex(ValueError, "duplicate episode keys"):
                _rollout_store.EpisodeStore(
                    output,
                    contract={},
                    expected_keys=[key, key],
                    resume=True,
                )
            store = _rollout_store.EpisodeStore(
                output,
                contract={},
                expected_keys=[key],
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "episode key mismatch"):
                store.commit(0, _rollout_store.episode_key(activity="other"), {})


class RLSmokeSubsetTest(unittest.TestCase):
    def test_filter_preserves_declared_order_and_rejects_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "all.jsonl"
            source.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": activity,
                            "solutions": json.dumps({"activity": activity}),
                        }
                    )
                    + "\n"
                    for activity in ("a", "b", "c")
                )
            )
            output = root / "smoke.jsonl"
            result = _filter_detect_rl.filter_rows(source, output, ["c", "a"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["prompt"] for row in rows], ["c", "a"])
            self.assertEqual(len(result["output_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "absent"):
                _filter_detect_rl.filter_rows(source, output, ["missing"])


class DetectExportManifestTest(unittest.TestCase):
    def test_manifest_verification_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model.safetensors"
            shard.write_bytes(b"weights")
            digest = _text_detect_manifest._sha256(shard)
            (root / "EXPORT_MANIFEST.sha256").write_text(
                f"{digest}  ./model.safetensors\n"
            )
            result = _verify_detect_export.verify_manifest(root, required=True)
            self.assertEqual(result["files"], 1)
            shard.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _verify_detect_export.verify_manifest(root, required=True)


class BoundedToolHistoryTest(unittest.TestCase):
    class NativeTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(text)

        @staticmethod
        def apply_chat_template(messages, *, tools, tokenize, add_generation_prompt):
            rendered = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return list(rendered) if tokenize else rendered

    def test_native_feedback_restores_role_boundaries_and_generation_header(self):
        tokenizer = self.NativeTokenizer()
        suffix = _native_feedback_ids(
            tokenizer,
            "tool",
            '{"ok": true}',
        )
        self.assertEqual(
            "".join(suffix),
            '</assistant><tool>{"ok": true}</tool><assistant>',
        )

    def test_native_feedback_does_not_retokenize_generated_assistant_text(self):
        tokenizer = self.NativeTokenizer()
        with self.assertRaisesRegex(ValueError, "feedback_role"):
            _native_feedback_ids(tokenizer, "assistant", "y")
        suffix = _native_feedback_ids(tokenizer, "user", "retry")
        self.assertEqual("".join(suffix), "</assistant><user>retry</user><assistant>")

    def test_loss_eval_keeps_targets_but_generation_eval_does_not(self):
        self.assertTrue(include_toolcall_targets(False, "generation"))
        self.assertTrue(include_toolcall_targets(True, "loss"))
        self.assertFalse(include_toolcall_targets(True, "generation"))

    def test_compaction_keeps_newest_complete_turns(self):
        prompt = [1, 2, 3]
        turns = [([10], [11, 12]), ([20, 21], [22]), ([30], [31])]
        context, first = compact_complete_turns(prompt, turns, max_context_len=8)
        self.assertEqual(first, 1)
        self.assertEqual(context, [1, 2, 3, 20, 21, 22, 30, 31])

    def test_compaction_refuses_partial_newest_observation(self):
        with self.assertRaisesRegex(ValueError, "newest complete turn"):
            compact_complete_turns([1, 2, 3], [([10], [11, 12, 13])], 6)

    def test_explicit_window_masks_history_and_trains_target(self):
        ids, mask = assemble_flat_sequence(
            [1, 2],
            [([10, 11], [12]), ([20, 21], [])],
            trainable=[False, True],
        )
        self.assertEqual(ids, [1, 2, 10, 11, 12, 20, 21])
        self.assertEqual(mask, [True, True, True, True, True, False, False])

    def test_trainable_flags_must_align_with_turns(self):
        with self.assertRaisesRegex(ValueError, "flags for 1 turns"):
            assemble_flat_sequence([1], [([2], [3])], trainable=[])


class TextDetectManifestTest(unittest.TestCase):
    def test_frozen_split_contract_accepts_only_seed_zero_592_148(self):
        split = {
            "seed": 0,
            "train_activities": [f"train_{index}" for index in range(592)],
            "s2_heldout_activities": [f"s2_{index}" for index in range(148)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(split))
            loaded, digest = _text_detect_manifest._load_frozen_split(str(path))
        self.assertEqual(loaded, split)
        self.assertEqual(len(digest), 64)

    def test_redrawn_or_overlapping_split_fails_closed(self):
        cases = [
            {"seed": 1, "train_activities": ["a"], "s2_heldout_activities": ["b"]},
            {"seed": 0, "train_activities": ["a"], "s2_heldout_activities": ["a"]},
        ]
        for split in cases:
            with self.subTest(split=split):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "split.json"
                    path.write_text(json.dumps(split))
                    with self.assertRaises(ValueError):
                        _text_detect_manifest._load_frozen_split(str(path))


class DynamicObservationStateTest(unittest.TestCase):
    def _state(self):
        return DynamicObservationState(
            {"item": (1, 2, 0.2), "target": (4, 5, 0.5)},
            {"item": (0, 0, 0, 1)},
        )

    def test_hide_show_and_commit_keep_reset_geometry_immutable(self):
        state = self._state()
        self.assertEqual(state.position("item"), (1.0, 2.0, 0.2))
        state.hide("item")
        self.assertIsNone(state.position("item"))
        self.assertEqual(state.last_position("item"), (1.0, 2.0, 0.2))
        self.assertEqual(state.revision, 0)
        state.show_at("item", (8, 9, 1.0))
        state.commit()
        self.assertEqual(state.position("item"), (8.0, 9.0, 1.0))
        self.assertEqual(state.revision, 1)
        self.assertEqual(
            state.pose_source("item", sampled_initial=True),
            "symbolic_transition",
        )

    def test_placement_preview_is_deterministic_and_does_not_consume_slot(self):
        state = self._state()
        kwargs = {
            "target_center": (4, 5, 0.5),
            "target_extent": (1.0, 0.8, 0.4),
            "object_extent": (0.1, 0.1, 0.1),
            "slot": state.next_placement_slot("target", "ontop"),
        }
        first = state.target_relative_center("item", "target", "ontop", **kwargs)
        second = state.target_relative_center("item", "target", "ontop", **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(state.next_placement_slot("target", "ontop"), 0)
        self.assertAlmostEqual(first[2], 1.02)
        state.record_placement("target", "ontop")
        self.assertEqual(state.next_placement_slot("target", "ontop"), 1)

    def test_inside_and_ontop_are_distinct_transition_geometries(self):
        state = self._state()
        common = {
            "target_center": (4, 5, 0.5),
            "target_extent": (1.0, 0.8, 0.4),
            "object_extent": (0.1, 0.1, 0.1),
        }
        inside = state.target_relative_center(
            "item", "target", "inside", slot=0, **common
        )
        ontop = state.target_relative_center(
            "item", "target", "ontop", slot=0, **common
        )
        self.assertEqual(inside[2], 0.5)
        self.assertEqual(ontop[2], 1.02)
        self.assertNotEqual(inside[:2], ontop[:2])


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
        for mode in PRIMITIVE_CAMERA_OBS_MODES:
            names = [tool["name"] for tool in all_schemas(mode)]
            self.assertEqual(len(names), 25, msg=mode)
            self.assertEqual(len(names), len(set(names)), msg=mode)
            self.assertEqual(set(names) - set(core_names), expected_camera, msg=mode)

    def test_scan_mode_exposes_only_target_independent_controller(self):
        core_names = [tool["name"] for tool in tool_schemas()]
        names = [tool["name"] for tool in all_schemas("detect_scope_scan")]
        self.assertEqual(len(names), 18)
        self.assertEqual(names[:-1], core_names)
        self.assertEqual(names[-1], "scan_next")
        self.assertFalse(
            {"turn_left", "move_ahead", "strafe_left", "look_down"} & set(names)
        )
        prompt = _sft_build.build_prompt_messages(
            "task",
            [],
            "detect_scope_scan",
            goal_nl="do it",
            selection_mode="handle",
        )[1]["content"]
        self.assertIn("scan_next()", prompt)
        self.assertIn("takes no target", prompt)
        self.assertNotIn("move_ahead", prompt)

    def test_memory_scan_adds_only_scene_landmark_revisit(self):
        core_names = [tool["name"] for tool in tool_schemas()]
        tools = all_schemas("detect_scope_memory")
        names = [tool["name"] for tool in tools]
        self.assertEqual(len(names), 19)
        self.assertEqual(names[:-2], core_names)
        self.assertEqual(names[-2:], ["scan_next", "scan_at"])
        scan_at = tools[-1]
        self.assertEqual(scan_at["parameters"]["required"], ["landmark"])
        self.assertFalse(
            {"turn_left", "move_ahead", "strafe_left", "look_down"} & set(names)
        )
        prompt = _sft_build.build_prompt_messages(
            "task",
            [],
            "detect_scope_memory",
            goal_nl="do it",
            selection_mode="handle",
        )[1]["content"]
        self.assertIn("activity-local", prompt)
        self.assertIn("session-stable", prompt)
        self.assertIn("scan_at", prompt)

    def test_detect_schema_exposes_exactly_one_selection_channel(self):
        handle = {
            tool["name"]: tool
            for tool in all_schemas("detect_scope", selection_mode="handle")
        }
        bbox = {
            tool["name"]: tool
            for tool in all_schemas("detect_scope", selection_mode="bbox")
        }
        handle_name = handle["go_to"]["parameters"]["properties"]["name"]
        bbox_name = bbox["go_to"]["parameters"]["properties"]["name"]
        self.assertIn("dN handle", handle_name["description"])
        self.assertNotIn("bbox", handle_name["description"])
        self.assertIn("bbox or point", bbox_name["description"])
        self.assertNotIn("dN handle", bbox_name["description"])
        for schemas in (handle, bbox):
            held = schemas["place_inside"]["parameters"]["properties"]["name"]
            self.assertEqual(held["enum"], ["held"])

    def test_invalid_selection_mode_is_rejected_before_a_run(self):
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            all_schemas("detect_scope", selection_mode="both")

    def test_detect_prompt_names_only_the_selected_channel(self):
        handle = _sft_build.build_prompt_messages(
            "task", [], "detect_scope", goal_nl="do it", selection_mode="handle"
        )[1]["content"]
        bbox = _sft_build.build_prompt_messages(
            "task", [], "detect_scope", goal_nl="do it", selection_mode="bbox"
        )[1]["content"]
        self.assertIn("only by its handle", handle)
        self.assertNotIn("copied box", handle)
        self.assertIn("only by a copied box", bbox)
        self.assertNotIn("d1, d2", bbox)


class MemoryManifestTest(unittest.TestCase):
    @staticmethod
    def _row(activity, scene, turns, split="train"):
        return {
            "activity": activity,
            "split": split,
            "scene": scene,
            "instance_source": "release",
            "geometry_instance_key": f"{activity}.json",
            "oracle_turns": turns,
        }

    def test_counterbalances_targets_and_uses_distinct_donor_scenes(self):
        panel = [
            self._row(f"target_{index}", "target_scene", 10 + index, split="s2")
            for index in range(5)
        ]
        donors = [
            self._row(f"donor_{index}", f"donor_scene_{index}", 10 + index)
            for index in range(5)
        ]
        source = {
            "obs_mode": "detect_scope_scan",
            "content_sha256": "source-hash",
            "rows": panel + donors,
        }

        manifest = build_memory_manifest(
            source, panel_scene="target_scene", panel_size=5
        )

        self.assertEqual(manifest["counts"]["cases"], 25)
        self.assertEqual(manifest["obs_mode"], "detect_scope_memory")
        for target in panel:
            positions = {
                case["position"]
                for case in manifest["cases"]
                if case["target"]["activity"] == target["activity"]
            }
            self.assertEqual(positions, {1, 2, 3, 4, 5})
        for case in manifest["cases"]:
            self.assertEqual(len(case["same_scene_prior"]), case["position"] - 1)
            donor_scenes = [row["scene"] for row in case["diff_scene_prior"]]
            self.assertEqual(len(donor_scenes), len(set(donor_scenes)))
            self.assertNotIn("target_scene", donor_scenes)
        self.assertEqual(
            manifest,
            build_memory_manifest(source, panel_scene="target_scene", panel_size=5),
        )


class MemoryContextTest(unittest.TestCase):
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return json.dumps(messages, sort_keys=True)

        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(text)

    def test_summary_keeps_only_public_scene_landmark_facts(self):
        trace = [
            {
                "payload": {
                    "view": {
                        "scene_id": "scene-a",
                        "landmark": {
                            "id": "l2",
                            "category": "cabinet",
                            "room": "kitchen_0",
                        },
                    },
                    "detections": [
                        {
                            "det": "d9",
                            "track": "s7",
                            "bbox": [1, 2, 3, 4],
                            "score": 0.99,
                            "category": "plate",
                            "landmark": "l2",
                            "scope": "private_scope",
                        }
                    ],
                }
            }
        ]
        memory = extract_public_memory(trace)
        self.assertEqual(memory["scene_id"], "scene-a")
        self.assertEqual(
            memory["landmarks"]["l2"],
            {"categories": ["cabinet", "plate"], "rooms": ["kitchen_0"]},
        )
        encoded = json.dumps(memory, sort_keys=True)
        for private in ("d9", "s7", "bbox", "score", "private_scope"):
            self.assertNotIn(private, encoded)
        self.assertEqual(render_public_memory(memory), render_public_memory(memory))

    def test_merge_rejects_cross_scene_summary(self):
        with self.assertRaisesRegex(ValueError, "different scenes"):
            merge_public_memory(
                {"scene_id": "scene-a", "landmarks": {}},
                {"scene_id": "scene-b", "landmarks": {}},
            )

    def test_context_compaction_drops_only_whole_activity_blocks(self):
        tokenizer = self.Tokenizer()
        system = {"role": "system", "content": "system"}
        current = {"role": "user", "content": "current"}
        blocks = [
            [
                {"role": "user", "content": "old-a"},
                {"role": "assistant", "content": "done-a"},
            ],
            [
                {"role": "user", "content": "old-b"},
                {"role": "assistant", "content": "done-b"},
            ],
        ]
        one_block_tokens = rendered_context_tokens(
            tokenizer, [system, *blocks[1], current], []
        )
        prior, metadata = compact_memory_context(
            tokenizer,
            system_message=system,
            current_user_message=current,
            blocks=blocks,
            tools=[],
            max_tokens=one_block_tokens,
        )
        self.assertEqual(prior, blocks[1])
        self.assertEqual(metadata["blocks_dropped"], 1)
        self.assertEqual(metadata["blocks_kept"], 1)


class BehaviorRewardTest(unittest.TestCase):
    @staticmethod
    def _turn(
        name,
        *,
        initial=0.0,
        final=0.0,
        success=False,
        ok=True,
        reason="",
        **extra,
    ):
        return {
            "name": name,
            "result": {"ok": ok, "reason": reason},
            "meta": {
                "initial_atom_coverage": initial,
                "atom_coverage": final,
                "is_success": success,
            },
            **extra,
        }

    def test_empty_and_no_end_returns_keep_terminal_penalty(self):
        self.assertEqual(compute_behavior_score([]), -0.5)
        self.assertEqual(sum(compute_behavior_staged_rewards([])), -0.5)
        unchanged = [self._turn("observe", initial=0.5, final=0.5)]
        self.assertEqual(sum(compute_behavior_staged_rewards(unchanged)), -0.5)
        partial = [self._turn("observe", initial=0.0, final=0.5)]
        self.assertAlmostEqual(sum(compute_behavior_staged_rewards(partial)), -0.35)

    def test_reward_uses_net_atom_progress_and_dominant_completion(self):
        proper = [self._turn("end_task", initial=0.25, final=1.0, success=True)]
        premature = [
            self._turn("end_task", initial=0.0, final=0.5, ok=False, success=False)
        ]
        self.assertAlmostEqual(sum(compute_behavior_staged_rewards(proper)), 1.225)
        fixed = {"premature_end_penalty": -0.5}
        self.assertAlmostEqual(
            sum(compute_behavior_staged_rewards(premature, **fixed)), -0.37
        )
        self.assertGreater(
            sum(compute_behavior_staged_rewards(proper, **fixed)),
            sum(compute_behavior_staged_rewards(premature, **fixed)),
        )

    def test_fixed_reward_ranks_partial_progress_above_zero_progress_stop(self):
        partial = [self._turn("observe", initial=0.0, final=0.5)]
        premature = [
            self._turn("end_task", initial=0.0, final=0.0, ok=False, success=False)
        ]
        shaping = {"premature_end_penalty": -0.5}
        partial_reward = sum(compute_behavior_staged_rewards(partial, **shaping))
        premature_reward = sum(compute_behavior_staged_rewards(premature, **shaping))
        self.assertAlmostEqual(partial_reward, -0.35)
        self.assertAlmostEqual(premature_reward, -0.52)
        self.assertGreater(partial_reward, premature_reward)

    def test_format_and_refusal_penalties_are_separate(self):
        trace = [
            self._turn(
                "__format__",
                ok=False,
                reason="no_tool_call",
                policy_error="no_tool_call",
            ),
            self._turn(
                "invented",
                ok=False,
                reason="unknown_tool:invented",
                policy_error="unknown_tool",
            ),
            self._turn(
                "go_to",
                ok=False,
                reason="bad_arguments: missing name",
                policy_error="bad_arguments",
            ),
            self._turn("grasp", ok=False, reason="not near target"),
            self._turn("observe", format_violations=2),
        ]
        self.assertAlmostEqual(sum(compute_behavior_staged_rewards(trace)), -0.77)

    def test_ragged_grpo_dynamic_preserves_groups_and_both_signs(self):
        trajectory_traces = [
            [self._turn("end_task", final=1.0, success=True)],
            [self._turn("observe", final=0.0)],
            [self._turn("observe", final=0.5)],
            [self._turn("end_task", final=0.5, ok=False)],
            [self._turn("end_task", initial=0.25, final=1.0, success=True)],
            [self._turn("observe", initial=0.25, final=0.25)],
            [self._turn("observe", initial=0.25, final=0.75)],
            [self._turn("end_task", initial=0.25, final=0.5, ok=False)],
        ]
        trajectory_rewards = [
            sum(compute_behavior_staged_rewards(trace)) for trace in trajectory_traces
        ]
        turn_counts = [1, 2, 3, 1, 2, 1, 3, 2]
        idx_to_traj = [
            trajectory
            for trajectory, count in enumerate(turn_counts)
            for _ in range(count)
        ]
        rewards = torch.tensor(
            [trajectory_rewards[index] for index in idx_to_traj],
            dtype=torch.float32,
        ).unsqueeze(-1)
        loss_mask = torch.ones((5, len(idx_to_traj)), dtype=torch.bool)
        loss_mask[4, ::2] = False

        advantages, returns = compute_grpo_dynamic_advantages(
            rewards,
            loss_mask,
            group_size=4,
            idx_to_traj=idx_to_traj,
            advantage_mode="trajectory",
        )

        self.assertIsNone(returns)
        self.assertEqual(advantages.shape, loss_mask.shape)
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue((advantages[loss_mask] > 0).any())
        self.assertTrue((advantages[loss_mask] < 0).any())
        self.assertTrue((advantages[~loss_mask] == 0).all())
        for trajectory in range(8):
            columns = [
                index for index, owner in enumerate(idx_to_traj) if owner == trajectory
            ]
            values = advantages[0, columns]
            self.assertTrue(torch.allclose(values, values[0].expand_as(values)))

    def test_ragged_grpo_exact_values_and_zero_variance(self):
        rewards = torch.tensor([1.0, 1.0, 3.0, 10.0, 10.0, 10.0, 10.0])[:, None]
        idx_to_traj = [0, 0, 1, 2, 2, 3, 3]
        loss_mask = torch.ones((3, len(idx_to_traj)), dtype=torch.bool)

        advantages, _ = compute_grpo_dynamic_advantages(
            rewards,
            loss_mask,
            group_size=2,
            idx_to_traj=idx_to_traj,
            advantage_mode="trajectory",
        )

        expected = torch.tensor([-(2**-0.5), -(2**-0.5), 2**-0.5, 0.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(advantages[0], expected, atol=2e-6, rtol=0.0)
        self.assertTrue((advantages[:, 3:] == 0).all())

    def test_dynamic_end_metric_uses_trajectory_denominator(self):
        ended_rate, terminal_turn_fraction = _dynamic_ending_rates(8, 80, 16)
        self.assertEqual(ended_rate, 0.5)
        self.assertEqual(terminal_turn_fraction, 0.1)


class InstanceSourcePolicyTest(unittest.TestCase):
    def test_instance_choice_does_not_depend_on_file_mtime(self):
        tree = ast.parse(_layout_source)
        find_instance = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_find_instance"
        )
        calls = [node for node in ast.walk(find_instance) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute) and call.func.attr == "getmtime"
                for call in calls
            )
        )
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "sorted"
                and any(
                    keyword.arg == "key"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "_instance_sort_key"
                    for keyword in call.keywords
                )
                for call in calls
            )
        )

    def test_default_layout_sources_exclude_local_samples(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BEHAVIOR_INSTANCE_SOURCES", None)
            names = [
                source.name for source in selected_layout_sources(data_root="/tmp/x")
            ]
        self.assertEqual(names, ["2026-v3.9.1", "2025-official"])
        self.assertNotIn("local-v3.7", names)

    def test_unknown_and_duplicate_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            selected_layout_sources("invented", data_root="/tmp/x")
        with self.assertRaises(ValueError):
            selected_layout_sources("2025-official,2025-official", data_root="/tmp/x")

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

    def test_scope_bound_scene_names_can_be_removed_from_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = Path(directory) / "task_template-tro_state.json"
            tro.write_text("{}")
            template = Path(directory) / "task_template.json"
            template.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "task": {
                                "inst_to_name": {
                                    "agent.n.01_1": "robot",
                                    "cabinet.n.01_1": "bottom_cabinet_0",
                                    "cup.n.01_1": "cup_7",
                                }
                            }
                        }
                    }
                )
            )
            self.assertEqual(
                scope_scene_names(str(tro)),
                frozenset({"bottom_cabinet_0", "cup_7"}),
            )

    def test_scope_asset_scale_is_preserved_for_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            tro = Path(directory) / "task_template-tro_state.json"
            tro.write_text("{}")
            template = Path(directory) / "task_template.json"
            template.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "task": {"inst_to_name": {"fridge.n.01_1": "fridge_0"}}
                        },
                        "objects_info": {
                            "init_info": {
                                "fridge_0": {
                                    "args": {
                                        "model": "abc123",
                                        "scale": [1.5, 2.0, 0.75],
                                    }
                                }
                            }
                        },
                    }
                )
            )
            self.assertEqual(
                scope_assets(str(tro))["fridge.n.01_1"],
                {"model": "abc123", "scale": (1.5, 2.0, 0.75)},
            )

    def test_world_bbox_rotates_scaled_extent_and_base_link_offset(self):
        centre, extent = world_bbox(
            (10.0, 20.0, 1.0),
            (2.0, 0.5, 0.25),
            local_offset=(1.0, 0.0, 0.0),
            scale=(2.0, 1.0, 1.0),
            orientation=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        )
        for actual, expected in zip(centre, (10.0, 22.0, 1.0)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(extent, (0.5, 4.0, 0.25)):
            self.assertAlmostEqual(actual, expected)

    def test_support_host_can_be_exempted_from_aabb_occlusion(self):
        view = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, pitch=0.0)
        entries = [
            ("scope:host", "cabinet", (2.0, 0.0, 1.2), (1.0, 1.0, 1.0)),
            ("scope:item", "cup", (3.0, 0.0, 1.2), (0.1, 0.1, 0.1)),
        ]
        hidden = project_detections(view, entries)
        self.assertNotIn("scope:item", {d.key for d in hidden})
        visible = project_detections(
            view,
            entries,
            occlusion_exempt_pairs=frozenset({("scope:item", "scope:host")}),
        )
        self.assertIn("scope:item", {d.key for d in visible})

    def test_projection_depth_is_bbox_centre_not_nearest_corner(self):
        view = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, pitch=0.0)
        projected = project_bbox(view, (3.0, 0.0, 1.2), (2.0, 0.5, 0.5))
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[1], 3.0)

    def test_projection_rejects_bbox_entirely_behind_camera(self):
        view = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, pitch=0.0)
        self.assertIsNone(project_bbox(view, (-3.0, 0.0, 1.2), (0.5, 0.5, 0.5)))


class HarnessCapabilityTest(unittest.TestCase):
    def test_omnigibson_fov_schema_has_dispatch_methods(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "rlinf/envs/behavior/semantic_tools.py").read_text())
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
            "detect_scope_scan",
            "detect_scope_memory",
        ):
            validate_harness("textworld", mode)


class TrajectoryVideoTest(unittest.TestCase):
    def test_official_scene_contract_uses_full_template_and_room_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "2026-challenge-task-instances"
            json_dir = root / "scenes" / "scene_a" / "json"
            instance_dir = json_dir / "scene_a_task_task_a_instances"
            metadata_dir = root / "metadata"
            instance_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "B100_task_misc.csv").write_text(
                '1,task_a,"kitchen_0\nliving_room_0"\n'
            )
            base = "scene_a_task_task_a_0_0_template"
            objects = {
                "table_0": {
                    "class_name": "DatasetObject",
                    "args": {"category": "table"},
                },
                "wall_0": {
                    "class_name": "DatasetObject",
                    "args": {"category": "walls"},
                },
            }
            payload = {"objects_info": {"init_info": objects}}
            (json_dir / f"{base}.json").write_text(json.dumps(payload))
            (json_dir / f"{base}-partial_rooms.json").write_text(
                json.dumps(
                    {"objects_info": {"init_info": {"table_0": objects["table_0"]}}}
                )
            )

            contract = _official_scene.resolve_official_scene_contract(
                activity="task_a",
                scene_model="scene_a",
                instance_dir=instance_dir,
            )

        self.assertEqual(contract.room_instances, ("kitchen_0", "living_room_0"))
        self.assertEqual(contract.full_template["dataset_object_count"], 2)
        self.assertEqual(contract.partial_room_template["dataset_object_count"], 1)
        self.assertTrue(contract.full_template["path"].endswith(f"{base}.json"))
        self.assertNotIn("partial_rooms", contract.full_template["path"])

    def test_official_scene_contract_rejects_non_2026_instance_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "2026-challenge-task-instances"):
                _official_scene.resolve_official_scene_contract(
                    activity="task_a",
                    scene_model="scene_a",
                    instance_dir=directory,
                )

    def test_debug_server_uses_physical_head_sensor_and_exposes_fov_cli(self):
        tree = ast.parse(_env_server_source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "configure_debug_renderer"
                for call in calls
            ),
            "Isaac 5.1 debug video must retain the official renderer defaults",
        )
        self.assertIn("zed_link:Camera:0", _env_server_source)
        self.assertNotIn("og.sim.viewer_camera", _env_server_source)
        obs_mode_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "--obs-mode"
        ]
        self.assertEqual(len(obs_mode_calls), 1)
        choices = next(
            keyword.value
            for keyword in obs_mode_calls[0].keywords
            if keyword.arg == "choices"
        )
        self.assertIn("fov", [elt.value for elt in choices.elts])

    def test_one_video_and_aligned_sidecar_per_complete_session(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryVideoRecorder(directory, fps=2)
            video = recorder.begin(
                {"activity": "task_a", "instance_id": 7, "session_id": "abc"}
            )
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            recorder.append(
                frame,
                tool="session_start",
                ok=True,
                event_metadata={"yaw_deg": 90.0, "pitch_deg": 0.0},
            )
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
            self.assertEqual(record["events"][0]["yaw_deg"], 90.0)

    def test_rgb_manipulation_is_explicit_and_absent_from_textworld(self):
        self.assertIn("rgb_manipulation_evidence: bool = False", _env_server_source)
        tree = ast.parse(_env_server_source)
        parent = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "rlinf.envs.behavior.rgb_manipulation"
        ]
        self.assertEqual(len(imports), 1)
        cursor = parent.get(imports[0])
        guarded = False
        while cursor is not None:
            if isinstance(
                cursor, ast.If
            ) and "rgb_manipulation_evidence" in ast.unparse(cursor.test):
                guarded = True
                break
            cursor = parent.get(cursor)
        self.assertTrue(guarded, "RGB helper import must stay behind the opt-in flag")

        root = Path(__file__).resolve().parents[2] / "rlinf/envs/behavior"
        for filename in ("textworld_env.py", "symbolic_world.py"):
            self.assertNotIn(
                "rgb_manipulation",
                (root / filename).read_text(),
                f"{filename} must not import or call the RGB G5 helper",
            )

    def test_rgb_manipulation_module_has_no_top_level_simulator_import(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/behavior/rgb_manipulation.py"
        )
        tree = ast.parse(path.read_text())
        imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        rendered = "\n".join(ast.unparse(node) for node in imports)
        self.assertNotIn("omnigibson", rendered)

    def test_rgb_aabb_projection_and_pixel_metrics_are_simulator_free(self):
        box = _rgb_manipulation.project_world_aabb(
            [[-0.1, -0.1, -2.1], [0.1, 0.1, -1.9]],
            camera_xyz=[0.0, 0.0, 0.0],
            camera_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            intrinsic=[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            width=100,
            height=100,
        )
        self.assertEqual(box, (44, 44, 56, 56))

        import numpy as np

        before = np.zeros((10, 10, 3), dtype=np.uint8)
        after = before.copy()
        after[2:6, 3:8] = 30
        metrics = _rgb_manipulation.pixel_change_metrics(
            before, after, (3, 2, 8, 6), threshold=12
        )
        self.assertEqual(metrics["pixels"], 20)
        self.assertEqual(metrics["mean_abs_diff"], 30.0)
        self.assertEqual(metrics["changed_pixel_fraction"], 1.0)

        motion = _rgb_manipulation.joint_position_delta(
            {"drawer": [0.0], "door": [0.0, 0.1]},
            {"drawer": [0.35], "door": [0.0, -0.2]},
        )
        self.assertEqual(motion["matching_joints"], 2)
        self.assertEqual(motion["changed_joints"], 2)
        self.assertAlmostEqual(motion["max_abs_delta"], 0.35)
        self.assertEqual(
            _rgb_manipulation.joint_position_delta({}, {"drawer": [0.4]})[
                "max_abs_delta"
            ],
            0.0,
        )

    def test_v391_scope_entities_resolve_before_pose_cache(self):
        self.assertIn(
            "return wrapped if wrapped is not None else entity",
            _expert_planner_source,
        )
        self.assertIn(
            'getattr(entity, "wrapped_obj", None) or entity',
            _semantic_tools_source,
        )
        self.assertLess(
            _env_server_source.index("self.plan = build_expert_plan(self.env.task)"),
            _env_server_source.index("build_cache_on_instance(self.aci"),
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

    def test_hydra_null_cap_means_uncapped(self):
        budget = turn_budget_for(
            "task_a",
            flat_max_turns=40,
            oracle_steps={"task_a": 56},
            budget_k=2,
            budget_cap=None,
        )
        self.assertEqual(budget.max_turns, 112)
        self.assertIsNone(budget.budget_cap)
        self.assertFalse(budget.capped)

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
    def test_single_activity_cli_forwards_privileged_tour_flag(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "rlinf/envs/behavior/viewpoint_search.py").read_text())
        explore_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "explore"
        ]
        self.assertTrue(
            any(
                any(
                    keyword.arg == "use_tour"
                    and isinstance(keyword.value, ast.Attribute)
                    and keyword.value.attr == "tour"
                    for keyword in call.keywords
                )
                for call in explore_calls
            )
        )

    def test_miss_taxonomy_is_mutually_exclusive(self):
        cases = {
            "nonvisual_substance": ([], ["nonvisual_substance"]),
            "missing_pose": ([], ["missing_pose"]),
            "missing_extent": ([], ["missing_extent"]),
            "found_by_primitive": (["visible"], ["visible"]),
            "primitive_search_miss": ([], ["visible"]),
            "sealed_by_closed_container": ([], ["target", "closed_container"]),
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
