"""Matched standalone-HF evaluation on the frozen BEHAVIOR text benchmark.

This adapter uses the same NL prompt, schemas, handle channel, source pinning,
complete-turn context compaction, three-nudge rule, oracle-relative action budget,
and proper-completion endpoint as the API and online-RL adapters. It does not train.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import time

from spatialcode.sft_format import (
    compact_complete_turns,
    format_tool_response,
)

from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.episode_contract import (
    MAX_NUDGES,
    NUDGE,
    outcome_from_goal_status,
    validate_detect_runtime,
)
from rlinf.envs.behavior.rft_sample import load_model, result_payload
from rlinf.envs.behavior.rollout_store import (
    EpisodeStore,
    atomic_write_json,
    episode_key,
)
from rlinf.envs.behavior.textworld_api_rollout import load_episode_specs
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld
from rlinf.envs.behavior.turn_budget import turn_budget_for

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_provenance(args) -> dict:
    activities_path = (
        os.path.abspath(args.activities[1:])
        if args.activities.startswith("@")
        else None
    )
    model_path = os.path.abspath(args.model)
    model_manifest = os.path.join(model_path, "EXPORT_MANIFEST.sha256")
    model_config = os.path.join(model_path, "config.json")
    identity_path = (
        model_manifest
        if os.path.isfile(model_manifest)
        else model_config
        if os.path.isfile(model_config)
        else None
    )
    return {
        "adapter": "standalone_hf",
        "model_id": args.model,
        "model_identity_file": identity_path,
        "model_identity_sha256": _sha256(identity_path) if identity_path else None,
        "activities_path": activities_path,
        "activities_sha256": _sha256(activities_path) if activities_path else None,
        "replicate_id": str(args.replicate_id),
        "rlinf_revision": args.rlinf_revision or None,
        "outer_revision": args.outer_revision or None,
        "container_image": args.container_image or None,
        "harness_file_sha256": _sha256(__file__),
        "prompt_mode": "guided",
        "reason_mode": "terse",
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "context_max_length": args.context_max_length,
            "max_new_tokens": args.context_max_new_tokens,
            "stop_sequence": "</tool_call>",
        },
    }


def _tool_call_stopping_criteria(tokenizer):
    """Stop immediately after the first complete serialized tool call."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    stop_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if not stop_ids:
        raise ValueError("tokenizer produced no ids for </tool_call>")

    class StopOnToolCallClose(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            del scores, kwargs
            if input_ids.shape[1] < len(stop_ids):
                return False
            expected = torch.tensor(stop_ids, device=input_ids.device)
            return bool(torch.equal(input_ids[0, -len(stop_ids) :], expected))

    return StoppingCriteriaList([StopOnToolCallClose()])


def _generate(model, tokenizer, prompt_ids: list[int], args) -> tuple[str, int]:
    import torch

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(ids)
    kwargs = {
        "max_new_tokens": args.context_max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "stopping_criteria": _tool_call_stopping_criteria(tokenizer),
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    else:
        # Override sampling-only values inherited from generation_config so
        # deterministic evaluation is explicit and warning-free.
        kwargs["temperature"] = 1.0
        kwargs["top_k"] = 50
    with torch.no_grad():
        output = model.generate(ids, attention_mask=attention_mask, **kwargs)
    generated_ids = output[0, ids.shape[1] :]
    return (
        tokenizer.decode(generated_ids, skip_special_tokens=True),
        int(generated_ids.numel()),
    )


def run_episode(model, tokenizer, spec: dict, args, budget, provenance=None) -> dict:
    started_at = time.monotonic()
    activity = spec["activity"]
    source = spec.get("instance_source")
    env = BehaviorTextWorld(
        activity,
        obs_mode=args.obs_mode,
        selection_mode=args.selection_mode,
        instance_source_selection=source,
    )
    start_info = env.start(
        spec.get("instance_id", 0),
        source,
        spec.get("geometry_instance_key"),
    )
    aci = env.aci
    messages = sb.build_prompt_messages(
        activity,
        env.goal_lines,
        args.obs_mode,
        goal_nl=env.goal_nl if args.goal_format == "nl" else None,
        selection_mode=args.selection_mode,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=sb.all_schemas(args.obs_mode, selection_mode=args.selection_mode),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    max_context_len = args.context_max_length - args.context_max_new_tokens
    compact_complete_turns(prompt_ids, [], max_context_len)

    encoded_turns: list[tuple[list[int], list[int]]] = []
    steps: list[dict] = []
    consecutive_nudges = 0
    nudge_count = 0
    model_turns = 0
    prompt_tokens = 0
    completion_tokens = 0
    context_compactions = 0
    ended = False
    stop_reason = "max_turns"
    error = ""

    while len(steps) < budget.max_turns:
        try:
            context_ids, first_kept = compact_complete_turns(
                prompt_ids, encoded_turns, max_context_len
            )
        except ValueError as exc:
            stop_reason, error = "context_overflow", str(exc)[:400]
            break
        context_compactions += int(first_kept > 0)
        model_turns += 1
        prompt_tokens += len(context_ids)
        generated, generated_tokens = _generate(model, tokenizer, context_ids, args)
        completion_tokens += generated_tokens
        match = TOOL_CALL_RE.search(generated)
        if not match:
            consecutive_nudges += 1
            nudge_count += 1
            if consecutive_nudges > MAX_NUDGES:
                stop_reason = "no_tool_call"
                break
            encoded_turns.append(
                (
                    tokenizer.encode(generated, add_special_tokens=False),
                    tokenizer.encode(NUDGE, add_special_tokens=False),
                )
            )
            continue

        consecutive_nudges = 0
        model_out = generated[: match.end()]
        try:
            call = json.loads(match.group(1))
            name = call["name"]
            call_args = call.get("arguments") or {}
            if not isinstance(call_args, dict):
                raise TypeError("tool arguments must be an object")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            stop_reason, error = "parse_error", str(exc)[:400]
            break

        fn = getattr(aci, name, None)
        if fn is None or name.startswith("_"):
            payload = {"ok": False, "reason": f"unknown_tool:{name}"}
        else:
            try:
                payload = result_payload(name, fn(**call_args))
            except TypeError as exc:
                payload = {"ok": False, "reason": f"bad_arguments: {exc}"}
        response_text = format_tool_response(payload)
        encoded_turns.append(
            (
                tokenizer.encode(model_out, add_special_tokens=False),
                tokenizer.encode(response_text, add_special_tokens=False),
            )
        )
        steps.append(
            {
                "name": name,
                "arguments": call_args,
                "result": payload,
                "model_text": model_out,
            }
        )
        if name == "end_task":
            ended, stop_reason = True, "end_task"
            break

    outcome = outcome_from_goal_status(env.world.goal_status(), ended)
    return {
        **dict(provenance or {}),
        "activity": activity,
        "instance_source": source,
        "geometry_instance_key": start_info.get("geometry_instance_key"),
        "obs_mode": args.obs_mode,
        "goal_format": args.goal_format,
        "selection_mode": args.selection_mode,
        **budget.record(),
        "n_steps": len(steps),
        "model_turns": model_turns,
        "tool_call_rate": len(steps) / model_turns if model_turns else 0.0,
        "nudge_count": nudge_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "seconds": round(time.monotonic() - started_at, 3),
        "context_compactions": context_compactions,
        "ended": ended,
        "stop_reason": stop_reason,
        "error": error,
        **outcome,
        "tools": [step["name"] for step in steps],
        "tool_trace": steps,
        "rejections": [
            str(step["result"].get("reason", ""))[:70]
            for step in steps
            if step["name"] != "end_task" and step["result"].get("ok") is False
        ],
    }


def failure_record(spec: dict, args, budget, error: Exception, provenance=None) -> dict:
    return {
        **dict(provenance or {}),
        "activity": spec["activity"],
        "instance_source": spec.get("instance_source"),
        "geometry_instance_key": spec.get("geometry_instance_key"),
        "obs_mode": args.obs_mode,
        "goal_format": args.goal_format,
        "selection_mode": args.selection_mode,
        **budget.record(),
        "n_steps": 0,
        "model_turns": 0,
        "tool_call_rate": 0.0,
        "nudge_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "seconds": 0.0,
        "context_compactions": 0,
        "ended": False,
        "stop_reason": f"harness_error:{type(error).__name__}",
        "error": str(error)[:400],
        "num_satisfied": 0,
        "num_goal": 0,
        "coverage": 0.0,
        "success": False,
        "proper_completion": False,
        "tools": [],
        "tool_trace": [],
        "rejections": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--activities",
        default="@/data/behavior-data/text_detect_v1/rl/val_s2.jsonl",
    )
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--replicate-id", default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rlinf-revision", default=os.environ.get("RLINF_REVISION"))
    parser.add_argument("--outer-revision", default=os.environ.get("OUTER_REVISION"))
    parser.add_argument("--container-image", default=os.environ.get("CONTAINER_IMAGE"))
    parser.add_argument("--obs-mode", default="detect_scope")
    parser.add_argument("--goal-format", default="nl", choices=["atoms", "nl"])
    parser.add_argument(
        "--selection-mode", default="handle", choices=["handle", "bbox"]
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--budget-k", type=float, default=2.0)
    parser.add_argument("--budget-cap", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--context-max-length", type=int, default=6144)
    parser.add_argument("--context-max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and resume atomically persisted per-episode records",
    )
    args = parser.parse_args()
    validate_detect_runtime(args.obs_mode)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    specs = load_episode_specs(args.activities, args.n)
    oracle = {
        spec["activity"]: int(spec["oracle_turns"])
        for spec in specs
        if spec.get("oracle_turns") is not None
    }
    budgets = {
        spec["activity"]: turn_budget_for(
            spec["activity"],
            flat_max_turns=args.max_turns,
            oracle_steps=oracle,
            budget_k=args.budget_k,
            budget_cap=args.budget_cap,
            allow_missing_oracle=False,
        )
        for spec in specs
    }
    provenance = _run_provenance(args)
    episode_specs = [
        (spec, sample_index) for spec in specs for sample_index in range(args.samples)
    ]
    expected_keys = [
        episode_key(
            activity=spec["activity"],
            instance_source=spec.get("instance_source"),
            geometry_instance_key=spec.get("geometry_instance_key"),
            instance_id=spec.get("instance_id", 0),
            sample_index=sample_index,
        )
        for spec, sample_index in episode_specs
    ]
    store = EpisodeStore(
        args.out,
        contract={
            "adapter": "standalone_hf",
            "provenance": provenance,
            "episode_specs": specs,
            "samples": args.samples,
            "base_seed": args.seed,
            "budgets": {
                activity: budget.record() for activity, budget in budgets.items()
            },
        },
        expected_keys=expected_keys,
        resume=args.resume,
    )
    if store.completed_count:
        print(
            f"resuming {store.completed_count}/{len(episode_specs)} "
            f"persisted episodes from {store.partial_dir}",
            flush=True,
        )
    model = tokenizer = None
    if store.completed_count < len(episode_specs):
        model, tokenizer = load_model(args.model)

    for episode_index, ((spec, sample_index), key) in enumerate(
        zip(episode_specs, expected_keys, strict=True)
    ):
        if store.is_complete(episode_index):
            record = store.record(episode_index)
            print(
                f"[{episode_index + 1}/{len(episode_specs)}] {spec['activity']} "
                f"proper={record['proper_completion']} "
                f"coverage={record['coverage']:.3f} persisted=true",
                flush=True,
            )
            continue
        if model is None or tokenizer is None:
            raise RuntimeError("model was not loaded for a pending episode")
        episode_seed = args.seed + episode_index
        import torch

        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        episode_provenance = {
            **provenance,
            "replicate_id": (
                f"{args.replicate_id}:{sample_index}"
                if args.samples > 1
                else str(args.replicate_id)
            ),
            "run_replicate_id": str(args.replicate_id),
            "sample_index": sample_index,
            "episode_seed": episode_seed,
        }
        try:
            record = run_episode(
                model,
                tokenizer,
                spec,
                args,
                budgets[spec["activity"]],
                episode_provenance,
            )
        except Exception as exc:  # noqa: BLE001
            record = failure_record(
                spec,
                args,
                budgets[spec["activity"]],
                exc,
                episode_provenance,
            )
        store.commit(episode_index, key, record)
        print(
            f"[{episode_index + 1}/{len(episode_specs)}] {spec['activity']} "
            f"proper={record['proper_completion']} "
            f"coverage={record['coverage']:.3f} persisted=true",
            flush=True,
        )

    records = store.records(require_complete=True)
    denominator = max(len(records), 1)
    summary = {
        **provenance,
        "base_seed": args.seed,
        "episodes": len(records),
        "proper_completion_rate": sum(r["proper_completion"] for r in records)
        / denominator,
        "goal_satisfied_rate": sum(r["success"] for r in records) / denominator,
        "goal_coverage": sum(r["coverage"] for r in records) / denominator,
        "end_task_rate": sum(r["ended"] for r in records) / denominator,
        "mean_action_turns": sum(r["n_steps"] for r in records) / denominator,
        "mean_model_turns": sum(r["model_turns"] for r in records) / denominator,
        "tool_call_rate": sum(r["n_steps"] for r in records)
        / max(sum(r["model_turns"] for r in records), 1),
        "nudges": sum(r["nudge_count"] for r in records),
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["completion_tokens"] for r in records),
        "seconds": round(sum(r["seconds"] for r in records), 3),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in records)),
    }
    output = os.path.abspath(args.out)
    atomic_write_json(output.replace(".jsonl", ".summary.json"), summary)
    store.finalize()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
