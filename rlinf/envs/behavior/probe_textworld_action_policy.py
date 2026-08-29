"""Compare checkpoint action preference on fixed TextWorld observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from spatialcode.sft_format import compact_complete_turns, format_tool_response

from rlinf.agents.behavior.behavior_agent_loop import _native_feedback_ids
from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.rft_sample import load_model, result_payload
from rlinf.envs.behavior.textworld_api_rollout import load_episode_specs
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld
from rlinf.envs.behavior.textworld_rollout import _sha256, _sha256_files

CANDIDATES = ("scan_next", "observe", "end_task")


def _tool_call(name: str) -> str:
    payload = json.dumps(
        {"name": name, "arguments": {}}, ensure_ascii=False, separators=(",", ":")
    )
    return f"<tool_call>{payload}</tool_call>"


def _candidate_logprob(model, prompt_ids: list[int], candidate_ids: list[int]) -> dict:
    import torch

    input_ids = torch.tensor(
        [prompt_ids + candidate_ids], dtype=torch.long, device=model.device
    )
    with torch.inference_mode():
        logits = model(input_ids).logits[0, len(prompt_ids) - 1 : -1]
        token_logprobs = logits.log_softmax(dim=-1).gather(
            1, input_ids[0, len(prompt_ids) :].unsqueeze(1)
        )
    values = token_logprobs.squeeze(1).float().cpu()
    return {
        "tokens": len(candidate_ids),
        "sum_logprob": float(values.sum()),
        "mean_logprob": float(values.mean()),
    }


def _model_identity(model_path: Path) -> dict:
    manifest = model_path / "EXPORT_MANIFEST.sha256"
    if manifest.is_file():
        paths = [str(manifest)]
    else:
        paths = [str(path) for path in sorted(model_path.glob("*.safetensors"))]
        config = model_path / "config.json"
        if config.is_file():
            paths.append(str(config))
    if not paths:
        raise FileNotFoundError(f"no model identity files in {model_path}")
    return {
        "model_identity_files": paths,
        "model_identity_sha256": _sha256_files(paths),
    }


def probe_activity(model, tokenizer, spec: dict, prefix_scans: int) -> dict:
    env = BehaviorTextWorld(
        spec["activity"],
        obs_mode="detect_scope_scan",
        selection_mode="handle",
        instance_source_selection=spec.get("instance_source"),
    )
    env.start(
        spec.get("instance_id", 0),
        spec.get("instance_source"),
        spec.get("geometry_instance_key"),
    )
    messages = sb.build_prompt_messages(
        spec["activity"],
        env.goal_lines,
        "detect_scope_scan",
        goal_nl=env.goal_nl,
        selection_mode="handle",
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=sb.all_schemas("detect_scope_scan", selection_mode="handle"),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    encoded_turns = []
    for _ in range(prefix_scans):
        call_text = _tool_call("scan_next")
        call_ids = tokenizer.encode(call_text, add_special_tokens=False)
        observation = result_payload("scan_next", env.aci.scan_next())
        feedback = format_tool_response(observation)
        encoded_turns.append(
            (call_ids, _native_feedback_ids(tokenizer, "tool", feedback))
        )
    context_ids, _ = compact_complete_turns(prompt_ids, encoded_turns, 5632)
    scores = {
        name: _candidate_logprob(
            model,
            context_ids,
            tokenizer.encode(_tool_call(name), add_special_tokens=False),
        )
        for name in CANDIDATES
    }
    max_score = max(score["sum_logprob"] for score in scores.values())
    normalizer = sum(
        math.exp(score["sum_logprob"] - max_score) for score in scores.values()
    )
    for score in scores.values():
        score["restricted_probability"] = (
            math.exp(score["sum_logprob"] - max_score) / normalizer
        )
    return {
        "activity": spec["activity"],
        "context_tokens": len(context_ids),
        "prefix_scans": prefix_scans,
        "candidates": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--activities", required=True)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--prefix-scans", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_scans < 0:
        raise SystemExit("--prefix-scans must be nonnegative")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    specs = load_episode_specs(args.activities, args.n)
    model, tokenizer = load_model(str(args.model))
    rows = [probe_activity(model, tokenizer, spec, args.prefix_scans) for spec in specs]
    means = {
        name: sum(row["candidates"][name]["restricted_probability"] for row in rows)
        / len(rows)
        for name in CANDIDATES
    }
    activities_path = Path(args.activities.removeprefix("@")).resolve()
    payload = {
        "schema": "behavior-textworld-action-policy-probe-v1",
        "model": str(args.model.resolve()),
        **_model_identity(args.model.resolve()),
        "activities": str(activities_path),
        "activities_sha256": _sha256(str(activities_path)),
        "prefix_scans": args.prefix_scans,
        "candidate_contract": list(CANDIDATES),
        "probability_scope": "softmax restricted to the three serialized candidates",
        "mean_restricted_probability": means,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
