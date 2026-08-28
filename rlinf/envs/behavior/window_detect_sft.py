"""Convert raw detect trajectories into bounded, loss-complete SFT windows.

Each output row trains exactly one expert tool call. Its context is the canonical
prompt plus the newest *complete* prior call/response pairs that fit below
``max_length - max_new_tokens``. The same compaction primitive is used by the online
BEHAVIOR agent loop. No middle turn and no partial observation is silently cut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from spatialcode.sft_format import (
    compact_complete_turns,
    format_tool_call,
)
from transformers import AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_turn(tokenizer, step: dict) -> tuple[list[int], list[int]]:
    call = format_tool_call(
        step["name"], step.get("arguments") or {}, step.get("reasoning")
    )
    return (
        tokenizer.encode(call, add_special_tokens=False),
        tokenizer.encode(step.get("result_text", ""), add_special_tokens=False),
    )


def build_windows(
    row: dict,
    tokenizer,
    tools: list[dict],
    *,
    max_length: int,
    max_new_tokens: int,
) -> list[dict]:
    prompt_text = tokenizer.apply_chat_template(
        row["prompt_messages"],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    max_context_len = max_length - max_new_tokens
    encoded = [_encode_turn(tokenizer, step) for step in row["tool_steps"]]
    windows = []
    for target_index, (target_call, _) in enumerate(encoded):
        if len(target_call) > max_new_tokens:
            raise ValueError(
                f"{row['activity']} step {target_index} call has {len(target_call)} "
                f"tokens, above max_new_tokens={max_new_tokens}"
            )
        context_ids, first_kept = compact_complete_turns(
            prompt_ids, encoded[:target_index], max_context_len
        )
        history = [
            {**row["tool_steps"][index], "trainable": False}
            for index in range(first_kept, target_index)
        ]
        # The response follows the decision and is unnecessary for its loss. Leaving
        # it out avoids charging every window for an observation no target sees.
        target = {
            **row["tool_steps"][target_index],
            "result_text": "",
            "trainable": True,
        }
        total_tokens = len(context_ids) + len(target_call)
        if total_tokens > max_length:
            raise AssertionError(
                f"window construction exceeded cap: {total_tokens} > {max_length}"
            )
        windows.append(
            {
                **{key: value for key, value in row.items() if key != "tool_steps"},
                "num_turns": len(history) + 1,
                "source_num_turns": len(row["tool_steps"]),
                "target_step": target_index,
                "history_start_step": first_kept,
                "context_tokens": len(context_ids),
                "target_call_tokens": len(target_call),
                "total_tokens": total_tokens,
                "tool_steps": history + [target],
            }
        )
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--val-out",
        help="optional activity-disjoint validation output; requires --val-activities",
    )
    parser.add_argument("--val-activities", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with open(args.tools) as stream:
        tools = json.load(stream)
    with open(args.dataset) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    if bool(args.val_out) != bool(args.val_activities):
        raise ValueError(
            "--val-out and a positive --val-activities must be used together"
        )
    activities = sorted({row["activity"] for row in rows})
    if args.val_activities < 0 or args.val_activities >= len(activities):
        raise ValueError(
            f"val_activities must be in [0, {len(activities) - 1}], "
            f"got {args.val_activities}"
        )
    shuffled = list(activities)
    random.Random(args.split_seed).shuffle(shuffled)
    val_activities = set(shuffled[: args.val_activities])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    val_output = Path(args.val_out) if args.val_out else None
    counts = {"train": 0, "val": 0}
    max_tokens = {"train": 0, "val": 0}
    val_stream = val_output.open("w") if val_output else None
    with output.open("w") as train_stream:
        for row in rows:
            split = "val" if row["activity"] in val_activities else "train"
            stream = val_stream if split == "val" else train_stream
            assert stream is not None
            for window in build_windows(
                row,
                tokenizer,
                tools,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            ):
                stream.write(json.dumps(window, ensure_ascii=False) + "\n")
                counts[split] += 1
                max_tokens[split] = max(max_tokens[split], window["total_tokens"])
    if val_stream:
        val_stream.close()

    outputs = {"train": output}
    if val_output:
        outputs["val"] = val_output
    for split_output in outputs.values():
        Path(str(split_output) + ".tools.json").write_bytes(
            Path(args.tools).read_bytes()
        )
    count = sum(counts.values())
    manifest = {
        "schema": "behavior-detect-sft-window-v1",
        "source_dataset_sha256": _sha256(Path(args.dataset)),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "rows": len(rows),
        "activity_split": {
            "seed": args.split_seed,
            "train": sorted(set(activities) - val_activities),
            "val": sorted(val_activities),
        },
        "windows": counts,
        "supervised_calls": count,
        "source_calls": sum(len(row["tool_steps"]) for row in rows),
        "max_window_tokens": max_tokens,
        "dataset_sha256": {
            split: _sha256(split_output) for split, split_output in outputs.items()
        },
        "tool_schemas_sha256": _sha256(Path(args.tools)),
        "training_ready": True,
    }
    if manifest["supervised_calls"] != manifest["source_calls"]:
        raise AssertionError("not every source call received exactly one loss window")
    for split_output in outputs.values():
        manifest_out = Path(str(split_output) + ".manifest.json")
        with manifest_out.open("w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
