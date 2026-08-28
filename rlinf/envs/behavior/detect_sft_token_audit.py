"""Measure raw detect SFT trajectories against real tokenizer/context limits.

The legacy SFT loader keeps the prompt, an initial prefix of turns, and the final
turn when a row is too long.  That is safe for short EQA episodes but silently drops
the middle of long camera trajectories.  This audit quantifies both rollout fit and
the exact fraction of expert calls that would receive loss at each candidate context
length before any training config is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-lengths", default="6144,16384,32768,65536,131072")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from spatialcode.sft_format import format_tool_call

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with open(args.tools) as stream:
        tools = json.load(stream)
    with open(args.dataset) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    records = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt_messages"],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        turn_costs = []
        call_costs = []
        for step in row["tool_steps"]:
            call = format_tool_call(
                step["name"],
                step.get("arguments") or {},
                step.get("reasoning"),
            )
            call_tokens = len(tokenizer.encode(call, add_special_tokens=False))
            response_tokens = len(
                tokenizer.encode(step.get("result_text", ""), add_special_tokens=False)
            )
            call_costs.append(call_tokens)
            turn_costs.append(call_tokens + response_tokens)
        records.append(
            {
                "activity": row["activity"],
                "prompt_tokens": prompt_tokens,
                "turns": len(turn_costs),
                "turn_costs": turn_costs,
                "call_costs": call_costs,
                "total_tokens": prompt_tokens + sum(turn_costs),
            }
        )

    lengths = [int(value) for value in args.seq_lengths.split(",") if value.strip()]
    by_length = {}
    total_turns = sum(record["turns"] for record in records)
    total_call_tokens = sum(sum(record["call_costs"]) for record in records)
    for seq_length in lengths:
        fit = 0
        retained_turns = 0
        retained_call_tokens = 0
        for record in records:
            if record["total_tokens"] <= seq_length:
                fit += 1
                retained_turns += record["turns"]
                retained_call_tokens += sum(record["call_costs"])
                continue
            # Exact policy of truncate_keeping_final: prompt + prefix + final.
            final_cost = record["turn_costs"][-1]
            used = record["prompt_tokens"]
            kept = []
            for index, cost in enumerate(record["turn_costs"][:-1]):
                if used + cost + final_cost > seq_length:
                    break
                kept.append(index)
                used += cost
            kept.append(record["turns"] - 1)
            retained_turns += len(kept)
            retained_call_tokens += sum(record["call_costs"][index] for index in kept)
        by_length[str(seq_length)] = {
            "rollout_fit_rows": fit,
            "rollout_fit_rate": fit / len(records),
            "legacy_sft_retained_turns": retained_turns,
            "legacy_sft_turn_coverage": retained_turns / total_turns,
            "legacy_sft_call_token_coverage": retained_call_tokens / total_call_tokens,
        }

    totals = [record["total_tokens"] for record in records]
    output = {
        "schema": "behavior-detect-sft-token-audit-v1",
        "dataset": str(Path(args.dataset)),
        "tokenizer": args.tokenizer,
        "rows": len(records),
        "turns": total_turns,
        "prompt_tokens_max": max(record["prompt_tokens"] for record in records),
        "total_tokens": {
            "min": min(totals),
            "median": _percentile(totals, 0.5),
            "p90": _percentile(totals, 0.9),
            "max": max(totals),
        },
        "by_seq_length": by_length,
        "largest_rows": [
            {key: record[key] for key in ("activity", "turns", "total_tokens")}
            for record in sorted(
                records, key=lambda item: item["total_tokens"], reverse=True
            )[:10]
        ],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
