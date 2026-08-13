"""Roll the SFT checkpoint out against BEHAVIOR-TextWorld, without Ray or Megatron.

The debugging surface ``env_server.py`` is for the OmniGibson backend: boot one
activity, drive one episode, read what the policy actually said. This is the same
thing for the symbolic backend, and it is much more useful there because an episode
costs milliseconds -- so a hundred activities is a coffee break rather than a day.

Answers the questions that a full training launch answers slowly and expensively:

  * does the policy emit parseable ``<tool_call>`` blocks at all?
  * does it terminate (``end_task``), and how often does it stop early?
  * does the reward signal move -- goal coverage, then success?
  * do tools get REJECTED, and for which precondition?

Nothing here trains. Success is read from the real BDDL evaluator via
``aci.goal_status()``; a rejection rate is a measurement, not a failure.

    python -m rlinf.envs.behavior.textworld_rollout \\
        --model /data/behavior-data/sft/hf_step120 \\
        --activities @/data/behavior-data/tw/val_s2.jsonl --n 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from rlinf.envs.behavior.rft_sample import load_model, rollout_once
from rlinf.envs.behavior.symbolic_world import SymbolicACI, SymbolicWorld
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld


def _activities(spec: str, limit: int) -> list[str]:
    """A comma list, or ``@path`` to a dataset JSONL / plain activity list."""
    if not spec.startswith("@"):
        acts = [a.strip() for a in spec.split(",") if a.strip()]
    else:
        acts, path = [], spec[1:]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    acts.append(json.loads(row["solutions"])["activity"])
                else:
                    acts.append(line)
    return acts[:limit] if limit > 0 else acts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/data/behavior-data/sft/hf_step120")
    ap.add_argument("--activities", default="picking_up_trash")
    ap.add_argument("--n", type=int, default=0, help="cap the activity count")
    ap.add_argument("--samples", type=int, default=1,
                    help="rollouts per activity (a GRPO group, to see variance)")
    ap.add_argument("--obs-mode", default="full",
                    choices=["full", "partial", "object"])
    ap.add_argument("--goal-format", default="atoms", choices=["atoms", "nl"],
                    help="atoms matches what SFT step 120 was trained on; nl is the "
                         "primary research condition and needs an NL cold start")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", help="write per-episode records as JSONL")
    args = ap.parse_args()

    acts = _activities(args.activities, args.n)
    print(f"model      : {args.model}")
    print(f"activities : {len(acts)}  x {args.samples} sample(s)  "
          f"obs={args.obs_mode} goal={args.goal_format} T={args.temperature}")
    model, tok = load_model(args.model)

    records = []
    rejects: collections.Counter = collections.Counter()
    n_success = n_ended = n_parsed = 0
    coverage_sum = 0.0

    for i, activity in enumerate(acts, 1):
        try:
            env = BehaviorTextWorld(activity, obs_mode=args.obs_mode)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  [{i}/{len(acts)}] {activity}: SKIP {type(ex).__name__}: {ex}")
            continue
        for _ in range(args.samples):
            world = SymbolicWorld(activity)
            aci = SymbolicACI(world, obs_mode=args.obs_mode)
            goal = env.goal_lines if args.goal_format == "atoms" else [env.goal_nl]
            steps, trace = rollout_once(
                model, tok, aci, activity, goal, args.obs_mode,
                args.max_turns, args.temperature, args.top_p)

            meta = trace[-1]["meta"] if trace else {"num_satisfied": 0, "num_goal": 0,
                                                    "is_success": False}
            coverage = (meta["num_satisfied"] / meta["num_goal"]
                        if meta["num_goal"] else 0.0)
            ended = bool(steps) and steps[-1]["name"] == "end_task"
            n_parsed += bool(steps)
            n_ended += ended
            n_success += bool(meta["is_success"])
            coverage_sum += coverage
            for s in steps:
                # end_task carries no `reason` and its ok=False just means "the goal
                # was not met" -- already reported as end_task_rate minus
                # success_rate. Counting it here only fills the table with blanks.
                if s["name"] == "end_task":
                    continue
                payload = json.loads(s["result_text"])
                if isinstance(payload, dict) and payload.get("ok") is False:
                    rejects[str(payload.get("reason", ""))[:60]] += 1
            records.append({"activity": activity, "n_steps": len(steps),
                            "ended": ended, "coverage": coverage,
                            "success": bool(meta["is_success"]),
                            "tools": [s["name"] for s in steps]})
        if i % 10 == 0 or i == len(acts):
            print(f"  [{i}/{len(acts)}] success={n_success} ended={n_ended} "
                  f"parsed={n_parsed}", flush=True)

    n = max(len(records), 1)
    print("\n--- rollout summary "
          "(the three numbers to read first, per CLAUDE.md) ---")
    print(f"  episodes           : {len(records)}")
    print(f"  parse_rate         : {n_parsed / n:.3f}   (emitted >=1 valid tool call)")
    print(f"  end_task_rate      : {n_ended / n:.3f}")
    print(f"  goal_coverage      : {coverage_sum / n:.3f}")
    print(f"  success_rate       : {n_success / n:.3f}")
    print("  NOTE: end_task_rate minus success_rate is premature termination -- "
          "the failure mode to watch.")
    if rejects:
        print("\n  top tool rejections:")
        for reason, count in rejects.most_common(10):
            print(f"    {count:5d}  {reason}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
