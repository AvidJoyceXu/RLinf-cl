"""Phase B prep for iterative RFT: turn scored-trajectory rows into a training set.

Sim-free (reads the JSONL emitted by ``rft_sample.py``); runs in any venv. Two
modes:

* ``raft``   : keep only BDDL-successful trajectories (``reward_terminal == 1``),
               dedup identical tool-call sequences, and drop the RFT sidecar so
               the survivors are plain SFT rows the EXISTING trainer consumes
               unchanged (ReST^EM iteration = SFT on the policy's own successes).
* ``weighted``: keep all trajectories and attach ``advantage`` = group-normalized
               ``reward_staged`` (mean/std within ``group_id``). Rows with
               near-zero |advantage| are dropped. Consumed by the
               advantage-weighted SFT loss (small trainer extension).

Prints the reward distribution so you can see whether RFT has signal (a group
where every sample succeeded or every sample failed contributes zero advantage —
the same variance concern GRPO's staged reward addresses).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def _seq_key(row) -> tuple:
    return tuple((s["name"], json.dumps(s.get("arguments", {}), sort_keys=True))
                 for s in row.get("tool_steps", []))


SIDECAR_KEYS = ("group_id", "instance_id", "reward_terminal", "reward_staged",
                "properly_ended", "advantage")


def _strip_sidecar(row) -> dict:
    return {k: v for k, v in row.items() if k not in SIDECAR_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="scored jsonl from rft_sample")
    ap.add_argument("--out", required=True, help="training jsonl for Phase B")
    ap.add_argument("--mode", choices=["raft", "weighted"], default="raft")
    ap.add_argument("--adv-eps", type=float, default=1e-6)
    ap.add_argument("--min-abs-adv", type=float, default=0.05,
                    help="weighted mode: drop rows with |advantage| below this")
    args = ap.parse_args()

    rows = []
    with open(args.inp) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    # Group stats (report signal / variance).
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("group_id", r["activity"])].append(r)
    n_succ = sum(1 for r in rows if r.get("reward_terminal", 0) >= 1.0)
    live_groups = sum(1 for g in groups.values()
                      if 0 < sum(x.get("reward_terminal", 0) >= 1 for x in g) < len(g))
    print(json.dumps({
        "n_rows": len(rows), "n_groups": len(groups),
        "n_success": n_succ, "overall_success_rate": round(n_succ / max(len(rows), 1), 3),
        "groups_with_reward_variance": live_groups,
    }, indent=2))

    out_rows = []
    if args.mode == "raft":
        seen = set()
        for r in rows:
            if r.get("reward_terminal", 0) < 1.0:
                continue
            k = (r["activity"], _seq_key(r))
            if k in seen:
                continue
            seen.add(k)
            out_rows.append(_strip_sidecar(r))
    else:  # weighted
        for gid, g in groups.items():
            rewards = [x.get("reward_staged", 0.0) for x in g]
            mu = statistics.fmean(rewards)
            sd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            for x, rw in zip(g, rewards):
                adv = (rw - mu) / (sd + args.adv_eps)
                if abs(adv) < args.min_abs_adv:
                    continue
                row = dict(x)
                row["advantage"] = round(adv, 4)
                out_rows.append(row)

    with open(args.out, "w") as fout:
        for r in out_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("RFT_FILTER_DONE " + json.dumps({
        "mode": args.mode, "out": args.out, "n_train_rows": len(out_rows)}))


if __name__ == "__main__":
    main()
