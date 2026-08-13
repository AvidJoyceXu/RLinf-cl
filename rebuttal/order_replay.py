"""Replay the greedy RFC-based merge algorithm over task arrival orders.

The decision an order produces is a deterministic function of (a) the pairwise
RFC statistics between the candidate residual and each existing expert, (b) the
threshold tau, and (c) -- when the reuse branch is enabled -- the cross-task
success-rate matrix and theta_reuse. None of that requires retraining, so an
order is replayed purely from cached measurements.

RFC against a *merged* expert is not derivable from the pairwise matrix; it has
to be measured on the merged checkpoint. Rather than silently approximating it,
this script looks the value up in the cache and, if absent, records the missing
measurement and reports it at the end. Run those measurements, add them to the
cache, and re-run.

Usage:
    python rebuttal/order_replay.py --rfc results/rfc_obj_priv/rfc_stats.json \
        --tasks 1 6 7 8 9 --tau-grid 0.0 0.1 0.2 0.3 0.4 0.5 \
        [--sr results/cross_task_sr.json --theta-reuse 0.8] \
        [--orders 1,6,7,8,9 9,7,1,8,6]
"""

import argparse
import itertools
import json
import random
from pathlib import Path

# merge_decision() in analyze_correction_vector_field.py also gates on p_bad
# (fraction of probe states with cosine below delta_dir), but that criterion was
# dropped from the method, so it is not applied here. The scale check is kept at
# the value hard-coded there; only the direction threshold is swept, since that
# is what "tau" refers to.
THRESHOLD_SCALE = 1.0


def key(a, b):
    """Cache key for an unordered pair of task groups."""
    ga = "-".join(str(t) for t in sorted(a))
    gb = "-".join(str(t) for t in sorted(b))
    return "|".join(sorted([ga, gb]))


def can_merge(stats, tau):
    """merge_decision() with the direction threshold set to tau, p_bad dropped."""
    return stats["mean_dir"] >= tau and stats["mean_log_scale"] <= THRESHOLD_SCALE


def replay(order, tau, rfc, sr, theta_reuse, missing):
    """Greedy replay for one arrival order. Returns the resulting expert bank."""
    experts = []  # list of task-id frozensets
    trace = []

    for task in order:
        # 1. reuse: can an existing expert already do the new task?
        if theta_reuse is not None and sr is not None and experts:
            scored = [(sr_lookup(sr, e, task), e) for e in experts]
            scored = [(v, e) for v, e in scored if v is not None]
            if scored:
                best_sr, best_e = max(scored, key=lambda x: x[0])
                if best_sr >= theta_reuse:
                    trace.append(f"T{task}: reuse expert {sorted(best_e)} (SR={best_sr:.3f})")
                    continue

        # 2. otherwise the task-specific residual is trained (here: its checkpoint)
        cand = frozenset([task])

        # 3. merge into the most compatible existing expert, or open a new one
        best = None
        for e in experts:
            k = key(e, cand)
            if k not in rfc:
                missing.add(k)
                continue
            stats = rfc[k]
            if can_merge(stats, tau) and (best is None or stats["mean_dir"] > best[0]):
                best = (stats["mean_dir"], e)

        if best is None:
            experts.append(cand)
            trace.append(f"T{task}: new expert")
        else:
            dc, target = best
            experts.remove(target)
            experts.append(target | cand)
            trace.append(f"T{task}: merge into {sorted(target)} (DC={dc:.4f})")

    return experts, trace


def sr_lookup(sr, expert, task):
    grp = "-".join(str(t) for t in sorted(expert))
    row = sr.get(grp)
    if row is None:
        return None
    return row.get(str(task))


def signature(experts):
    return tuple(sorted(tuple(sorted(e)) for e in experts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfc", required=True, help="JSON of pairwise RFC stats")
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--sr", help="JSON of cross-task SR, for the reuse branch")
    ap.add_argument("--theta-reuse", type=float, default=None)
    ap.add_argument("--orders", nargs="*", default=None,
                    help="comma-separated orders; default = 2 random orders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-orders", action="store_true",
                    help="replay every permutation instead of a sample")
    args = ap.parse_args()

    rfc = json.loads(Path(args.rfc).read_text())
    sr = json.loads(Path(args.sr).read_text()) if args.sr else None

    if args.all_orders:
        orders = [list(p) for p in itertools.permutations(args.tasks)]
    elif args.orders:
        orders = [[int(x) for x in o.split(",")] for o in args.orders]
    else:
        rng = random.Random(args.seed)
        perms = [list(p) for p in itertools.permutations(args.tasks)]
        orders = rng.sample(perms, 2)

    missing = set()
    print(f"tasks={args.tasks}  theta_reuse={args.theta_reuse}  n_orders={len(orders)}")
    for tau in args.tau_grid:
        print(f"\n===== tau = {tau} =====")
        sigs = {}
        for order in orders:
            experts, trace = replay(order, tau, rfc, sr, args.theta_reuse, missing)
            sig = signature(experts)
            sigs.setdefault(sig, []).append(order)
            print(f"  order {','.join(map(str, order))}: "
                  f"{len(experts)} experts {[sorted(e) for e in experts]}")
            for line in trace:
                print(f"      {line}")
        print(f"  -> {len(sigs)} distinct expert bank(s) across {len(orders)} order(s)")

    if missing:
        print("\n!! missing RFC measurements (merged expert vs. new task):")
        for k in sorted(missing):
            print(f"   {k}")
        print("Measure these on the merged checkpoints, add them to the cache, re-run.")


if __name__ == "__main__":
    main()
