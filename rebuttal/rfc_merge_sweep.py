"""Replay the paper's greedy RFC merge (Eq. 6) over a tau sweep.

For each arriving task the candidate residual is compared against every expert
already in the bank; the best match is picked and merged if its RFC clears tau,
otherwise the candidate becomes a new expert:

    j* = argmax_j RFC(R_cand, R_j)      merge iff RFC(R_cand, R_j*) >= tau

RFC between two *single-task* residuals comes from the measured pairwise matrix.
RFC against a *merged* expert is not derivable from that matrix -- it has to be
measured on the merged checkpoint. Missing entries are reported rather than
approximated, so the loop is: replay -> measure what it asks for -> replay again.

    python rebuttal/rfc_merge_sweep.py --rfc results/rfc_rgbfixed \
        --order 1 6 7 8 9 --tau-grid 0.25 0.2 0.15 0.1 0.05 0 -0.05 -0.1
"""

import argparse
import json
import re
from pathlib import Path


def group_key(group):
    return "-".join(str(t) for t in sorted(group))


def pair_key(a, b):
    return "|".join(sorted([group_key(a), group_key(b)]))


def load_rfc_dir(path: Path) -> dict:
    """Scrape RFC values out of the per-pair analysis logs."""
    cache = {}
    for log in sorted(path.glob("pair_*.log")):
        m = re.match(r"pair_(\d+)_(\d+)$", log.stem)
        if not m:
            continue
        i, j = int(m.group(1)), int(m.group(2))
        txt = log.read_text(errors="ignore")
        vals = re.findall(r"RFC \(DC \* MC\): (-?[0-9.]+)", txt)
        if vals:
            cache[pair_key([i], [j])] = float(vals[-1])
    return cache


def replay(order, tau, cache):
    """Return (bank, decisions, missing) for one arrival order at one tau."""
    bank = []          # list of groups, each a list of task ids
    decisions = []
    missing = []

    for task in order:
        if not bank:
            bank.append([task])
            decisions.append((task, None, None, "new (first)"))
            continue

        scored = []
        for expert in bank:
            k = pair_key([task], expert)
            if k not in cache:
                missing.append((task, tuple(expert)))
                scored.append((None, expert))
            else:
                scored.append((cache[k], expert))

        known = [(v, e) for v, e in scored if v is not None]
        if len(known) < len(scored):
            # Cannot decide this step without the missing measurements.
            decisions.append((task, None, None, "BLOCKED (missing RFC)"))
            bank.append([task])
            continue

        best_val, best_expert = max(known, key=lambda x: x[0])
        if best_val >= tau:
            best_expert.extend([task])
            decisions.append((task, group_key(best_expert), best_val, "merge"))
        else:
            bank.append([task])
            decisions.append((task, group_key(best_expert), best_val, "new"))

    return bank, decisions, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfc", type=Path, required=True, help="dir of pair_*.log")
    ap.add_argument("--order", type=int, nargs="+", default=[1, 6, 7, 8, 9])
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[0.25, 0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, -0.15, -0.2])
    ap.add_argument("--extra-cache", type=Path, default=None,
                    help="json of measured merged-expert RFCs, {pair_key: value}")
    args = ap.parse_args()

    cache = load_rfc_dir(args.rfc)
    if args.extra_cache and args.extra_cache.exists():
        cache.update(json.loads(args.extra_cache.read_text()))

    print(f"# pairwise RFC entries loaded: {len(cache)}")
    vals = sorted(cache.values())
    if vals:
        print(f"# RFC range: [{vals[0]:.4f}, {vals[-1]:.4f}]")
    print(f"# arrival order: {args.order}\n")

    all_missing = set()
    banks_seen = {}
    for tau in args.tau_grid:
        bank, decisions, missing = replay(list(args.order), tau, cache)
        sig = " | ".join(sorted(group_key(g) for g in bank))
        all_missing.update(missing)
        banks_seen.setdefault(sig, []).append(tau)
        flag = "  <-- needs measurement" if missing else ""
        print(f"tau={tau:>6.2f}  experts={len(bank)}  bank: {sig}{flag}")
        for task, best, val, what in decisions:
            v = f"{val:.4f}" if val is not None else "  n/a "
            print(f"           task {task}: best={best or '-':<8} RFC={v}  -> {what}")
        print()

    print("=== distinct expert banks over the sweep ===")
    for sig, taus in banks_seen.items():
        print(f"  {sig:<24} tau in {taus}")

    if all_missing:
        print("\n=== RFC measurements still needed (merged expert vs candidate) ===")
        for task, expert in sorted(all_missing):
            print(f"  RFC( task{task} , merged[{group_key(expert)}] )")


if __name__ == "__main__":
    main()
