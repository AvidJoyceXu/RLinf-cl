"""Is `obs_mode=fov` solvable at all? A reference search, and the audit that uses it.

A harder observation condition is only a benchmark if a competent agent can still win.
Otherwise it is not difficulty, it is a broken environment -- and the difference is
invisible from a success rate alone, which is why this exists before any baseline is
run under `fov`. Same role `symbolic_expert --all` plays for the base benchmark: it
answers "solvable" by *solving*, not by arguing.

THE SEARCH. Sweep in place, and whenever the sweep turns up somewhere new to stand,
walk there and sweep again -- frontier expansion over the objects discovered so far:

    seen = sweep(here)
    frontier = [things seen]
    while frontier and budget:
        go_to(next unvisited thing)      # navigation is still the oracle
        seen |= sweep(here)

Turning is free of side effects and `go_to` only works on something already seen, so
this is exactly the loop the interface affords a policy. It is deliberately NOT
privileged: it reads `observe()` and nothing else, so a number it produces is an upper
bound a policy could in principle reach, not one only an oracle could.
"""
from __future__ import annotations

import json

from rlinf.envs.behavior.symbolic_world import (
    SymbolicACI,
    SymbolicWorld,
    properties_of,
)

TURNS_PER_SWEEP = 4          # 4 x 90 degrees = full circle
# Level and down only. A third (upward) pitch was measured to add nothing to
# coverage and ~10 steps to the median sweep -- most BEHAVIOR objects sit at or
# below camera height. `look_up` stays in the tool set so the policy can undo a
# `look_down`; the reference search just never needs it.
PITCHES = (0, -1)


def sweep(aci: SymbolicACI, on_view=None) -> int:
    """Turn a full circle at every pitch. Returns steps spent.

    @on_view is called after each `observe`, because in `detect` mode the ACI keeps
    only the LAST observation -- harvesting after the sweep returns would see one
    view out of eight. That bug cost a solvability audit reading 0 found when the
    objects were plainly visible mid-sweep.
    """
    steps = 0
    for pitch in PITCHES:
        if pitch == -1:
            aci.look_down()
            steps += 1
        for _ in range(TURNS_PER_SWEEP):
            aci.observe()
            steps += 1
            if on_view is not None:
                on_view()
            aci.turn_left()
            steps += 1
    aci.look_up()            # back to level
    return steps + 1


def explore(activity: str, layout_prefer: str = "sampled",
            budget: int = 400, obs_mode: str = "fov") -> dict:
    """Search until nothing new is found or the budget runs out.

    @obs_mode "fov" tracks `view.seen` (scope names, because the mode still reports
    them); "detect" has no names, so what counts as "found" is a task object that
    appeared as a DETECTION -- which is the honest analogue and the only thing a
    policy could act on.
    """
    world = SymbolicWorld(activity)
    aci = SymbolicACI(world, obs_mode=obs_mode, layout_prefer=layout_prefer)
    targets = {n for n in world.scope_names
               if n != world.agent and world.is_real(n)}
    # Substances with no host have no coordinates and are always reported; counting
    # them as "found" would flatter the search, so they are excluded from the target
    # set rather than silently satisfied.
    locatable = {n for n in targets if aci.layout.pos(n) is not None}

    steps = sweep(aci)
    visited: set = set()
    if obs_mode in SymbolicACI.DETECT_MODES:
        return _explore_detect(world, aci, locatable, steps, budget)
    while steps < budget:
        frontier = [n for n in aci.view.seen if n not in visited
                    and aci.layout.pos(n) is not None]
        if not frontier or locatable <= aci.view.seen:
            break
        # Nearest unvisited first: a cheap heuristic, and one a policy could follow
        # from the ordinal `where` field alone.
        frontier.sort(key=lambda n: aci.view.range_to(aci.layout.pos(n))
                      if hasattr(aci.view, "range_to") else 0)
        target = frontier[0]
        visited.add(target)
        res = aci.go_to(world.to_display.get(target, target))
        steps += 1
        if not res.ok:
            continue
        # Open what you arrive at, if it shuts things away. `fov` keeps `object`
        # mode's rule that a closed container hides its contents, so looking alone can
        # never find the butter in the fridge -- a search that only turns its head
        # scores those activities as unsolvable when they are merely unopened. This
        # is the same move `symbolic_expert._reach` makes for the same reason.
        if "openable" in properties_of(target) and not world.sim.get_open(
                (world.scope[target],)):
            steps += 1
            if aci.open(world.to_display.get(target, target)).ok:
                steps += sweep(aci)
                continue
        steps += sweep(aci)

    found = aci.view.seen & locatable
    return {
        "activity": activity,
        "source": aci.layout.source,
        "locatable": len(locatable),
        "found": len(found),
        "complete": locatable <= aci.view.seen,
        "steps": steps,
        "missing": sorted(world.to_display.get(n, n) for n in (locatable - found))[:5],
    }


def _explore_detect(world, aci, locatable, steps, budget) -> dict:
    """Frontier expansion when selection is spatial rather than by name.

    Everything the fov search does by name is done here by handle: go to a detection,
    open it if it is shut, sweep again. The one asymmetry is that a handle dies the
    moment the camera moves, so the target must be re-acquired after arriving --
    which is the cost the design intends, not a workaround.
    """
    found, visited = set(), set()

    def harvest():
        for d in aci._dets:
            if d.key.startswith("scope:"):
                found.add(d.key[len("scope:"):])

    steps += sweep(aci, on_view=harvest)
    while steps < budget and not locatable <= found:
        target = None
        aci.observe()
        steps += 1
        harvest()
        for d in aci._dets:
            if d.key not in visited:
                target, visited = d, visited | {d.key}
                break
        if target is None:
            break
        if aci.go_to(target.det).ok:
            steps += 1
            aci.observe()
            steps += 1
            harvest()
            # Open EVERY shut container here, not just the first. With true asset
            # sizes a cabinet is large and its contents are fully covered, so a
            # search that opens one container per stop leaves the rest sealed.
            shut = [x for x in aci._dets
                    if x.key.startswith("scope:")
                    and "openable" in properties_of(x.key[len("scope:"):])]
            for x in shut:
                aci.open(x.det)
                steps += 1
            steps += sweep(aci, on_view=harvest)
            # Step back and sweep again. Occlusion is asymmetric coverage from ONE
            # viewpoint, so a small object hidden behind large furniture from where we
            # arrived may be plainly visible half a metre away. Turning alone cannot
            # recover it -- only translation can, which is the difficulty the true
            # sizes introduced and the reason a turn-only search under-reports.
            for _ in range(2):
                aci.turn_left()
                aci.turn_left()
                aci.move_ahead()
                steps += 3
                steps += sweep(aci, on_view=harvest)
        else:
            steps += 1
    return {
        "activity": world.activity, "source": aci.layout.source,
        "locatable": len(locatable), "found": len(found & locatable),
        "complete": locatable <= found, "steps": steps,
        "missing": sorted(world.to_display.get(n, n)
                          for n in (locatable - found))[:5],
    }


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("activity", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="limit --all to the first N")
    ap.add_argument("--prefer", default="sampled", choices=["sampled", "generated"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--obs-mode", default="fov",
                    choices=["fov", "fov_distract", "detect_scope", "detect"])
    args = ap.parse_args()

    if not args.all:
        print(json.dumps(explore(args.activity, args.prefer, args.budget,
                                 args.obs_mode), indent=1))
        return

    acts = json.load(open("/data/behavior-data/tw_verified.json"))["solved"]
    if args.n:
        acts = acts[:args.n]
    complete = partial = fail = 0
    steps_ok, worst = [], []
    for a in acts:
        try:
            r = explore(a, args.prefer, args.budget, args.obs_mode)
        except Exception:                                          # noqa: BLE001
            fail += 1
            continue
        if r["complete"]:
            complete += 1
            steps_ok.append(r["steps"])
        else:
            partial += 1
            worst.append((r["found"] / max(r["locatable"], 1), a, r["missing"]))
    n = len(acts)
    print(f"activities            : {n}   (prefer={args.prefer}, budget={args.budget})")
    print(f"  ALL objects found   : {complete} = {complete / n:.1%}")
    print(f"  incomplete          : {partial}")
    print(f"  errored             : {fail}")
    if steps_ok:
        steps_ok.sort()
        print(f"  search steps        : median {steps_ok[len(steps_ok) // 2]}  "
              f"p90 {steps_ok[int(0.9 * len(steps_ok))]}  max {steps_ok[-1]}")
    if worst:
        worst.sort()
        print("  worst coverage:")
        for frac, a, missing in worst[:5]:
            print(f"    {a:<44} {frac:.0%}  missing e.g. {missing[:3]}")


if __name__ == "__main__":
    _main()
