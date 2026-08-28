"""Symbolic expert planner for BEHAVIOR-TextWorld.

Exists for one reason first and another second.

FIRST: to keep the benchmark's coverage claim honest. ``symbolic_world.audit()``
can only check that a goal's *predicates* are in the set some tool can drive. That
is necessary, not sufficient -- ``real(diced__zucchini.n.01_1)`` is an affectable
predicate, but it is unreachable if the activity's scope contains no zucchini to
dice. The only way to know an activity is solvable is to solve it. This planner
does, and ``rlinf/envs/behavior/symbolic_world.py --audit`` reports the verified
number rather than the predicate-compatible one.

SECOND: it is the SFT source for the symbolic benchmark, in the same role
``expert_planner.py`` plays for the geometric one -- privileged, goal-aware, and
never exposed to the policy.

It is deliberately a *greedy achiever*, not a search: read the ground goal atoms,
emit the fixed tool sequence each one needs, run them in order. Long-horizon
BEHAVIOR goals are conjunctions of independent atoms, so ordering rarely matters,
and where it does (``not open(fridge)`` after ``inside(x, fridge)``) sorting
negative atoms last is enough. Anything a greedy pass cannot reach is reported as
unsolved rather than papered over with a search that hides how hard the task is.
"""

from __future__ import annotations

from rlinf.envs.behavior.symbolic_world import (
    SymbolicACI,
    SymbolicWorld,
    producer_of,
    properties_of,
)


def _split(atom) -> tuple[bool, str, list]:
    """``["not", ["open", x]]`` -> ``(False, "open", [x])``."""
    body = list(atom)
    positive = True
    if body and body[0] == "not":
        positive = False
        body = list(body[1])
    return positive, body[0], list(body[1:])


def _reach(world: SymbolicWorld, obj: str) -> list:
    """Navigate to @obj, opening anything shut between the robot and it.

    ``SymbolicACI._require`` refuses every tool on an object sealed in a closed
    container, so a plan that takes a pepper out of a shut fridge has to open the
    fridge first. ``enclosing_closed`` walks outward, so reversing it opens the
    outermost door first -- the order a person would use.
    """
    disp = world.to_display.get
    steps = []
    for container in reversed(world.enclosing_closed(obj)):
        steps += [
            ("go_to", {"name": disp(container)}),
            ("open", {"name": disp(container)}),
        ]
    steps.append(("go_to", {"name": disp(obj)}))
    return steps


def _plan_atom(world: SymbolicWorld, positive: bool, pred: str, args: list) -> list:
    """Tool calls that make one ground atom hold, or [] if none exist."""
    obj = args[0] if args else None
    disp = world.to_display.get

    if pred == "ontop" and positive:
        # The destination needs opening too: `_place` runs `_require` on the target,
        # so a chopping board shut in a cabinet cannot be placed onto either.
        return (
            _reach(world, obj)
            + [("grasp", {"name": disp(obj)})]
            + _reach(world, args[1])
            + [("place_on", {"name": disp(obj), "surface": disp(args[1])})]
        )

    if pred == "nextto" and positive and obj != args[1]:
        # There is no dedicated nextto tool. `place_on` establishes both ontop and
        # nextto in the symbolic dynamics, matching the relation used by existing
        # placement plans. A self-nextto atom is invalid and remains unplanned.
        return (
            _reach(world, obj)
            + [("grasp", {"name": disp(obj)})]
            + _reach(world, args[1])
            + [("place_on", {"name": disp(obj), "surface": disp(args[1])})]
        )

    if pred == "inside" and positive:
        target = args[1]
        steps = _reach(world, obj) + [("grasp", {"name": disp(obj)})]
        steps += _reach(world, target)
        if "openable" in properties_of(target):
            steps.append(("open", {"name": disp(target)}))
        steps.append(("place_inside", {"name": disp(obj), "container": disp(target)}))
        return steps

    if pred == "covered":
        tool = "spray" if positive else "uncover"
        return _reach(world, obj) + [
            (tool, {"name": disp(obj), "system_name": disp(args[1])})
        ]

    if pred == "filled" or pred == "contains":
        if not positive:
            return []
        return _reach(world, obj) + [
            ("fill", {"name": disp(obj), "system_name": disp(args[1])})
        ]

    if pred == "cooked" and positive:
        return _reach(world, obj) + [("cook", {"name": disp(obj)})]

    if pred == "open":
        return _reach(world, obj) + [
            ("open" if positive else "close", {"name": disp(obj)})
        ]

    if pred == "toggled_on":
        return _reach(world, obj) + [
            ("toggle_on" if positive else "toggle_off", {"name": disp(obj)})
        ]

    if pred in ("ontop", "inside") and not positive:
        # Getting an object OFF something: picking it up clears every positional
        # relation, and releasing puts it down where the robot stands. Only reached
        # when the atom is still unsatisfied -- `solve` skips satisfied ones, so a
        # `not ontop(x, floor)` that an earlier `inside(x, box)` already cleared
        # does not undo that placement.
        return _reach(world, obj) + [("grasp", {"name": disp(obj)}), ("release", {})]

    if pred == "real":
        if not positive:
            # `not real(bell_pepper_1)` means the whole object must be consumed --
            # which the slice/dice that creates its products already does. Emitting
            # nothing here is correct, not a gap.
            return []
        producer = producer_of(world, obj)
        if producer is None:
            return []
        whole, tool = producer
        return _reach(world, whole) + [(tool, {"name": disp(whole)})]

    return []


def solve(activity: str, obs_mode: str = "full", max_steps: int = 400) -> dict:
    """Plan and execute one activity in the symbolic layer.

    Returns the trajectory plus whether the REAL bddl evaluator reports success --
    never the planner's own opinion of whether it finished.
    """
    world = SymbolicWorld(activity)
    aci = SymbolicACI(world, obs_mode=obs_mode)

    atoms = world.ground_goal_atoms()
    parsed = [_split(a) for a in atoms]
    # Negative atoms last: `not open(fridge)` must not run before the `inside(x,
    # fridge)` that needs the fridge open.
    parsed.sort(key=lambda t: t[0], reverse=True)

    # Plan and execute ONE ATOM AT A TIME rather than emitting the whole sequence up
    # front. Atoms interact -- `inside(x, box)` already makes `not ontop(x, floor)`
    # true -- so the "is it already satisfied?" test is only meaningful against the
    # state as it actually is when the atom's turn comes.
    trace, rejected, unplanned = [], [], []
    for positive, pred, args in parsed:
        if world.holds(pred, args) == positive:
            continue
        plan = _plan_atom(world, positive, pred, args)
        if not plan:
            if not (pred == "real" and not positive):
                unplanned.append(
                    ("" if positive else "not ") + f"{pred}({', '.join(args)})"
                )
            continue
        for tool, kwargs in plan:
            if len(trace) >= max_steps:
                break
            res = getattr(aci, tool)(**kwargs)
            trace.append(
                {
                    "name": tool,
                    "arguments": kwargs,
                    "ok": bool(res.ok),
                    "reason": res.reason,
                }
            )
            if not res.ok:
                rejected.append(f"{tool}({kwargs}) -> {res.reason}")

    return {
        "activity": activity,
        "success": world.is_success(),
        "goal_status": world.goal_status(),
        "n_steps": len(trace),
        "n_rejected": len(rejected),
        "rejected": rejected[:10],
        "unplanned_atoms": unplanned,
        "trace": trace,
    }


def _main() -> None:
    import argparse
    import collections
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("activity", nargs="?")
    ap.add_argument(
        "--all",
        action="store_true",
        help="solve every predicate-compatible activity and report the "
        "VERIFIED solvable set",
    )
    ap.add_argument("--out", help="write the verified activity list as JSON")
    args = ap.parse_args()

    if not args.all:
        act = args.activity or "picking_up_trash"
        out = solve(act)
        for step in out["trace"]:
            print(
                f"  {'OK ' if step['ok'] else 'REJ'} {step['name']}"
                f"({step['arguments']}) -> {step['reason'][:80]}"
            )
        print(json.dumps({k: v for k, v in out.items() if k != "trace"}, indent=1))
        return

    from rlinf.envs.behavior.symbolic_world import audit

    report = audit()
    candidates = report["solvable"]
    print(f"predicate-compatible: {len(candidates)} / {report['n_total']}")
    solved, failed = [], {}
    reasons: collections.Counter = collections.Counter()
    for i, act in enumerate(candidates, 1):
        try:
            out = solve(act)
        except Exception as ex:  # noqa: BLE001
            failed[act] = f"{type(ex).__name__}: {ex}"
            reasons[type(ex).__name__] += 1
            continue
        if out["success"]:
            solved.append(act)
        else:
            why = (
                "unplanned:"
                + ",".join(sorted({a.split("(")[0] for a in out["unplanned_atoms"]}))
                if out["unplanned_atoms"]
                else f"rejected:{out['n_rejected']}"
            )
            failed[act] = why
            reasons[why.split(":")[0]] += 1
        if i % 100 == 0:
            print(f"  ... {i}/{len(candidates)} solved={len(solved)}", flush=True)

    print(
        f"\nVERIFIED SOLVABLE: {len(solved)} / {len(candidates)} "
        f"predicate-compatible ({report['n_total']} total activities)"
    )
    print("failure classes:")
    for k, v in reasons.most_common(12):
        print(f"  {k:24s} {v:4d}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"solved": solved, "failed": failed}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()
