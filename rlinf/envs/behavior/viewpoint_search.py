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

import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter

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
#
# TWO downward steps, not one. The camera is at 1.2 m and `PITCH_LIMIT_RAD` is 60
# degrees, so a single -30 step still leaves anything close and low below the frame --
# an object 1 m away at floor level sits about 50 degrees down. Measured per object
# over a full sweep on 4 instance-backed activities: level only 8/28 found,
# (0, -30) 18/28, (0, -30, -60) 20/28. The sweep costs one extra `look_down` and one
# extra circle, against a search that spends a median 36 steps of a 400 budget.
PITCHES = (0, -1, -2)
ACTIVITY_SOURCE = "/data/behavior-data/tw_verified.json"


def _sha256(path: str) -> str | None:
    """Hash a small benchmark input, or return None when it is unavailable."""
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activities_from_source(path: str) -> tuple[list[str], str]:
    """Load a pinned list or derive one from an official ``scenes`` directory."""
    if os.path.isdir(path):
        directories = glob.glob(os.path.join(path, "*", "json", "*_task_*_instances"))
        activities = sorted(
            {
                os.path.basename(directory)
                .split("_task_", 1)[1]
                .rsplit("_instances", 1)[0]
                for directory in directories
            }
        )
    else:
        with open(path) as stream:
            payload = json.load(stream)
        activities = payload if isinstance(payload, list) else payload.get(
            "solved", payload.get("activities")
        )
    if not isinstance(activities, list) or not activities or not all(
        isinstance(activity, str) for activity in activities
    ):
        raise ValueError("activity source must resolve to a non-empty string list")
    canonical = json.dumps(sorted(activities), separators=(",", ":")).encode()
    return activities, hashlib.sha256(canonical).hexdigest()


def sweep(aci: SymbolicACI, on_view=None) -> int:
    """Turn a full circle at every pitch. Returns steps spent.

    @on_view is called after each `observe`, because in `detect` mode the ACI keeps
    only the LAST observation -- harvesting after the sweep returns would see one
    view out of eight. That bug cost a solvability audit reading 0 found when the
    objects were plainly visible mid-sweep.
    """
    steps = 0
    prev_pitch = 0
    for pitch in PITCHES:
        while prev_pitch > pitch:            # one look_down per step of pitch
            aci.look_down()
            steps += 1
            prev_pitch -= 1
        for _ in range(TURNS_PER_SWEEP):
            aci.observe()
            steps += 1
            if on_view is not None:
                on_view()
            aci.turn_left()
            steps += 1
    while prev_pitch < 0:                    # back to level
        aci.look_up()
        steps += 1
        prev_pitch += 1
    return steps



def _waypoints(aci, spacing=2.5):
    """Coarse coverage waypoints over the scene's own furniture positions.

    The blind-translation search walks a lattice of headings through whatever corridor
    it started in; it has no way to know that a room it never entered exists. This
    reads the scene layout -- which the oracle is allowed to do -- and returns one
    waypoint per @spacing-metre cell of occupied floor, so a tour visits every part of
    the scene that has furniture in it.
    """
    from rlinf.envs.behavior.detect import scene_furniture

    cells = {}
    for so in scene_furniture(aci.layout.scene or ""):
        key = (round(so.pos[0] / spacing), round(so.pos[1] / spacing))
        cells.setdefault(key, so.pos)
    here = (aci.view.x, aci.view.y)
    pts = sorted(cells.values(),
                 key=lambda p: (p[0] - here[0]) ** 2 + (p[1] - here[1]) ** 2)
    return [(float(p[0]), float(p[1])) for p in pts]


def _target_orbit_waypoints(aci):
    """Three rings of eight target-facing views around every required object.

    The scene-furniture grid covers rooms, not necessarily the small object itself.
    This second privileged pass distinguishes "our coarse tour never stood nearby"
    from "the analytic geometry hides the object from every tested side".

    These are geometric candidate viewpoints, not navmesh certificates. Physical
    reachability remains a synchronized OmniGibson smoke-test obligation.
    """
    from rlinf.envs.behavior.detect import (
        extent_for_model,
        offset_for_model,
        scope_assets,
        world_bbox,
    )

    assets = scope_assets(aci.layout.instance_path or "")
    for name in aci.world.scope_names:
        if name == aci.world.agent or not aci.world.is_real(name):
            continue
        base_pos = aci.layout.pos(name)
        if base_pos is None:
            continue
        radius = 1.2
        pos = base_pos
        asset = assets.get(name)
        if asset:
            native = extent_for_model(asset["model"])
            if native:
                pos, world_extent = world_bbox(
                    base_pos,
                    native,
                    local_offset=offset_for_model(asset["model"]),
                    scale=asset["scale"],
                    orientation=aci.layout.orientation.get(
                        name, (0.0, 0.0, 0.0, 1.0)
                    ),
                )
                radius = max(
                    radius,
                    math.hypot(world_extent[0], world_extent[1]) + 0.5,
                )
        for ring_offset in (0.0, 1.0, 2.0):
            ring_radius = radius + ring_offset
            for angle_index in range(8):
                angle = angle_index * math.pi / 4.0
                x = pos[0] + ring_radius * math.cos(angle)
                y = pos[1] + ring_radius * math.sin(angle)
                yaw = math.atan2(pos[1] - y, pos[0] - x)
                yield x, y, yaw


def _focused_pitch_sweep(aci, on_view=None) -> int:
    """Look at one known target at every supported pitch, without turning away."""
    steps = 0
    for pitch_index in PITCHES:
        aci.view.pitch = pitch_index * math.radians(30.0)
        aci._view_epoch += 1
        aci.observe()
        steps += 1
        if on_view is not None:
            on_view()
    aci.view.pitch = 0.0
    aci._view_epoch += 1
    return steps


def tour(aci, budget, on_view=None) -> int:
    """PRIVILEGED coverage tour: stand at every occupied cell of the scene and sweep.

    This answers a different question from `explore` and the two must not be confused.

        explore   unprivileged, primitive-only. Its step count is what an
                  ORACLE-RELATIVE turn budget for a policy is computed from.
        tour      privileged: it teleports the viewpoint to waypoints derived from the
                  scene layout. Its step count is NOT a policy budget. It answers
                  "is this object findable from ANY reachable viewpoint at all?" --
                  the solvability certificate that tells breakage apart from difficulty.

    Keeping both is the point. If `tour` finds everything and `explore` does not, the
    mode is solvable and our search is weak. If `tour` also misses objects, the geometry
    or the instance is wrong and no amount of search strength will fix it.
    """
    opened = set()

    def _look_and_open():
        # OPEN WHAT IS SHUT, IN THIS VIEW. Standing everywhere is not enough: a sealed
        # cabinet hides its contents from every viewpoint in the scene, so a tour that
        # only looks reports them unfindable and understates solvability.
        #
        # This MUST happen inside the sweep callback rather than after it. `sweep`
        # leaves `aci._dets` holding only its LAST view, so a check afterwards sees one
        # view out of twelve -- measured: 77 detections accumulated, 0 openables found,
        # on an activity whose scope contains a washer and a cabinet. Det handles are
        # view-local and expire when the camera moves, so the only moment a container
        # can be opened is while it is still in frame.
        if on_view is not None:
            on_view()
        for x in list(aci._dets):
            if not x.key.startswith("scope:") or x.key in opened:
                continue                 # open each container ONCE, not once per view
            if "openable" in properties_of(x.key[len("scope:"):]):
                # Navigate with the still-live handle, then reacquire it from the
                # new frame before opening. The old tour moved only `view`, leaving
                # `_at` unchanged, so every open failed `not near` and all contents
                # were misclassified as outside range.
                key = x.key
                if not aci.go_to(x.det).ok:
                    continue
                # A tall nearby appliance can leave the level and -30-degree
                # frames while filling the -60-degree frame (measured on the
                # cook_bacon refrigerator). Reacquire at every supported pitch.
                # Direct pitch assignment is acceptable here because this is the
                # privileged findability tour; its step count is never a policy
                # budget. The policy has the equivalent look_down tools.
                saved_pitch = aci.view.pitch
                target_name = key[len("scope:"):]
                target_pos = aci.layout.pos(target_name)
                stands = [(aci.view.x, aci.view.y)]
                if target_pos is not None:
                    current_radius = max(
                        1.2,
                        math.hypot(
                            aci.view.x - target_pos[0],
                            aci.view.y - target_pos[1],
                        ),
                    )
                    # If the approach side is blocked, orbit two reachable radii.
                    # This is a privileged certificate search, not a policy action
                    # trace; the policy has equivalent strafe/move primitives.
                    for radius in (current_radius, current_radius + 1.0):
                        for angle_index in range(8):
                            angle = angle_index * math.pi / 4.0
                            stands.append(
                                (
                                    target_pos[0] + radius * math.cos(angle),
                                    target_pos[1] + radius * math.sin(angle),
                                )
                            )
                did_open = False
                for stand_x, stand_y in stands:
                    if target_pos is not None:
                        yaw = math.atan2(
                            target_pos[1] - stand_y,
                            target_pos[0] - stand_x,
                        )
                        aci.view.teleport(stand_x, stand_y, yaw)
                    for pitch_index in PITCHES:
                        aci.view.pitch = pitch_index * math.radians(30.0)
                        aci._view_epoch += 1
                        aci.observe()
                        reacquired = next(
                            (d for d in aci._dets if d.key == key), None
                        )
                        if reacquired is not None and aci.open(reacquired.det).ok:
                            opened.add(key)
                            did_open = True
                            break
                    if did_open:
                        break
                aci.view.pitch = saved_pitch
                aci._view_epoch += 1
                # go_to invalidated every other handle in the old list. Return to
                # the sweep; later views will discover additional containers.
                return

    steps = 0
    for (wx, wy) in _waypoints(aci):
        if steps >= budget:
            break
        aci.view.teleport(wx, wy)
        steps += 1
        steps += sweep(aci, on_view=_look_and_open)
    for wx, wy, yaw in _target_orbit_waypoints(aci):
        if steps >= budget:
            break
        aci.view.teleport(wx, wy, yaw)
        steps += 1
        steps += _focused_pitch_sweep(aci, on_view=_look_and_open)
    return steps

def explore(activity: str, layout_prefer: str = "sampled",
            budget: int = 400, obs_mode: str = "fov",
            use_tour: bool = False) -> dict:
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
        return _explore_detect(world, aci, targets, locatable, steps, budget, use_tour)
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
        "instance_source": aci.layout.instance_source or None,
        "bddl_release": aci.layout.bddl_release or None,
        "asset_release": aci.layout.asset_release or None,
        "instance_path": aci.layout.instance_path or None,
        "locatable": len(locatable),
        "found": len(found),
        "complete": locatable <= aci.view.seen,
        "steps": steps,
        "missing": sorted(world.to_display.get(n, n) for n in (locatable - found))[:5],
    }


def _explore_detect(world, aci, targets, locatable, steps, budget, use_tour=False) -> dict:
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

    explored = 0
    steps += sweep(aci, on_view=harvest)
    if use_tour:
        steps += tour(aci, budget - steps, on_view=harvest)
    while (not use_tour) and steps < budget and not locatable <= found:
        target = None
        aci.observe()
        steps += 1
        harvest()
        for d in aci._dets:
            if d.key not in visited:
                target, visited = d, visited | {d.key}
                break
        if target is None:
            # NOTHING NEW IS VISIBLE FROM HERE. The old search gave up at this point,
            # which is why it terminated at a median 56 steps of a 400 budget: it can
            # only ever `go_to` something it has ALREADY detected, so an object that is
            # not visible from any viewpoint it happened to reach was unreachable in
            # principle, not in practice.
            #
            # Translate blind instead. Occlusion and the vertical frustum are both
            # properties of WHERE YOU STAND -- 77 of 273 objects in a 32-activity audit
            # were `outside_frustum` from the start pose alone -- so moving and
            # re-sweeping is the only operator that can recover them. We turn to an
            # unexplored heading, walk, and sweep, until the budget is gone.
            if steps + 12 > budget:
                break
            for _ in range(explored % 4 + 1):
                aci.turn_left()
                steps += 1
            for _ in range(2):
                aci.move_ahead()
                steps += 1
            # One lateral step per excursion, alternating side. Blind forward walking
            # traces a lattice of headings through the same corridors; adding a
            # perpendicular offset makes successive excursions cover different ground.
            (aci.strafe_left if explored % 2 else aci.strafe_right)()
            steps += 1
            explored += 1
            steps += sweep(aci, on_view=harvest)
            continue
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
            # PARALLAX. Occlusion is asymmetric coverage from ONE viewpoint, so a small
            # object hidden behind large furniture from where we arrived may be plainly
            # visible half a metre to the side. Turning cannot recover it; only
            # translation can.
            #
            # This used to be `turn_left, turn_left, move_ahead` -- a 3-call reversal
            # that walks BACKWARDS along the approach and re-sweeps from a point we have
            # effectively already seen. Strafing costs 1 call instead of 3 and moves
            # PERPENDICULAR to the line of sight, which is the direction that actually
            # changes what is behind what. Left then right also brackets the arrival
            # point rather than retreating from it.
            for strafe in (aci.strafe_left, aci.strafe_right, aci.strafe_right):
                strafe()
                steps += 1
                steps += sweep(aci, on_view=harvest)
        else:
            steps += 1
    object_events = {
        name: sorted(aci._det_audit.get(f"scope:{name}", ())) for name in sorted(targets)
    }
    return {
        "activity": world.activity, "source": aci.layout.source,
        "instance_source": aci.layout.instance_source or None,
        "bddl_release": aci.layout.bddl_release or None,
        "asset_release": aci.layout.asset_release or None,
        "instance_path": aci.layout.instance_path or None,
        "locatable": len(locatable), "found": len(found & locatable),
        "complete": locatable <= found, "steps": steps,
        "search_kind": "tour" if use_tour else "primitive",
        "target_objects": sorted(targets),
        "locatable_objects": sorted(locatable),
        "found_objects": sorted(found & locatable),
        "object_events": object_events,
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
    ap.add_argument("--tour", action="store_true",
                    help="privileged coverage tour instead of primitive-only search: "
                         "certifies findability, does NOT produce a policy budget")
    ap.add_argument("--tour-budget", type=int, default=8000,
                    help="privileged-tour budget used only by --certificate")
    ap.add_argument("--certificate", help="write paired primitive/tour per-object JSON; "
                                            "requires --all, --source-id and --container-id")
    ap.add_argument("--source-id", help="exact RLinf Git commit recorded in a certificate")
    ap.add_argument("--container-id", help="container image name/digest recorded in a certificate")
    ap.add_argument("--activity-source", default=ACTIVITY_SOURCE,
                    help="JSON list, or an object containing a 'solved' or "
                         "'activities' list; the exact file is hashed")
    ap.add_argument("--out", help="write {activity: oracle_steps} as JSON. This table is "
                                  "what an ORACLE-RELATIVE turn budget is computed from: "
                                  "max_turns = K * oracle_steps(activity). A flat budget "
                                  "is unfair in both directions -- the oracle needs a "
                                  "median 56 steps here but a p90 in the hundreds -- so a "
                                  "flat 40 floors the hard activities while a flat 400 "
                                  "lets the easy ones idle.")
    ap.add_argument("--obs-mode", default="fov",
                    choices=["fov", "fov_distract", "detect_scope", "detect"])
    args = ap.parse_args()

    if args.tour and args.out:
        ap.error(
            "--tour cannot be combined with --out: tour teleports using privileged "
            "scene layout, so its steps must never become a policy turn budget"
        )
    if args.certificate:
        if not args.all:
            ap.error("--certificate requires --all")
        if args.tour or args.out:
            ap.error("--certificate runs both searches itself; omit --tour and --out")
        if args.obs_mode not in SymbolicACI.DETECT_MODES:
            ap.error("--certificate requires detect_scope or detect")
        if not args.source_id or not args.container_id:
            ap.error("--certificate requires --source-id and --container-id")

    if not args.all:
        print(json.dumps(
            explore(
                args.activity,
                args.prefer,
                args.budget,
                args.obs_mode,
                use_tour=args.tour,
            ),
            indent=1,
        ))
        return

    try:
        acts, activity_list_sha256 = _activities_from_source(args.activity_source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"invalid --activity-source: {exc}")
    if args.n:
        acts = acts[:args.n]
    if args.certificate:
        from rlinf.envs.behavior.detect import native_bbox_path
        from rlinf.envs.behavior.solvability_certificate import (
            build_activity_certificate,
        )

        records, errors = [], []
        outcomes = Counter()
        for index, activity in enumerate(acts, 1):
            print(f"[{index}/{len(acts)}] {activity}", flush=True)
            try:
                primitive = explore(
                    activity, args.prefer, args.budget, args.obs_mode, use_tour=False
                )
                privileged = explore(
                    activity,
                    args.prefer,
                    args.tour_budget,
                    args.obs_mode,
                    use_tour=True,
                )
                record = build_activity_certificate(activity, primitive, privileged)
                record["instance_sha256"] = _sha256(record.get("instance_path") or "")
                records.append(record)
                outcomes.update(record["outcome_counts"])
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"activity": activity, "type": type(exc).__name__, "message": str(exc)}
                )
        payload = {
            "schema_version": 1,
            "source_id": args.source_id,
            "container_id": args.container_id,
            "argv": sys.argv,
            "obs_mode": args.obs_mode,
            "layout_prefer": args.prefer,
            "primitive_budget": args.budget,
            "tour_budget": args.tour_budget,
            "activity_source": {
                "path": args.activity_source,
                "sha256": _sha256(args.activity_source),
                "activity_list_sha256": activity_list_sha256,
            },
            "native_bbox": {
                "path": native_bbox_path(),
                "sha256": _sha256(native_bbox_path()),
            },
            "summary": {
                "requested_activities": len(acts),
                "certified_activities": len(records),
                "errored_activities": len(errors),
                "objects": sum(outcomes.values()),
                "outcome_counts": dict(sorted(outcomes.items())),
            },
            "activities": records,
            "errors": errors,
        }
        with open(args.certificate, "w") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
        print(json.dumps(payload["summary"], indent=1), flush=True)
        print(f"wrote certificate -> {args.certificate}", flush=True)
        return
    complete = partial = fail = 0
    steps_ok, worst = [], []
    oracle_steps = {}
    for a in acts:
        try:
            r = explore(a, args.prefer, args.budget, args.obs_mode, use_tour=args.tour)
        except Exception:                                          # noqa: BLE001
            fail += 1
            continue
        if r["complete"]:
            complete += 1
            steps_ok.append(r["steps"])
            oracle_steps[a] = r["steps"]
        else:
            partial += 1
            worst.append((r["found"] / max(r["locatable"], 1), a, r["missing"]))
    n = len(acts)
    print(f"activities            : {n}   (prefer={args.prefer}, budget={args.budget})")
    print(f"  ALL objects found   : {complete} = {complete / n:.1%}")
    print(f"  incomplete          : {partial}")
    print(f"  errored             : {fail}")
    if args.out:
        # Only COMPLETE runs define a budget: an incomplete search spent its whole
        # allowance without finding everything, so its step count is a censored
        # observation, not a measurement of how long the activity takes.
        with open(args.out, "w") as f:
            json.dump({"obs_mode": args.obs_mode, "search_budget": args.budget,
                       "oracle_steps": oracle_steps}, f, indent=1)
        print(f"  wrote {len(oracle_steps)} oracle step counts -> {args.out}")
    if worst:
        # Mean coverage over the activities the mode can actually attempt. The
        # all-or-nothing `ALL objects found` gate is the certificate we need before
        # running a baseline, but it hides progress: a search that goes from finding a
        # tenth of the objects to finding most of them moves this line and not that
        # one. Report both, and note the denominator is the ATTEMPTED activities, not
        # the 740 -- `detect` needs a sampled instance and most activities have none.
        fracs = sorted(f for f, _, _ in worst)
        mean = sum(fracs) / len(fracs)
        print(f"  mean coverage       : {mean:.1%} over {len(fracs)} attempted "
              f"(median {fracs[len(fracs) // 2]:.0%})")
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
