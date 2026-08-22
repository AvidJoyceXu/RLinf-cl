"""Metric layout for BEHAVIOR-TextWorld: where things are, in metres.

The symbolic layer has no coordinates by construction, so a viewpoint layer needs a
source of them. There are two, and **which one is in use is recorded on every layout**
so a caller can never accidentally report generated geometry as BEHAVIOR's:

* ``source="sampled"`` -- real poses, read from a `*-tro_state.json` produced by
  BEHAVIOR's own task-instance sampler (see
  `code-reading/BEHAVIOR task instance sampling & validation.md`). Sim-free: the file
  is on disk, no OmniGibson boot.
* ``source="generated"`` -- coordinates this module invents from the BDDL room and
  containment structure, deterministically per activity.

**Generated layouts are legitimate and must never be described as BEHAVIOR's
geometry.** TextWorld's own worlds are procedurally generated -- that is what the
framework is for -- so a generated-but-consistent world is squarely in the genre. What
would be dishonest is presenting one as measured. Hence `Layout.source`, and hence the
refusal in `viewpoint.py` to report distances as if they meant anything physical when
the source is generated.

WHY A GENERATED LAYOUT IS STILL THE DEFAULT. Real poses cover a small fraction of the
benchmark: as of 2026-08-21, 117 activities have an instance on disk and 71 of the 740
verified-solvable activities are resolvable here. The research question the viewpoint
layer originally addressed -- whether explicit camera control separates a prompted
baseline from the SFT checkpoint -- is answerable on any consistent layout. Modes that
claim real geometry (`detect`, `detect_scope`, `fov_distract`) override this default and
refuse generated layouts.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from rlinf.envs.behavior.instance_compatibility import scope_mismatch
from rlinf.envs.behavior.instance_sources import (
    InstanceSource,
    assert_detect_eligible,
    selected_layout_sources,
)
from rlinf.envs.behavior.symbolic_world import SymbolicWorld, properties_of

# Generated-layout geometry, in metres. Chosen to look like a house rather than to
# match one: rooms are 6x5, objects orbit their support at 0.6 m.
ROOM_W, ROOM_H = 6.0, 5.0
ROOM_GAP = 2.0
ORBIT_R = 0.6
SURFACE_H = 0.75  # nominal height of a support surface
FLOOR_H = 0.05


@dataclass
class Layout:
    """Positions in metres, plus an honest label for where they came from."""

    activity: str
    source: str  # "sampled" | "generated"
    xyz: dict = field(default_factory=dict)  # scope name -> (x, y, z)
    rooms: dict = field(default_factory=dict)  # room name -> (cx, cy)
    robot_xy: tuple = (0.0, 0.0)
    robot_yaw: float = 0.0
    scene: str = ""  # scene_model, for furniture
    instance_path: str = ""  # the tro_state file, for models
    instance_source: str = ""  # versioned source name from instance_sources.py
    bddl_release: str = ""
    asset_release: str = ""

    @property
    def is_real(self) -> bool:
        return self.source == "sampled"

    def pos(self, name: str) -> Optional[tuple]:
        return self.xyz.get(name)


# --------------------------------------------------------------------------- #
# sampled poses
# --------------------------------------------------------------------------- #
def _find_instance(
    activity: str,
    expected_scope=None,
    ignored_scope=(),
    *,
    source_selection: str | None = None,
    require_detect_models: bool = False,
) -> Optional[tuple[str, InstanceSource]]:
    """Preferred compatible `*-tro_state.json` for @activity, or None.

    The selected source registry is ordered by PREFERENCE, and the first source wins
    outright; mtime
    only breaks ties within a root. Official challenge instances are preferred because
    they passed the published pipeline and, in the 2026 set, provide hundreds of
    within-activity variants. Locally generated instances remain the fallback.

    An earlier rationale claimed this ordering was required because generated files
    lack `robot_poses`. A paired test on 22 activities found no material detect-coverage
    difference between the sampled pose and `_from_instance`'s centroid fallback. Keep
    the provenance preference, but do not describe generated instances as second-class
    for detect on that basis.

    When ``expected_scope`` is supplied, partial overlap is never accepted. This is a
    provenance gate, not a pose-coverage test: BDDL and the template must declare the
    same objects before geometry from that template can be attributed to the task.
    """
    for source in selected_layout_sources(source_selection):
        if require_detect_models:
            try:
                assert_detect_eligible(source)
            except ValueError:
                continue
        root = source.root
        hits = glob.glob(
            os.path.join(
                root, "*", "json", f"*_task_{activity}_instances", "*tro_state.json"
            )
        )
        hits += glob.glob(os.path.join(root, activity, "*tro_state.json"))
        for path in sorted(hits, key=os.path.getmtime, reverse=True):
            if (
                expected_scope is None
                or scope_mismatch(expected_scope, path, ignored_scope=ignored_scope)
                is None
            ):
                return path, source
    return None


def _scene_of_path(path: str) -> str:
    """scene_model for an instance file, from its directory or its name.

    Official instances live under `.../scenes/<scene_model>/json/...`. Instances our
    own generator wrote live under `<out_dir>/<activity>/`, where the only record of
    the scene is the filename prefix:

        house_double_floor_lower_task_assembling_gift_baskets_0_0_template-tro_state.json
        |________ scene_model ________|

    Getting this wrong is not cosmetic. `detect` mixes task-object poses with scene
    furniture, and with no scene there is no furniture -- the mode degrades silently
    to reporting only the task objects, which is the oracle it exists to remove.
    """
    parts = path.split(os.sep)
    if "scenes" in parts:
        return parts[parts.index("scenes") + 1]
    base = os.path.basename(path)
    if "_task_" in base:
        return base.split("_task_")[0]
    return ""


def _from_instance(activity: str, path: str, source: InstanceSource) -> Layout:
    raw = json.load(open(path))
    lay = Layout(
        activity=activity,
        source="sampled",
        scene=_scene_of_path(path),
        instance_path=path,
        instance_source=source.name,
        bddl_release=source.bddl_release,
        asset_release=source.asset_release,
    )
    for name, entry in raw.items():
        if name == "robot_poses":
            # Present in official instances, absent from ours -- the generator sets
            # use_presampled_robot_pose=False and nothing writes the metadata key.
            poses = entry.get("R1Pro") or next(iter(entry.values()), None)
            if poses:
                p = poses[0]["position"]
                lay.robot_xy = (float(p[0]), float(p[1]))
            continue
        rl = entry.get("root_link")
        if not rl:
            # Substances have no pose by construction; they are located by host.
            continue
        if name.startswith("agent.n.01"):
            # The agent is written into the instance at its STAGING pose -- the
            # off-scene parking spot the sampler uses while it places objects
            # ((-50,-50,-50) for instances from multiply_b1k_tasks). It is not task
            # state and it is not where the robot starts. Keeping it here corrupted
            # the centroid below: averaging six objects around the origin with one
            # entry at -50 put the robot ~10 m from the nearest object, outside
            # detect's range, so `observe()` reported NOTHING. See
            # `code-reading ... sampling & validation` §8.3, which found the same
            # staging pose written by our own generator and fixed it there.
            continue
        x, y, z = (float(v) for v in rl["pos"])
        lay.xyz[name] = (x, y, z)
    if lay.xyz and lay.robot_xy == (0.0, 0.0):
        # No robot_poses in the file: start the robot at the centroid of the task
        # objects rather than at the origin, which would usually be outside the house.
        # This is a fallback, not a sampled pose -- upstream's stage 4
        # (`sample_robot_pose.py`) is what writes a real one.
        xs = [p[0] for p in lay.xyz.values()]
        ys = [p[1] for p in lay.xyz.values()]
        lay.robot_xy = (sum(xs) / len(xs), sum(ys) / len(ys))
    return lay


# --------------------------------------------------------------------------- #
# generated layout
# --------------------------------------------------------------------------- #
def _rng(activity: str) -> float:
    """Deterministic [0,1) from the activity name. Same activity, same world, always."""
    h = hashlib.sha256(activity.encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def _anchors(world: SymbolicWorld) -> list:
    """Objects that get a fixed spot: room-assigned things and scene furniture."""
    return sorted(
        n
        for n in world.scope_names
        if n != world.agent and (n in world.rooms or "sceneObject" in properties_of(n))
    )


@functools.lru_cache(maxsize=8)
def _scene_room_centroids(scene_model: str) -> dict:
    """room type -> (x, y) centroid of that room's objects in the real scene."""
    if not scene_model:
        return {}
    hits = glob.glob(
        os.path.join(
            "/data/behavior-data/behavior-1k-assets/scenes",
            scene_model,
            "json",
            f"{scene_model}_best.json",
        )
    )
    if not hits:
        return {}
    d = json.load(open(hits[0]))
    reg = d.get("state", {}).get("registry", {}).get("object_registry", {})
    acc: dict = defaultdict(list)
    for name, info in d.get("objects_info", {}).get("init_info", {}).items():
        pos = (reg.get(name) or {}).get("root_link", {}).get("pos")
        if not pos:
            continue
        for r in info.get("args", {}).get("in_rooms") or []:
            acc[str(r).rsplit("_", 1)[0]].append((float(pos[0]), float(pos[1])))
    return {
        r: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v))
        for r, v in acc.items()
        if v
    }


def _generated(activity: str, world: SymbolicWorld) -> Layout:
    """INVENTED positions. Not BEHAVIOR's geometry and never to be reported as such.

    Reachable only by asking for it explicitly (`prefer="generated"`); every default in
    this codebase is `sampled`. Kept for one narrow purpose -- a layout for an activity
    that has no sampled instance, where nothing reads coordinates as physical fact.
    It fails two things a real layout satisfies: the positions are in an invented frame
    rather than the scene's, and the BDDL initial conditions do not hold geometrically,
    so `inside(can, ashcan)` can be true in the literal set while the can sits three
    metres away. `obs_mode=detect` therefore refuses it outright.
    """
    # Distractors come from a REAL scene even when the task objects do not:
    # `detect` mode needs furniture, and inventing furniture would be a second
    # fabrication on top of the generated positions.
    lay = Layout(activity=activity, source="generated", scene="house_single_floor")
    jitter = _rng(activity)

    # Rooms are placed at the REAL scene's room centroids when the scene has a room of
    # that type, and on a fallback row otherwise. Without this the generated task
    # objects and the scene furniture live in two unrelated coordinate frames, and
    # `detect` mode reports a robot standing among furniture it is nowhere near.
    rooms = sorted(set(world.rooms.values())) or ["room"]
    real = _scene_room_centroids(lay.scene)
    for i, room in enumerate(rooms):
        lay.rooms[room] = real.get(room) or (
            i * (ROOM_W + ROOM_GAP) + ROOM_W / 2,
            ROOM_H / 2,
        )

    # Anchors spread inside their room on a coarse grid, so two pieces of furniture
    # never coincide and the agent has somewhere to walk between them.
    per_room = defaultdict(list)
    for name in _anchors(world):
        per_room[world.room_of(name) or rooms[0]].append(name)
    for room, names in per_room.items():
        cx, cy = lay.rooms.get(room, lay.rooms[rooms[0]])
        cols = max(1, math.ceil(math.sqrt(len(names))))
        for k, name in enumerate(names):
            col, row = k % cols, k // cols
            step_x = ROOM_W / (cols + 1)
            step_y = ROOM_H / (cols + 1)
            lay.xyz[name] = (
                cx - ROOM_W / 2 + step_x * (col + 1) + 0.1 * jitter,
                cy - ROOM_H / 2 + step_y * (row + 1) + 0.1 * jitter,
                FLOOR_H,
            )

    # Everything else orbits whatever supports it, resolved outward one level at a
    # time so a can inside a box that is on a table ends up near the table.
    children = defaultdict(list)
    for name in sorted(world.scope_names):
        if name == world.agent or name in lay.xyz:
            continue
        sup = world.support_of(name)
        if sup:
            children[sup[1]].append(name)
    frontier, depth = list(lay.xyz), 0
    while frontier and depth < 6:
        nxt = []
        for parent in frontier:
            px, py, pz = lay.xyz[parent]
            kids = [k for k in children.get(parent, []) if k not in lay.xyz]
            # A thing resting on the floor sits at floor height; a thing resting on
            # furniture sits a surface-height above it. Getting this wrong is not
            # cosmetic here -- `look_down` gates on height, so a can on the floor
            # reported at table height would be visible from the wrong pitch.
            on_floor = "floor" in parent or "lawn" in parent
            for j, name in enumerate(kids):
                ang = (j / max(len(kids), 1)) * 2 * math.pi + jitter * math.pi
                r = ORBIT_R * (1 + 0.3 * depth)
                lay.xyz[name] = (
                    px + r * math.cos(ang),
                    py + r * math.sin(ang),
                    pz if on_floor else pz + SURFACE_H,
                )
                nxt.append(name)
        frontier, depth = nxt, depth + 1

    # Anything unplaced (no support, no room -- substances mostly) is left out; it has
    # no location in BDDL either, and inventing one would be the fake this file avoids.
    first_room = lay.rooms[rooms[0]]
    lay.robot_xy = (first_room[0], first_room[1] - ROOM_H / 2 + 0.5)
    return lay


# --------------------------------------------------------------------------- #
def build_layout(
    activity: str,
    world: Optional[SymbolicWorld] = None,
    prefer: str = "sampled",
    *,
    source_selection: str | None = None,
    require_detect_models: bool = False,
) -> Layout:
    """Layout for @activity. Uses real sampled poses when available unless told not to.

    @prefer "sampled" -> real poses if an instance exists, else generated.
            "generated" -> always generated (for a matched comparison across the
            whole benchmark, where mixing sources would confound the condition).
    """
    world = world or SymbolicWorld(activity)
    if prefer == "sampled":
        ignored_scope = [
            name
            for name in world.scope_names
            if not world.is_real(name) or "sceneObject" in properties_of(name)
        ]
        found = _find_instance(
            activity,
            world.scope_names,
            ignored_scope=ignored_scope,
            source_selection=source_selection,
            require_detect_models=require_detect_models,
        )
        if found:
            path, source = found
            lay = _from_instance(activity, path, source)
            if lay.xyz:
                return lay
    return _generated(activity, world)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("activity", nargs="?")
    ap.add_argument(
        "--audit",
        action="store_true",
        help="build a layout for every verified-solvable activity",
    )
    ap.add_argument("--prefer", default="sampled", choices=["sampled", "generated"])
    args = ap.parse_args()

    if args.audit:
        acts = json.load(open("/data/behavior-data/tw_verified.json"))["solved"]
        n_sampled = n_gen = fail = 0
        unplaced = 0
        for a in acts:
            try:
                w = SymbolicWorld(a)
                lay = build_layout(a, w, prefer=args.prefer)
                n_sampled += lay.is_real
                n_gen += not lay.is_real
                unplaced += sum(
                    1
                    for n in w.scope_names
                    if n != w.agent and w.is_real(n) and n not in lay.xyz
                )
            except Exception:  # noqa: BLE001
                fail += 1
        print(f"activities        : {len(acts)}")
        print(f"  sampled poses   : {n_sampled}")
        print(f"  generated       : {n_gen}")
        print(f"  failed          : {fail}")
        print(f"  unplaced objects: {unplaced} (substances with no host)")
        return

    lay = build_layout(args.activity, prefer=args.prefer)
    print(f"{args.activity}  source={lay.source}  robot={lay.robot_xy}")
    for name, p in sorted(lay.xyz.items()):
        print(f"  {name:<44} ({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:5.2f})")


if __name__ == "__main__":
    _main()
