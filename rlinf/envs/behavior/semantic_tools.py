"""Semantic-oracle tool ACI layer for BEHAVIOR long-horizon planning (M1).

Wraps an OmniGibson ``BehaviorTask`` environment with high-level *symbolic*
tools that bypass the low-level continuous controller. Each tool:

  1. checks preconditions via object-state ``get_value()`` (cheap; see 0710 sim-doc),
  2. applies a symbolic effect via ``set_value()`` / base-pose set,
  3. returns a structured :class:`ToolResult` carrying a fresh symbolic observation.

Design constraints (code-doc/0710 sim-doc + proposal):
  - NEVER call ``sample_kinematics`` at rollout time. Geometric-placement tools
    (``grasp``/``place_on``/``place_inside``) are deferred to M2 (offline cached
    poses) and raise ``NotImplementedError`` here.
  - Reward/goal come from the real BDDL ``goal_status``; ``observe()`` reports
    object *states*, never the goal or the next action.
  - Observability has two modes (ablation axis, proposal §3.3):
      * ``"full"``    -- report all task-relevant (object_scope) objects.
      * ``"partial"`` -- geometric FOV+range gate (NO rendering; occlusion/LOS is
        a documented refinement, see ``_in_fov``). Named honestly: this is an
        FOV+range approximation, not segmentation-based visibility.

This module is intentionally dependency-light: it only imports OmniGibson object
states, so it can be unit-driven from a booted env without the RLinf actor stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch as th

import omnigibson as og
from omnigibson.object_states import (
    AABB,
    Cooked,
    Covered,
    Filled,
    Inside,
    OnTop,
    Open,
    ToggledOn,
)
from omnigibson.utils.object_state_utils import sample_kinematics

# Robot base within this XY distance of an object counts as "near / reachable".
NEAR_THRESHOLD_M = 1.5
# Default half-angle of the FOV cone used by the geometric partial-obs gate.
FOV_HALF_ANGLE_RAD = math.radians(60.0)
# Max range for the partial-obs gate (m).
FOV_RANGE_M = 8.0
# Object-state classes we report/act on in M1 (all cheap "flag" states).
REPORTED_STATES = {"ToggledOn": ToggledOn, "Open": Open}


@dataclass
class ToolResult:
    """Structured return for every tool call."""

    ok: bool
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""
    observation: Optional[dict] = None

    def __repr__(self) -> str:  # compact, log-friendly
        tag = "OK " if self.ok else "REJ"
        return f"<{tag} {self.tool}({self.args}) {self.reason}>"


class SemanticACI:
    """High-level semantic tool interface over one OmniGibson BehaviorTask env."""

    def __init__(self, env, near_threshold: float = NEAR_THRESHOLD_M,
                 obs_mode: str = "full"):
        # env: an omnigibson.envs.Environment (e.g. vec_env.envs[0]).
        assert obs_mode in ("full", "partial"), obs_mode
        self.env = env
        self.scene = env.scene
        self.task = env.task
        robots = getattr(env, "robots", None) or getattr(self.scene, "robots", None)
        assert robots, "no robot found on env/scene"
        self.robot = robots[0]
        self.near_threshold = near_threshold
        self.obs_mode = obs_mode
        self._pred = self.task._termination_conditions["predicate"]
        self._held = None          # name of the object currently grasped, or None
        # (movable_name, target_name, predicate) -> (pos, orn) valid placement,
        # built OFFLINE via build_pose_cache(); rollout place tools only read it.
        self._pose_cache = {}

    # ------------------------------------------------------------------ #
    # object / geometry helpers
    # ------------------------------------------------------------------ #
    def _resolve(self, name: str):
        """Resolve an object by its sim registry name (what observe() reports)."""
        return self.scene.object_registry("name", name)

    @staticmethod
    def _xy(obj) -> th.Tensor:
        pos, _ = obj.get_position_orientation()
        return pos[:2]

    def _robot_pose(self):
        return self.robot.get_position_orientation()

    def _robot_xy(self) -> th.Tensor:
        pos, _ = self._robot_pose()
        return pos[:2]

    def _robot_yaw(self) -> float:
        _, quat = self._robot_pose()  # xyzw
        x, y, z, w = [float(v) for v in quat]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _distance(self, obj) -> float:
        return float(th.norm(self._robot_xy() - self._xy(obj)))

    def _is_near(self, obj) -> bool:
        return self._distance(obj) <= self.near_threshold

    def _in_fov(self, obj) -> bool:
        """Geometric FOV+range gate for partial observability.

        NOTE: this checks bearing within a cone + max range only. True occlusion
        (line-of-sight) would need a physics raycast or the rendering-based
        ObjectsInFOVOfRobot state; that is a documented refinement for the M4
        observability study, deliberately not enabled here (render-off).
        """
        rxy, oxy = self._robot_xy(), self._xy(obj)
        d = float(th.norm(oxy - rxy))
        if d > FOV_RANGE_M:
            return False
        if d < 1e-3:
            return True
        bearing = math.atan2(float(oxy[1] - rxy[1]), float(oxy[0] - rxy[0]))
        diff = abs((bearing - self._robot_yaw() + math.pi) % (2 * math.pi) - math.pi)
        return diff <= FOV_HALF_ANGLE_RAD

    # ------------------------------------------------------------------ #
    # goal (reward signal) -- refreshed WITHOUT a physics step
    # ------------------------------------------------------------------ #
    def goal_status(self) -> dict:
        try:
            self._pred.step(self.task, self.env, None)
        except TypeError:
            self._pred._step(self.task, self.env, None)
        return self._pred.goal_status

    def is_success(self) -> bool:
        gs = self.goal_status()
        return len(gs["unsatisfied"]) == 0 and len(gs["satisfied"]) > 0

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def _obj_report(self, obj) -> dict:
        states = {}
        for label, cls in REPORTED_STATES.items():
            if cls in getattr(obj, "states", {}):
                try:
                    states[label] = bool(obj.states[cls].get_value())
                except Exception:
                    pass
        xy = self._xy(obj)
        return {
            "name": obj.name,
            "category": getattr(obj, "category", None),
            "xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
            "is_near": self._is_near(obj),
            "states": states,
        }

    def observe(self) -> dict:
        """Return a symbolic observation of task-relevant objects.

        Full mode reports every object in the BDDL object_scope; partial mode
        keeps only those passing the geometric FOV+range gate. Never leaks the
        goal.
        """
        robot_xy = self._robot_xy()
        objs = []
        for entity in self.task.object_scope.values():
            obj = getattr(entity, "wrapped_obj", None)
            if obj is None or getattr(entity, "is_system", False):
                continue
            if self.obs_mode == "partial" and not self._in_fov(obj):
                continue
            objs.append(self._obj_report(obj))
        return {
            "obs_mode": self.obs_mode,
            "robot_xy": [round(float(robot_xy[0]), 2), round(float(robot_xy[1]), 2)],
            "robot_yaw_deg": round(math.degrees(self._robot_yaw()), 1),
            "held": self._held,
            "objects": objs,
        }

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _result(self, ok, tool, args, reason):
        return ToolResult(ok=ok, tool=tool, args=args, reason=reason,
                          observation=self.observe())

    def go_to(self, name: str) -> ToolResult:
        """Symbolic navigation: teleport the robot base to a point near @name,
        facing it. Oracle locomotion (no navmesh check yet -- M2 refinement)."""
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "go_to", {"name": name}, "no such object")
        pos, _ = self._robot_pose()
        rz = float(pos[2])
        oxy = self._xy(obj)
        rxy = self._robot_xy()
        direction = rxy - oxy
        n = float(th.norm(direction))
        direction = direction / n if n > 1e-3 else th.tensor([1.0, 0.0])
        stop = self.near_threshold * 0.6
        target_xy = oxy + direction * stop
        yaw = math.atan2(float(oxy[1] - target_xy[1]), float(oxy[0] - target_xy[0]))
        quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        target_pos = th.tensor([float(target_xy[0]), float(target_xy[1]), rz])
        self.robot.set_position_orientation(position=target_pos, orientation=quat)
        return self._result(True, "go_to", {"name": name},
                            f"now {self._distance(obj):.2f}m from {name}")

    def _set_flag(self, tool, name, state_cls, value, need_closed=False):
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, tool, {"name": name}, "no such object")
        if state_cls not in getattr(obj, "states", {}):
            return self._result(False, tool, {"name": name},
                                f"{name} has no {state_cls.__name__} state")
        if not self._is_near(obj):
            return self._result(False, tool, {"name": name},
                                f"precondition failed: not near {name} "
                                f"({self._distance(obj):.2f}m > {self.near_threshold}m)")
        # Set the state, settle one step so the actuated joint registers, and
        # VERIFY the change actually took. Open._set_value samples partial joint
        # positions and can miss (esp. multi-door objects like cars), so on a miss
        # retry fully-open/closed. ok reflects the REAL achieved state, not intent.
        st = obj.states[state_cls]
        st.set_value(value)
        og.sim.step_physics()
        if bool(st.get_value()) != value:
            try:
                st.set_value(value, fully=True)  # Open supports fully=; others don't
            except TypeError:
                st.set_value(value)
            og.sim.step_physics()
        got = bool(st.get_value())
        return self._result(got == value, tool, {"name": name},
                            f"{state_cls.__name__}={got}")

    def toggle_on(self, name: str) -> ToolResult:
        return self._set_flag("toggle_on", name, ToggledOn, True)

    def toggle_off(self, name: str) -> ToolResult:
        return self._set_flag("toggle_off", name, ToggledOn, False)

    def open(self, name: str) -> ToolResult:
        return self._set_flag("open", name, Open, True)

    def close(self, name: str) -> ToolResult:
        return self._set_flag("close", name, Open, False)

    # ---- manipulation (cached-pose; NEVER sample_kinematics at rollout) ----
    # Gate 4-B (0709 §8): offline sample of a feasible placement ~3.4s, a failed
    # one 20-78s; restoring a cached pose is ~12ms (set_pose + one step_physics to
    # propagate the kinematic state). So placement poses are sampled OFFLINE via
    # build_pose_cache(); rollout place tools only read the cache.

    _PRED = {"OnTop": OnTop, "Inside": Inside}
    _SAMPLE_PRED = {"OnTop": "onTop", "Inside": "inside"}
    _SETTLE_STEPS = 10

    def build_pose_cache(self, pairs) -> dict:
        """OFFLINE: for each (movable_name, target_name, predicate), find a valid
        pose and cache it. Call before rollout, NOT during it.

        Hybrid, because no single strategy covers all container geometries:
          1) sample_kinematics(use_last_ditch_effort=True) -- uses the target's
             real interior volume, so it handles elevated/shelved containers
             (fridge, cabinet) the drop-in can't;
          2) fallback ring-spread drop-in on the AABB floor/top -- handles tight
             multi-object bins (3 cans in a small trash can) that the sampler
             stacks and overflows.
        Pairs processed in order; per-target counter spreads the fallback ring.
        Returns {key: bool success}."""
        out = {}
        placed = {}
        for movable, target, pred in pairs:
            m, t = self._resolve(movable), self._resolve(target)
            cls = self._PRED[pred]
            k = placed.get(target, 0)
            placed[target] = k + 1
            ok = False
            if m is not None and t is not None and cls in getattr(m, "states", {}):
                # 1) real-interior sampler (best for roomy/elevated containers)
                try:
                    sample_kinematics(self._SAMPLE_PRED[pred], m, t,
                                      use_last_ditch_effort=True)
                    ok = bool(m.states[cls].get_value(t))
                except Exception:
                    ok = False
                # 2) fallback: ring-spread drop-in (tight multi-object bins).
                #    sample_kinematics is rejection-based and flaky in deep
                #    containers (a fridge shelf can miss on the 2nd item), so try
                #    several ring slots before giving up on this pair.
                if not ok:
                    for kk in (k, k + 2, k + 4, k + 1):
                        if self._seat_dropin(m, t, cls, pred, kk):
                            ok = True
                            break
                if ok:
                    pos, orn = m.get_position_orientation()
                    self._pose_cache[(movable, target, pred)] = (pos.clone(), orn.clone())
            out[(movable, target, pred)] = ok
        return out

    def _seat_dropin(self, m, t, cls, pred, k):
        """Seat @m at a ring position on @t's AABB floor (Inside) / top (OnTop)
        and settle. @k spreads successive objects around a ring."""
        lo, hi = t.states[AABB].get_value()
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        radius = 0.28 * min(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        ang = 2 * math.pi * k / 6.0
        dx = 0.0 if k == 0 else radius * math.cos(ang)
        dy = 0.0 if k == 0 else radius * math.sin(ang)
        mlo, mhi = m.states[AABB].get_value()
        half_h = float(mhi[2] - mlo[2]) / 2
        base_z = float(lo[2]) if pred == "Inside" else float(hi[2])
        m.set_position_orientation(position=th.tensor([cx + dx, cy + dy, base_z + half_h + 0.03]))
        for _ in range(self._SETTLE_STEPS):
            og.sim.step_physics()
        return bool(m.states[cls].get_value(t))

    def grasp(self, name: str) -> ToolResult:
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "grasp", {"name": name}, "no such object")
        if self._held is not None:
            return self._result(False, "grasp", {"name": name},
                                f"precondition failed: already holding {self._held}")
        if not self._is_near(obj):
            return self._result(False, "grasp", {"name": name},
                                f"precondition failed: not near {name} "
                                f"({self._distance(obj):.2f}m > {self.near_threshold}m)")
        # oracle grasp: lift the object to a carry pose above the robot base so it
        # is no longer resting on its support (kept physically real, not a flag).
        rpos, _ = self._robot_pose()
        carry = th.tensor([float(rpos[0]), float(rpos[1]), float(rpos[2]) + 1.0])
        obj.set_position_orientation(position=carry)
        og.sim.step_physics()
        self._held = name
        return self._result(True, "grasp", {"name": name}, f"holding {name}")

    def release(self, name: str = None) -> ToolResult:
        if self._held is None:
            return self._result(False, "release", {}, "precondition failed: not holding anything")
        held = self._held
        self._held = None
        return self._result(True, "release", {"name": held}, f"released {held}")

    def _place(self, tool, name, target, pred):
        if self._held != name:
            return self._result(False, tool, {"name": name, "target": target},
                                f"precondition failed: not holding {name} (held={self._held})")
        t = self._resolve(target)
        if t is None:
            return self._result(False, tool, {"name": name, "target": target},
                                f"no such target {target}")
        if not self._is_near(t):
            return self._result(False, tool, {"name": name, "target": target},
                                f"precondition failed: not near {target} "
                                f"({self._distance(t):.2f}m > {self.near_threshold}m)")
        key = (name, target, pred)
        if key not in self._pose_cache:
            return self._result(False, tool, {"name": name, "target": target},
                                f"no cached pose for {key}; run build_pose_cache offline "
                                f"(rollout must not call sample_kinematics)")
        pos, orn = self._pose_cache[key]
        obj = self._resolve(name)
        obj.set_position_orientation(position=pos, orientation=orn)
        og.sim.step_physics()  # propagate kinematics so the predicate reads correctly
        ok = bool(obj.states[self._PRED[pred]].get_value(t))
        if ok:
            self._held = None
        return self._result(ok, tool, {"name": name, "target": target},
                            f"{pred}({name},{target})={ok}")

    def place_on(self, name: str, surface: str) -> ToolResult:
        return self._place("place_on", name, surface, "OnTop")

    def place_inside(self, name: str, container: str) -> ToolResult:
        return self._place("place_inside", name, container, "Inside")

    # ---- transformation tools (class B1: state-based; real state/particle
    # ---- changes via the genuine object-state setters, not faked flags) ----
    def _system(self, system_name: str):
        return self.scene.get_system(system_name)

    def cook(self, name: str) -> ToolResult:
        """Cook @name (raises MaxTemperature past the cook threshold)."""
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "cook", {"name": name}, "no such object")
        if Cooked not in getattr(obj, "states", {}):
            return self._result(False, "cook", {"name": name}, f"{name} not cookable")
        if not self._is_near(obj):
            return self._result(False, "cook", {"name": name},
                                f"precondition failed: not near {name}")
        obj.states[Cooked].set_value(True)
        og.sim.step_physics()
        return self._result(bool(obj.states[Cooked].get_value()), "cook", {"name": name},
                            f"Cooked={obj.states[Cooked].get_value()}")

    def _set_covered(self, tool, name, system_name, value):
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, tool, {"name": name}, "no such object")
        if Covered not in getattr(obj, "states", {}):
            return self._result(False, tool, {"name": name}, f"{name} has no Covered state")
        if not self._is_near(obj):
            return self._result(False, tool, {"name": name, "system": system_name},
                                f"precondition failed: not near {name}")
        try:
            system = self._system(system_name)
        except Exception as ex:
            return self._result(False, tool, {"name": name, "system": system_name},
                                f"no such system {system_name} ({type(ex).__name__})")
        obj.states[Covered].set_value(system, value)
        og.sim.step_physics()
        got = bool(obj.states[Covered].get_value(system))
        return self._result(got == value, tool, {"name": name, "system": system_name},
                            f"Covered({name},{system_name})={got}")

    def spray(self, name: str, system_name: str) -> ToolResult:
        """Cover @name with the @system_name substance (e.g. pesticide)."""
        return self._set_covered("spray", name, system_name, True)

    def uncover(self, name: str, system_name: str) -> ToolResult:
        """Remove the @system_name substance from @name (e.g. clean mud)."""
        return self._set_covered("uncover", name, system_name, False)

    def fill(self, name: str, system_name: str) -> ToolResult:
        """Fill container @name with the @system_name physical particle system."""
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "fill", {"name": name}, "no such object")
        if Filled not in getattr(obj, "states", {}):
            return self._result(False, "fill", {"name": name}, f"{name} not fillable")
        if not self._is_near(obj):
            return self._result(False, "fill", {"name": name, "system": system_name},
                                f"precondition failed: not near {name}")
        try:
            system = self._system(system_name)
        except Exception as ex:
            return self._result(False, "fill", {"name": name, "system": system_name},
                                f"no such system {system_name} ({type(ex).__name__})")
        obj.states[Filled].set_value(system, True)
        og.sim.step_physics()
        return self._result(bool(obj.states[Filled].get_value(system)), "fill",
                            {"name": name, "system": system_name}, "filled")

    # ---- class B2: slice / dice (product-creating transforms) --------------
    # These invoke the REAL OmniGibson transition rules (SlicingRule/DicingRule),
    # so the spawned parts/particles are genuine sim entities. The sim's add-object
    # and system-init callbacks auto-rebind the BDDL "future" scope entries, which
    # is exactly what flips the `real(...)` goal atom (see BehaviorTask.
    # _update_bddl_scope_from_added_obj / _from_system_init). Nothing is faked: we
    # only bypass the low-level "bring a slicer into contact" control -- the same
    # abstraction grasp/place make -- driving the identical downstream effect.
    @staticmethod
    def _abilities(obj) -> set:
        return set(getattr(obj, "abilities", {}) or {})

    def _do_transition(self, rule_cls, filter_key, obj):
        """Execute @rule_cls's transition on a single @obj and settle. Returns the
        TransitionResults (list of added ObjectAttrs / removed objs)."""
        rule = rule_cls(self.scene)
        results = rule.transition({filter_key: [obj]})
        api = self.scene.transition_rule_api
        api.execute_transition(added_obj_attrs=results.add, removed_objs=results.remove)
        # One full sim step applies the added-object init callbacks (which propagate
        # cooked/saturated onto the parts) and lets scope rebind / systems init.
        og.sim.step()
        return results

    def slice(self, name: str) -> ToolResult:
        """Slice @name into its annotated object parts (e.g. log -> 2x half__log).

        Parts spawn as real DatasetObjects; the future BDDL scope entries bind to
        them so ``real(half__...)`` becomes satisfied."""
        from omnigibson.transition_rules import SlicingRule
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "slice", {"name": name}, "no such object")
        if "sliceable" not in self._abilities(obj):
            return self._result(False, "slice", {"name": name},
                                f"precondition failed: {name} is not sliceable")
        if not self._is_near(obj):
            return self._result(False, "slice", {"name": name},
                                f"precondition failed: not near {name} "
                                f"({self._distance(obj):.2f}m > {self.near_threshold}m)")
        results = self._do_transition(SlicingRule, "sliceable", obj)
        n = len(results.add)
        return self._result(n > 0 and self._resolve(name) is None, "slice",
                            {"name": name}, f"sliced into {n} part(s)")

    def dice(self, name: str) -> ToolResult:
        """Mince @name into its ``diced__<category>`` particle system.

        Only ``diceable`` objects dice directly. A whole vegetable is typically
        ``sliceable`` but not ``diceable`` (BEHAVIOR models "chop an onion" as
        slice-into-halves THEN dice-the-halves), so for a sliceable-only object we
        run the real two-stage chain: SlicingRule -> parts, then DicingRule on each
        diceable part. Both stages are genuine transitions; the resulting particles
        are real, which is what flips ``real(diced__...)`` / ``contains(...)``."""
        from omnigibson.transition_rules import DicingRule, SlicingRule
        obj = self._resolve(name)
        if obj is None:
            return self._result(False, "dice", {"name": name}, "no such object")
        abil = self._abilities(obj)
        if not self._is_near(obj):
            return self._result(False, "dice", {"name": name},
                                f"precondition failed: not near {name} "
                                f"({self._distance(obj):.2f}m > {self.near_threshold}m)")
        if "diceable" in abil:
            self._do_transition(DicingRule, "diceable", obj)
            return self._result(self._resolve(name) is None, "dice",
                                {"name": name}, "diced into particle system")
        if "sliceable" in abil:
            parts = [a.obj for a in self._do_transition(SlicingRule, "sliceable", obj).add]
            diced = 0
            for part in parts:
                if "diceable" in self._abilities(part):
                    self._do_transition(DicingRule, "diceable", part)
                    diced += 1
            return self._result(diced > 0, "dice", {"name": name},
                                f"sliced into {len(parts)} part(s), diced {diced}")
        return self._result(False, "dice", {"name": name},
                            f"precondition failed: {name} is neither diceable nor sliceable")

    def end_task(self) -> ToolResult:
        gs = self.goal_status()
        return ToolResult(ok=self.is_success(), tool="end_task", args={},
                          reason=f"goal_status={gs}", observation=self.observe())
