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

from omnigibson.object_states import Open, ToggledOn

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
        obj.states[state_cls].set_value(value)
        return self._result(True, tool, {"name": name},
                            f"{state_cls.__name__}={value}")

    def toggle_on(self, name: str) -> ToolResult:
        return self._set_flag("toggle_on", name, ToggledOn, True)

    def toggle_off(self, name: str) -> ToolResult:
        return self._set_flag("toggle_off", name, ToggledOn, False)

    def open(self, name: str) -> ToolResult:
        return self._set_flag("open", name, Open, True)

    def close(self, name: str) -> ToolResult:
        return self._set_flag("close", name, Open, False)

    # ---- deferred to M2 (need offline cached poses; must not call
    # ---- sample_kinematics at rollout -- see 0710 sim-doc §6 / Gate 4-B) ----
    def grasp(self, name: str) -> ToolResult:
        raise NotImplementedError("grasp deferred to M2 (cached-pose manipulation)")

    def place_on(self, name: str, surface: str) -> ToolResult:
        raise NotImplementedError("place_on deferred to M2 (cached-pose manipulation)")

    def place_inside(self, name: str, container: str) -> ToolResult:
        raise NotImplementedError("place_inside deferred to M2 (cached-pose manipulation)")

    def end_task(self) -> ToolResult:
        gs = self.goal_status()
        return ToolResult(ok=self.is_success(), tool="end_task", args={},
                          reason=f"goal_status={gs}", observation=self.observe())
