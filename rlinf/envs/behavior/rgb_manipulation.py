"""Opt-in physical manipulation evidence for RGB debug trajectories.

This module is deliberately outside the TextWorld dependency graph.  It augments
the semantic OmniGibson tools only when ``BehaviorEnv`` is explicitly constructed
with RGB manipulation evidence enabled; the default semantic and text-only paths
keep their existing state-update cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MUTATION_TOOLS = frozenset({"open", "close", "grasp", "place_inside", "place_on"})


def _values(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _pose_record(obj) -> dict | None:
    if obj is None:
        return None
    pos, orn = obj.get_position_orientation()
    return {"xyz": _values(pos), "orientation_xyzw": _values(orn)}


def _truth_name(value: Any) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


def _quat_rotation_xyzw(quaternion):
    x, y, z, w = (float(item) for item in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def project_world_aabb(
    aabb: list[list[float]] | None,
    *,
    camera_xyz,
    camera_orientation_xyzw,
    intrinsic,
    width: int,
    height: int,
    near: float = 0.03,
) -> tuple[int, int, int, int] | None:
    """Project a world AABB through a Kit camera (local -Z forward, +Y up)."""
    if not aabb or len(aabb) != 2:
        return None
    low, high = aabb
    rotation = _quat_rotation_xyzw(camera_orientation_xyzw)
    fx, fy = float(intrinsic[0][0]), float(intrinsic[1][1])
    cx, cy = float(intrinsic[0][2]), float(intrinsic[1][2])
    xs, ys, depths = [], [], []
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                delta = (
                    float(x) - float(camera_xyz[0]),
                    float(y) - float(camera_xyz[1]),
                    float(z) - float(camera_xyz[2]),
                )
                # Camera pose maps local to world, so world to local is R^T.
                local = tuple(
                    sum(rotation[row][col] * delta[row] for row in range(3))
                    for col in range(3)
                )
                depth = -local[2]
                depths.append(depth)
                clipped = max(depth, near)
                xs.append(cx + fx * local[0] / clipped)
                ys.append(cy - fy * local[1] / clipped)
    if max(depths, default=-1.0) <= near:
        return None
    x0, x1 = max(0, math.floor(min(xs))), min(width, math.ceil(max(xs)))
    y0, y1 = max(0, math.floor(min(ys))), min(height, math.ceil(max(ys)))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def union_roi(*boxes) -> tuple[int, int, int, int] | None:
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    return (
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    )


def pixel_change_metrics(reference, candidate, roi, threshold: int = 12) -> dict:
    """Compute simple, reproducible RGB change metrics inside ``roi``."""
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError(
            f"frame shape mismatch: {reference.shape!r} != {candidate.shape!r}"
        )
    x0, y0, x1, y1 = roi
    lhs = np.asarray(reference[y0:y1, x0:x1, :3], dtype=np.int16)
    rhs = np.asarray(candidate[y0:y1, x0:x1, :3], dtype=np.int16)
    if lhs.size == 0:
        return {"mean_abs_diff": 0.0, "changed_pixel_fraction": 0.0, "pixels": 0}
    delta = np.abs(lhs - rhs)
    return {
        "mean_abs_diff": round(float(delta.mean()), 6),
        "changed_pixel_fraction": round(
            float((delta.max(axis=-1) >= int(threshold)).mean()), 6
        ),
        "pixels": int(lhs.shape[0] * lhs.shape[1]),
        "threshold": int(threshold),
    }


def joint_position_delta(before: dict | None, after: dict | None) -> dict:
    """Summarize motion of matching articulation joints without simulator imports."""
    before = before or {}
    after = after or {}
    by_joint = {}
    for name in sorted(set(before) & set(after)):
        lhs = before[name]
        rhs = after[name]
        if not isinstance(lhs, list):
            lhs = [lhs]
        if not isinstance(rhs, list):
            rhs = [rhs]
        if len(lhs) != len(rhs) or not lhs:
            continue
        by_joint[name] = max(abs(float(a) - float(b)) for a, b in zip(lhs, rhs))
    max_delta = max(by_joint.values(), default=0.0)
    return {
        "matching_joints": len(by_joint),
        "changed_joints": sum(delta > 1e-5 for delta in by_joint.values()),
        "max_abs_delta": round(float(max_delta), 8),
        "by_joint": {name: round(float(delta), 8) for name, delta in by_joint.items()},
    }


@dataclass
class ManipulationContext:
    tool: str
    arguments: dict
    before: dict
    released_for_place: bool = False
    released_object: Any = None
    robot_position: Any = None
    robot_orientation: Any = None
    robot_joint_positions: Any = None


class RGBManipulationEvidence:
    """Synchronize semantic manipulation with R1 assisted-grasp state."""

    def __init__(self, env, aci):
        self.env = env
        self.aci = aci
        self.robot = aci.robot
        self.arm = self.robot.default_arm

    @staticmethod
    def handles(tool: str) -> bool:
        return tool in MUTATION_TOOLS

    def _is_grasping(self, obj=None) -> bool:
        return (
            _truth_name(self.robot.is_grasping(self.arm, candidate_obj=obj)) == "TRUE"
        )

    def _joint_positions(self, obj) -> dict:
        out = {}
        for name, joint in getattr(obj, "joints", {}).items():
            try:
                state = joint.get_state()
                position = state[0] if isinstance(state, tuple) else state
                out[str(name)] = _values(position)
            except Exception:
                continue
        return out

    def _snapshot(self, tool: str, arguments: dict) -> dict:
        from omnigibson.object_states import AABB, Inside, OnTop, Open

        name = arguments.get("name")
        target_name = arguments.get("container") or arguments.get("surface")
        obj = self.aci._resolve(name) if name else None
        target = self.aci._resolve(target_name) if target_name else None
        record = {
            "held": self.aci._held,
            "robot_grasping": self._is_grasping(),
            "robot_grasping_object": self._is_grasping(obj)
            if obj is not None
            else False,
            "robot_pose": _pose_record(self.robot),
            "object_name": name,
            "object_pose": _pose_record(obj),
            "target_name": target_name,
            "target_pose": _pose_record(target),
            "eef_pose": _pose_record(self.robot.eef_links[self.arm]),
        }
        if obj is not None:
            try:
                low, high = obj.states[AABB].get_value()
                record["object_aabb"] = [_values(low), _values(high)]
            except Exception:
                record["object_aabb"] = None
            if Open in getattr(obj, "states", {}):
                record["open"] = bool(obj.states[Open].get_value())
                record["joint_positions"] = self._joint_positions(obj)
        if target is not None:
            pred_cls = Inside if tool == "place_inside" else OnTop
            if obj is not None and pred_cls in getattr(obj, "states", {}):
                try:
                    record["target_predicate"] = bool(
                        obj.states[pred_cls].get_value(target)
                    )
                except Exception:
                    record["target_predicate"] = False
        return record

    def _release_grasp(self) -> bool:
        if not self._is_grasping():
            return False
        self.robot.release_grasp_immediately(self.arm)
        return True

    def _attach(self, obj) -> tuple[bool, str]:
        import omnigibson as og
        from omnigibson.object_states import AABB

        if obj is None:
            return False, "object_unresolved"
        if self._is_grasping(obj):
            return True, "already_attached"
        if self._is_grasping():
            self._release_grasp()

        eef_pos, _ = self.robot.eef_links[self.arm].get_position_orientation()
        obj_pos, obj_orn = obj.get_position_orientation()
        low, high = obj.states[AABB].get_value()
        centre = (low + high) / 2.0
        obj.set_position_orientation(
            position=obj_pos + (eef_pos - centre), orientation=obj_orn
        )
        og.sim.step_physics()
        link_name = obj.root_link_name
        joint_type = self.robot._get_assisted_grasp_joint_type(obj, link_name)
        if joint_type is None:
            return False, "object_not_assisted_graspable"
        self.robot._establish_grasp(
            target_obj=obj,
            target_link_name=link_name,
            arm=self.arm,
            contact_pos_world=eef_pos,
            joint_type=joint_type,
        )
        og.sim.step_physics()
        return self._is_grasping(obj), f"assisted_{joint_type}"

    def _restore_robot_configuration(self, context: ManipulationContext) -> None:
        """Keep the physical head camera fixed across one semantic mutation.

        R1Pro has a floating base. The physics steps required by Open and assisted
        grasp otherwise settle both its base and arm, confounding object changes
        with a moving camera. This restoration is RGB-evidence-only and happens
        after the physical state transition; it is never called by text tools.
        """
        if context.robot_joint_positions is not None:
            self.robot.set_joint_positions(context.robot_joint_positions)
        self.robot.set_position_orientation(
            position=context.robot_position,
            orientation=context.robot_orientation,
        )
        self.robot.keep_still()

    def before(self, tool: str, arguments: dict) -> ManipulationContext:
        robot_position, robot_orientation = self.robot.get_position_orientation()
        context = ManipulationContext(
            tool=tool,
            arguments=dict(arguments),
            before=self._snapshot(tool, arguments),
            robot_position=robot_position.clone(),
            robot_orientation=robot_orientation.clone(),
            robot_joint_positions=self.robot.get_joint_positions().clone(),
        )
        if tool not in {"place_inside", "place_on"}:
            return context

        name = arguments.get("name")
        target_name = arguments.get("container") or arguments.get("surface")
        pred = "Inside" if tool == "place_inside" else "OnTop"
        target = self.aci._resolve(target_name) if target_name else None
        valid = (
            name is not None
            and self.aci._held == name
            and target is not None
            and self.aci._is_near(target)
            and (name, target_name, pred) in self.aci._pose_cache
        )
        if valid and self._is_grasping():
            context.released_object = self.aci._resolve(name)
            context.released_for_place = self._release_grasp()
        return context

    def abort(self, context: ManipulationContext) -> None:
        """Restore a carry constraint if dispatch failed after a place pre-hook."""
        if context.released_for_place:
            self._attach(context.released_object)
        self._restore_robot_configuration(context)

    def after(self, context: ManipulationContext, result) -> dict:
        physical_ok, physical_reason = bool(result.ok), "semantic_state_only"
        obj = self.aci._resolve(context.arguments.get("name"))
        if context.tool == "grasp" and result.ok:
            physical_ok, physical_reason = self._attach(obj)
        elif context.tool in {"place_inside", "place_on"}:
            if result.ok:
                after = self._snapshot(context.tool, context.arguments)
                physical_ok = (
                    bool(after.get("target_predicate")) and not self._is_grasping()
                )
                physical_reason = "predicate_true_and_constraint_released"
            elif context.released_for_place:
                restored, reason = self._attach(context.released_object)
                physical_ok = False
                physical_reason = f"place_failed_carry_restored={restored}:{reason}"
        elif context.tool in {"open", "close"} and result.ok:
            expected = context.tool == "open"
            state_after = self._snapshot(context.tool, context.arguments)
            motion = joint_position_delta(
                context.before.get("joint_positions"),
                state_after.get("joint_positions"),
            )
            state_changed = context.before.get("open") is not expected
            joint_motion_ok = not state_changed or motion["max_abs_delta"] > 1e-5
            physical_ok = state_after.get("open") is expected and joint_motion_ok
            physical_reason = (
                f"Open={expected}; state_changed={state_changed}; "
                f"joint_max_delta={motion['max_abs_delta']}"
            )

        self._restore_robot_configuration(context)
        after = self._snapshot(context.tool, context.arguments)
        if result.ok and not physical_ok:
            result.ok = False
            result.reason = (
                f"{result.reason}; rgb_physical_sync_failed:{physical_reason}"
            )
        return {
            "physical_ok": bool(physical_ok),
            "physical_reason": physical_reason,
            "state_before": context.before,
            "state_after": after,
            "joint_motion": joint_position_delta(
                context.before.get("joint_positions"),
                after.get("joint_positions"),
            )
            if context.tool in {"open", "close"}
            else None,
        }

    def pixel_audit(
        self,
        *,
        before_frame,
        control_frame,
        after_frame,
        before_camera: dict,
        after_camera: dict,
        evidence: dict,
    ) -> dict:
        before_pos = before_camera["camera_xyz"]
        after_pos = after_camera["camera_xyz"]
        before_quat = before_camera["camera_orientation_xyzw"]
        after_quat = after_camera["camera_orientation_xyzw"]
        camera_delta = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(before_pos, after_pos))
        )
        quat_dot = abs(
            sum(float(a) * float(b) for a, b in zip(before_quat, after_quat))
        )
        camera_unchanged = camera_delta <= 1e-4 and abs(1.0 - quat_dot) <= 1e-4
        if not camera_unchanged:
            return {
                "available": False,
                "reason": "camera_moved_within_pair",
                "camera_position_delta_m": camera_delta,
                "camera_quaternion_abs_dot": quat_dot,
            }

        common = {
            "camera_xyz": before_pos,
            "camera_orientation_xyzw": before_quat,
            "intrinsic": before_camera["intrinsic"],
            "width": int(before_camera["resolution"][1]),
            "height": int(before_camera["resolution"][0]),
        }
        before_box = project_world_aabb(
            evidence["state_before"].get("object_aabb"), **common
        )
        after_box = project_world_aabb(
            evidence["state_after"].get("object_aabb"), **common
        )
        roi = union_roi(before_box, after_box)
        if roi is None:
            return {"available": False, "reason": "object_aabb_not_projectable"}
        control = pixel_change_metrics(before_frame, control_frame, roi)
        mutation = pixel_change_metrics(before_frame, after_frame, roi)
        above_control = mutation["mean_abs_diff"] > max(
            control["mean_abs_diff"] + 0.5,
            control["mean_abs_diff"] * 1.5,
        )
        return {
            "available": True,
            "roi_xyxy": list(roi),
            "projected_before_xyxy": list(before_box) if before_box else None,
            "projected_after_xyxy": list(after_box) if after_box else None,
            "unchanged_render_control": control,
            "tool_mutation": mutation,
            "above_control": bool(above_control),
            "camera_position_delta_m": camera_delta,
            "camera_quaternion_abs_dot": quat_dot,
        }

    def release_before_reset(self) -> None:
        """Remove a stale RGB-only assisted-grasp joint before reset or close."""
        self._release_grasp()


__all__ = [
    "MUTATION_TOOLS",
    "RGBManipulationEvidence",
    "joint_position_delta",
    "pixel_change_metrics",
    "project_world_aabb",
    "union_roi",
]
