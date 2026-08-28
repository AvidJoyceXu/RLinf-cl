"""Run an official-room G5 manipulation trajectory with paired RGB evidence."""

from __future__ import annotations

import argparse
import json
import traceback


def _run(env, tool: str, arguments: dict, trace: list[dict]) -> dict:
    result = env.call(tool, arguments)
    row = {
        "tool": tool,
        "arguments": arguments,
        "payload": result["payload"],
        "meta": result["meta"],
    }
    trace.append(row)
    if not bool(result["payload"].get("ok", True)):
        raise RuntimeError(f"G5 tool failed: {json.dumps(row, sort_keys=True)}")
    return result


def _scope_objects(task) -> dict[str, object]:
    """Return v3.8-wrapper and v3.9-direct scope entities by simulator name."""
    objects = {}
    for entity in task.object_scope.values():
        obj = getattr(entity, "wrapped_obj", None) or entity
        name = getattr(obj, "name", None)
        if name:
            objects[name] = obj
    return objects


def _select_open_target(env, explicit: str | None, Open, Inside):
    objects = _scope_objects(env.env.task)
    ordered_names = []
    if explicit:
        ordered_names.append(explicit)
    for names in (
        env.plan.open_targets,
        env.plan.place_targets,
        env.plan.close_targets,
        sorted(objects),
    ):
        for name in names:
            if name not in ordered_names:
                ordered_names.append(name)

    candidates = []
    for order, name in enumerate(ordered_names):
        obj = env.aci._resolve(name)
        if obj is None or Open not in getattr(obj, "states", {}):
            continue
        children = []
        for child_name, child in objects.items():
            if child is obj or Inside not in getattr(child, "states", {}):
                continue
            try:
                if bool(child.states[Inside].get_value(obj)):
                    children.append(child_name)
            except Exception:
                continue
        candidates.append(
            (
                bool(children),
                not bool(obj.states[Open].get_value()),
                -order,
                name,
                sorted(children),
            )
        )
    if not candidates:
        raise RuntimeError("G5c found no task-scope object with an Open state")
    _, _, _, name, children = max(candidates)
    if explicit and name != explicit:
        raise RuntimeError(f"requested G5c target is not openable: {explicit}")
    return name, children


def _validate_open_close_events(events: list[dict]) -> dict:
    accepted = {}
    for tool, before_value, after_value in (
        ("open", False, True),
        ("close", True, False),
    ):
        matches = [event for event in events if event.get("tool") == f"{tool}:post"]
        if not matches:
            raise RuntimeError(f"G5c sidecar is missing {tool}:post")
        evidence = matches[-1].get("rgb_manipulation", {})
        motion = evidence.get("joint_motion") or {}
        pixel = evidence.get("pixel_audit") or {}
        state_before = evidence.get("state_before") or {}
        state_after = evidence.get("state_after") or {}
        checks = {
            "physical_ok": evidence.get("physical_ok") is True,
            "state_transition": state_before.get("open") is before_value
            and state_after.get("open") is after_value,
            "joint_motion": motion.get("matching_joints", 0) > 0
            and motion.get("changed_joints", 0) > 0
            and motion.get("max_abs_delta", 0.0) > 1e-5,
            "camera_fixed": pixel.get("camera_position_delta_m", 1.0) <= 1e-4,
            "pixel_above_control": pixel.get("available") is True
            and pixel.get("above_control") is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"G5c {tool} acceptance failed: {checks}; {evidence}")
        accepted[tool] = {"checks": checks, "joint_motion": motion, "pixel": pixel}
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="picking_up_trash")
    parser.add_argument("--instance-source", default="2026-v3.9.1")
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--render-iters", type=int, default=8)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="0 runs the complete expert plan; a positive value stops after that "
        "many place calls for a grasp/place micro-test",
    )
    parser.add_argument(
        "--open-close-only",
        action="store_true",
        help="run the G5c closed->open->closed physical and pixel acceptance only",
    )
    parser.add_argument(
        "--open-target",
        default=None,
        help="simulator object name for G5c; otherwise prefer a task-scope "
        "openable container that already contains an object",
    )
    parser.add_argument(
        "--allow-empty-open-target",
        action="store_true",
        help="permit G5c on an openable object with no task-scope contents",
    )
    parser.add_argument(
        "--second-session",
        action="store_true",
        help="reset the same instance again and verify no assisted-grasp state leaks",
    )
    args = parser.parse_args()

    import omnigibson as og
    from omnigibson.object_states import Inside, Open

    from rlinf.envs.behavior.env_server import BehaviorEnv
    from rlinf.envs.behavior.official_scene import runtime_scene_inventory

    env = BehaviorEnv(
        args.activity,
        obs_mode="full",
        instances_per_activity=0,
        rgb=True,
        fast_reset=True,
        instance_source=args.instance_source,
        debug_video_dir=args.out_dir,
        debug_video_fps=args.fps,
        debug_render_iters=args.render_iters,
        scene_profile="official_rooms",
        rgb_manipulation_evidence=True,
    )
    trace: list[dict] = []
    try:
        session = env.start(args.instance_id)
        instance_id = int(session["instance_id"])
        contract = env.official_scene_contract
        if contract is None:
            raise RuntimeError("official_rooms did not produce a scene contract")
        inventory = runtime_scene_inventory(env.env.scene)
        assert env.video_recorder is not None
        env.video_recorder.metadata.update(
            {
                "scene_profile": env.scene_profile,
                "scene_contract": contract.record(),
                "runtime_scene_inventory": inventory,
                "camera_source": "physical_r1_zed_head_sensor",
                "trajectory_kind": (
                    "official_room_g5c_open_close"
                    if args.open_close_only
                    else "official_room_g5_semantic_manipulation"
                ),
                "rgb_manipulation_evidence": True,
            }
        )

        opened = []
        if not args.open_close_only:
            for target_name in env.plan.place_targets:
                target = env.aci._resolve(target_name)
                if target is not None and Open in getattr(target, "states", {}):
                    _run(env, "go_to", {"name": target_name}, trace)
                    _run(env, "open", {"name": target_name}, trace)
                    opened.append(target_name)

        completed_places = 0
        stopped_early = bool(args.open_close_only)
        open_close_target = None
        open_close_children = []
        open_close_audit = None
        if args.open_close_only:
            open_close_target, open_close_children = _select_open_target(
                env, args.open_target, Open, Inside
            )
            if not open_close_children and not args.allow_empty_open_target:
                raise RuntimeError(
                    f"G5c target {open_close_target} has no task-scope contents; "
                    "choose another target or pass --allow-empty-open-target"
                )
            env.video_recorder.metadata.update(
                {
                    "g5c_target": open_close_target,
                    "g5c_inside_children": open_close_children,
                    "text_visibility_contract": (
                        "closed hides transitive inside descendants; open reveals "
                        "their dynamic bboxes; RGB helper is not imported by text"
                    ),
                }
            )
            _run(env, "go_to", {"name": open_close_target}, trace)
            target = env.aci._resolve(open_close_target)
            if bool(target.states[Open].get_value()):
                _run(env, "close", {"name": open_close_target}, trace)
            _run(env, "open", {"name": open_close_target}, trace)
            _run(env, "close", {"name": open_close_target}, trace)
            open_close_audit = _validate_open_close_events(env.video_recorder.events)
        else:
            for tool, arguments in env.plan.calls:
                _run(env, tool, arguments, trace)
                if tool in {"place_inside", "place_on"}:
                    completed_places += 1
                    if args.max_pairs and completed_places >= args.max_pairs:
                        stopped_early = True
                        break

        if not stopped_early:
            close_targets = list(env.plan.close_targets)
            for target_name in opened:
                if (
                    target_name not in close_targets
                    and target_name not in env.plan.open_targets
                ):
                    close_targets.append(target_name)
            for target_name in close_targets:
                _run(env, "go_to", {"name": target_name}, trace)
                _run(env, "close", {"name": target_name}, trace)
            final = _run(env, "end_task", {}, trace)
            if not final["meta"]["is_success"]:
                raise RuntimeError("complete G5 plan did not satisfy the BDDL goal")

        first_record = env.end()["debug_video"]
        if not first_record or not first_record["complete"]:
            raise RuntimeError(f"G5 recorder did not finalize: {first_record!r}")
        first_sidecar_path = env.video_recorder.sidecar_path

        second_record = None
        if args.second_session:
            env.start(instance_id)
            if env.aci._held is not None or env.rgb_manipulation._is_grasping():
                raise RuntimeError("assisted-grasp state leaked across session reset")
            second_record = env.end()["debug_video"]

        print(
            "G5_RGB_MANIPULATION_OK "
            + json.dumps(
                {
                    "activity": args.activity,
                    "instance_id": instance_id,
                    "completed_places": completed_places,
                    "stopped_early": stopped_early,
                    "open_close_target": open_close_target,
                    "open_close_children": open_close_children,
                    "open_close_audit": open_close_audit,
                    "trace": trace,
                    "video_path": first_record["video_path"],
                    "sidecar_path": first_sidecar_path,
                    "frames": first_record["frames"],
                    "source_frames_dir": first_record["source_frames_dir"],
                    "second_session_complete": bool(
                        second_record and second_record["complete"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
        og.shutdown()


if __name__ == "__main__":
    main()
