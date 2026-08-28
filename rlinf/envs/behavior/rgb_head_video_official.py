"""Record a tool-aligned R1 head-camera video under official 2026 scene conditions."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4


def _validate_inventory(activity: str, expected: dict, actual: dict) -> None:
    if actual["dataset_object_count"] < expected["dataset_object_count"]:
        raise RuntimeError(
            "official-room runtime loaded fewer DatasetObjects than the published "
            f"partial template: expected>={expected['dataset_object_count']} "
            f"actual={actual['dataset_object_count']}"
        )
    if activity == "picking_up_trash":
        categories = actual["categories"]
        required_groups = (
            {"breakfast_table", "coffee_table"},
            {"bottom_cabinet", "top_cabinet"},
            {"sofa"},
        )
        missing = [
            sorted(group) for group in required_groups if not group & categories.keys()
        ]
        if missing:
            raise RuntimeError(
                "official picking_up_trash scene is missing furniture groups: "
                f"{missing}; loaded={sorted(categories)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="picking_up_trash")
    parser.add_argument("--instance-source", default="2026-v3.9.1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--render-iters", type=int, default=8)
    args = parser.parse_args()

    import omnigibson as og

    from rlinf.envs.behavior.env_server import BehaviorEnv
    from rlinf.envs.behavior.official_scene import runtime_scene_inventory
    from rlinf.envs.behavior.rft_sample import reset_to_instance

    env = BehaviorEnv(
        args.activity,
        obs_mode="fov",
        instances_per_activity=1,
        rgb=True,
        fast_reset=True,
        instance_source=args.instance_source,
        debug_video_dir=args.out_dir,
        debug_video_fps=args.fps,
        debug_render_iters=args.render_iters,
        scene_profile="official_rooms",
    )
    instance = env.instances[0]
    reset_to_instance(env.aci, env.env, instance, reset_scene=False)

    contract = env.official_scene_contract
    if contract is None:
        raise RuntimeError("official_rooms did not produce a scene contract")
    actual_inventory = runtime_scene_inventory(env.env.scene)
    _validate_inventory(args.activity, contract.partial_room_template, actual_inventory)

    env.session_id = uuid4().hex
    assert env.video_recorder is not None
    video_path = env.video_recorder.begin(
        {
            "session_id": env.session_id,
            "activity": args.activity,
            "scene": env.scene,
            "scene_profile": env.scene_profile,
            "scene_contract": contract.record(),
            "runtime_scene_inventory": actual_inventory,
            "instance_id": instance.instance_id,
            "instance_path": instance.path,
            "instance_source": args.instance_source,
            "obs_mode": "fov",
            "camera_source": "physical_r1_zed_head_sensor",
            "renderer": "official_runtime_default_RealTimePathTracing",
            "trajectory_kind": "official_room_camera_tool_trajectory",
        }
    )
    env._record_debug_frame("session_start", ok=True)
    calls = [
        ("turn_left", {}),
        ("move_ahead", {}),
        ("turn_right", {}),
        ("strafe_right", {}),
        ("move_back", {}),
        ("turn_right", {}),
    ]
    results = []
    for tool, arguments in calls:
        result = env.call(tool, arguments)
        results.append(
            {
                "tool": tool,
                "arguments": arguments,
                "ok": bool(result["payload"].get("ok", True)),
            }
        )
    end_record = env.end()
    sidecar = end_record["debug_video"]
    if not sidecar or sidecar.get("frames") != len(calls) + 1:
        raise RuntimeError(f"expected {len(calls) + 1} aligned frames, got {sidecar!r}")
    print(
        "G4_OFFICIAL_ROOM_VIDEO_OK "
        + json.dumps(
            {
                "video_path": video_path,
                "frames": sidecar["frames"],
                "events": results,
                "sidecar_path": env.video_recorder.sidecar_path,
                "source_frames_dir": sidecar["source_frames_dir"],
                "runtime_scene_inventory": actual_inventory,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    env.close()
    og.shutdown()


if __name__ == "__main__":
    main()
