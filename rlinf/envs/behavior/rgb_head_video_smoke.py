"""Record one tool-aligned trajectory from the physical R1 head camera."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="picking_up_trash")
    parser.add_argument("--instance-source", default="2026-v3.9.1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--render-iters", type=int, default=3)
    args = parser.parse_args()

    import omnigibson as og

    from rlinf.envs.behavior.env_server import BehaviorEnv
    from rlinf.envs.behavior.rft_sample import reset_to_instance

    env = BehaviorEnv(
        args.activity,
        obs_mode="fov",
        instances_per_activity=1,
        rgb=True,
        partial_scene=True,
        fast_reset=True,
        instance_source=args.instance_source,
        debug_video_dir=args.out_dir,
        debug_video_fps=args.fps,
        debug_render_iters=args.render_iters,
        scene_profile="task_relevant",
    )
    instance = env.instances[0]
    reset_to_instance(env.aci, env.env, instance, reset_scene=False)

    env.session_id = uuid4().hex
    assert env.video_recorder is not None
    video_path = env.video_recorder.begin(
        {
            "session_id": env.session_id,
            "activity": args.activity,
            "scene": env.scene,
            "instance_id": instance.instance_id,
            "instance_path": instance.path,
            "instance_source": args.instance_source,
            "obs_mode": "fov",
            "camera_source": "physical_r1_zed_head_sensor",
            "trajectory_kind": "camera_tool_harness_smoke",
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
        "G4_HEAD_VIDEO_OK "
        + json.dumps(
            {
                "video_path": video_path,
                "frames": sidecar["frames"],
                "events": results,
                "sidecar_path": env.video_recorder.sidecar_path,
                "source_frames_dir": sidecar["source_frames_dir"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    env.close()
    og.shutdown()


if __name__ == "__main__":
    main()
