"""Capture one lossless frame from the official R1 Pro head sensor."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _usd_pinhole_intrinsics(sensor):
    """Compute K from the USD camera model without Replicator metadata."""
    width = float(sensor.image_width)
    height = float(sensor.image_height)
    focal_length = float(sensor.focal_length)
    horizontal_aperture = float(sensor.horizontal_aperture)
    fx = focal_length * width / horizontal_aperture
    # USD cameras use square pixels. The vertical field of view follows from the
    # render-product aspect ratio when no explicit vertical aperture is authored.
    fy = fx
    return [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", default="picking_up_trash")
    parser.add_argument("--instance-source", default="2026-v3.9.1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--render-iters", type=int, default=8)
    parser.add_argument("--full-scene", action="store_true")
    args = parser.parse_args()

    if args.render_iters < 1:
        raise ValueError("--render-iters must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    import imageio.v2 as imageio
    import numpy as np
    import omnigibson as og

    from rlinf.envs.behavior.env_server import BehaviorEnv
    from rlinf.envs.behavior.rft_sample import reset_to_instance
    from rlinf.envs.behavior.utils import convert_uint8_rgb

    env = BehaviorEnv(
        args.activity,
        obs_mode="fov",
        instances_per_activity=1,
        rgb=True,
        partial_scene=not args.full_scene,
        fast_reset=True,
        instance_source=args.instance_source,
    )
    instance = env.instances[0]
    reset_to_instance(env.aci, env.env, instance, reset_scene=False)

    robot = env.env.robots[0]
    matches = [
        (name, sensor)
        for name, sensor in robot.sensors.items()
        if "zed_link:Camera:0" in name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one R1 ZED head sensor, got "
            f"{[name for name, _ in matches]!r}; all={sorted(robot.sensors)}"
        )
    sensor_name, sensor = matches[0]

    for _ in range(args.render_iters):
        og.sim.render()
    obs = sensor.get_obs()[0]
    if "rgb" not in obs:
        raise RuntimeError(
            f"head sensor {sensor_name!r} returned modalities {sorted(obs)} without rgb"
        )
    frame = convert_uint8_rgb(obs["rgb"])
    frame = frame.detach().cpu().numpy() if hasattr(frame, "detach") else frame
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.shape != (720, 720, 3):
        raise RuntimeError(
            f"expected official 720x720 head RGB, got shape={frame.shape}"
        )
    channel_std = frame.reshape(-1, 3).std(axis=0)
    if int(frame.max()) == int(frame.min()) or float(channel_std.mean()) < 2.0:
        raise RuntimeError(
            "head RGB is blank or nearly uniform: "
            f"min={frame.min()} max={frame.max()} channel_std={channel_std.tolist()}"
        )

    png_path = os.path.join(
        args.out_dir,
        f"{args.activity}__instance-{instance.instance_id}__r1-zed.png",
    )
    json_path = os.path.splitext(png_path)[0] + ".json"
    imageio.imwrite(png_path, frame)

    position, orientation = sensor.get_position_orientation()
    record = {
        "activity": args.activity,
        "scene": env.scene,
        "instance_id": instance.instance_id,
        "instance_path": instance.path,
        "instance_source": args.instance_source,
        "partial_scene": not args.full_scene,
        "sensor_name": sensor_name,
        "sensor_prim_path": sensor.prim_path,
        "resolution": [int(sensor.image_height), int(sensor.image_width)],
        "intrinsic_matrix": _usd_pinhole_intrinsics(sensor),
        "intrinsic_source": "usd_focal_length_horizontal_aperture_square_pixels",
        "world_position": _tolist(position),
        "world_orientation_xyzw": _tolist(orientation),
        "frame": {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "min": int(frame.min()),
            "max": int(frame.max()),
            "mean": round(float(frame.mean()), 6),
            "channel_std": [round(float(value), 6) for value in channel_std],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "omnigibson": importlib.metadata.version("omnigibson"),
            "isaacsim": importlib.metadata.version("isaacsim"),
            "bddl": importlib.metadata.version("bddl"),
        },
        "render_iters": args.render_iters,
        "elapsed_s": round(time.time() - started, 3),
        "png_path": png_path,
    }
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)

    print("G3_HEAD_RGB_OK " + json.dumps(record, sort_keys=True), flush=True)
    env.close()
    og.shutdown()


if __name__ == "__main__":
    main()
