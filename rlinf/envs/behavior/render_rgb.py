"""Render RGB from a booted BEHAVIOR scene: one frame, and a moving-camera video.

Everything in this project so far has been pixel-free -- the semantic ACI reads state,
and `detect.py` projects ground-truth boxes through a pinhole model we wrote. This is
the first code that looks at the renderer, and it exists for two reasons:

1. **A deliverable.** A PNG of the initial scene and an MP4 of a moving camera, written
   into the repo so they can be eyeballed.
2. **A correctness check we cannot skip.** `detect.py` reports boxes in a *virtual*
   320x240 sensor with a 60-degree half-angle. The R1's head camera and Kit's viewer
   camera have different intrinsics. If the camera we render and the camera we project
   through are not the same camera, then every future comparison between a text-only
   agent (which sees projected boxes) and a multimodal agent (which sees pixels) is
   confounded. So this prints both intrinsic matrices side by side.

Cost note: the robot's cameras stay OFF (`obs_modalities: ["proprio"]`). We use
`og.sim.viewer_camera`, which is a full VisionSensor that is not attached to the robot,
so enabling RGB costs exactly one camera rather than the R1's three.

    python -m rlinf.envs.behavior.render_rgb --activity picking_up_trash \
        --out /workspace/spatialcode/rgb_demo --frames 120
"""

from __future__ import annotations

import argparse
import math
import os

IMAGE_W, IMAGE_H = 640, 640      # square, so the vertical and horizontal FOV agree


def look_quat(yaw: float, pitch: float = 0.0):
    """Quaternion (x,y,z,w) for a camera looking along `yaw`, measured from world +X.

    Kit's cameras look down local -Z with +Y up, and with pan=0 the camera faces world
    +Y -- so a yaw measured from +X needs `pan = yaw - pi/2`. Getting this wrong is a
    silent 90-degree error, which is exactly the kind of thing that looks like a
    perception bug later.
    """
    import omnigibson.utils.transform_utils as T
    import torch as th

    return T.euler2quat(th.tensor([math.pi / 2 + pitch, 0.0, yaw - math.pi / 2]))


def configure_debug_renderer(
    *,
    render_mode: str = "PathTracing",
    texture_budget: float = 1.0,
    dlss_quality: int = 2,
    spp: int = 8,
    total_spp: int = 128,
) -> None:
    """Apply the renderer settings shared by demos and trajectory recording.

    Isaac Sim 4.5 real-time RTX produces grey noise on the Blackwell host used for
    this project. Path tracing is the verified workaround. The OptiX denoiser must
    remain disabled here: on this runtime it fails to initialise and returns an empty
    frame instead of a noisy one.
    """
    import omnigibson.lazy as lazy

    settings = lazy.carb.settings.get_settings()
    settings.set_float(
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
        texture_budget,
    )
    settings.set_int("/rtx/post/dlss/execMode", dlss_quality)
    settings.set_string("/rtx/rendermode", render_mode)
    if render_mode == "PathTracing":
        settings.set_int("/rtx/pathtracing/spp", spp)
        settings.set_int("/rtx/pathtracing/totalSpp", total_spp)
        settings.set_int("/rtx/pathtracing/clampSpp", total_spp)
        settings.set_int("/rtx/pathtracing/maxBounces", 4)
        settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activity", default="picking_up_trash")
    ap.add_argument("--out", default="/workspace/spatialcode/rgb_demo")
    ap.add_argument("--frames", type=int, default=120, help="video frames")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--radius", type=float, default=2.5, help="orbit radius, metres")
    ap.add_argument("--height", type=float, default=1.6, help="camera height, metres")
    ap.add_argument("--width", type=int, default=IMAGE_W)
    ap.add_argument("--height-px", type=int, default=IMAGE_H)
    ap.add_argument("--match-detect-fov", action="store_true",
                    help="set the aperture so the render matches detect.py's 60-degree "
                         "half-angle instead of Kit's default")
    ap.add_argument("--render-iters", type=int, default=16,
                    help="render passes per frame. OmniGibson uses 3 to PROPAGATE a "
                         "pose change, which is not the same as letting the RTX "
                         "denoiser converge -- at 3 the image is heavily speckled.")
    ap.add_argument("--texture-budget", type=float, default=1.0,
                    help="texture-streaming memory budget. RLinf sets 0.1 at runtime "
                         "(utils.apply_runtime_renderer_settings) to save VRAM for "
                         "training, which starves textures in a render.")
    ap.add_argument("--render-mode", default="PathTracing",
                    choices=["PathTracing", "RaytracedLighting"],
                    help="RTX Real-Time (RaytracedLighting) renders BLURRY, PIXELATED "
                         "AND NOISY on Blackwell GPUs under Isaac Sim 4.5 -- a known "
                         "Kit/Blackwell incompatibility (isaac-sim/IsaacLab#2869), "
                         "fixed only in Isaac Sim 5.0. Path tracing is the documented "
                         "workaround and, for a static scene we render repeatedly, it "
                         "also accumulates to a cleaner image.")
    ap.add_argument("--spp", type=int, default=8, help="path-tracing samples/pixel/frame")
    ap.add_argument("--total-spp", type=int, default=256,
                    help="accumulation target; repeated render() calls converge to it")
    ap.add_argument("--hq", action="store_true", default=True,
                    help="gm.ENABLE_HQ_RENDERING -- reflections, indirect diffuse, AO")
    ap.add_argument("--no-hq", dest="hq", action="store_false")
    ap.add_argument("--dlss-quality", type=int, default=2,
                    help="/rtx/post/dlss/execMode; OmniGibson sets 0 (Performance), "
                         "which speckles a sparsely sampled frame")
    ap.add_argument("--full-scene", action="store_true",
                    help="load the whole house rather than task-relevant objects only")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import imageio.v2 as imageio
    import numpy as np
    import omnigibson as og
    import torch as th
    from omegaconf import OmegaConf
    from omnigibson.envs import VectorEnvironment

    from rlinf.envs.behavior.harvest_sft import resolve_scene_and_dir
    from rlinf.envs.behavior.utils import setup_omni_cfg

    scene, _ = resolve_scene_and_dir(args.activity)
    cfg = OmegaConf.load(
        "/workspace/RLinf/examples/embodiment/config/env/behavior_r1pro.yaml")
    OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.activity_name", args.activity, force_add=True)
    OmegaConf.update(cfg, "omni_config.scene.scene_model", scene, force_add=True)
    if not args.full_scene:
        OmegaConf.update(cfg, "omni_config.scene.load_task_relevant_only", True,
                         force_add=True)
    # The robot's own cameras stay off; we render from the viewer camera instead.
    OmegaConf.update(cfg, "omni_config.robots.0.obs_modalities", ["proprio"],
                     force_add=True)
    # This is a BOOT-TIME macro: the viewer camera is created during launch, so setting
    # it afterwards does nothing and `og.sim.viewer_camera` stays None.
    OmegaConf.update(cfg, "omni_config.macro.render_viewer_camera", True, force_add=True)

    print(f"activity : {args.activity}\nscene    : {scene}", flush=True)
    resolved = OmegaConf.to_container(setup_omni_cfg(cfg), resolve=True)
    if args.hq:
        # `setup_omni_cfg` maps a FIXED list of macros and ENABLE_HQ_RENDERING is not
        # in it, so this cannot come from the yaml. It is read at launch, hence here.
        # It turns on reflections, indirect diffuse, ambient occlusion and sampled
        # direct lighting (simulator.py:574-582) -- without it the image is speckled.
        from omnigibson.macros import gm
        gm.ENABLE_HQ_RENDERING = True
    vec_env = VectorEnvironment(1, resolved)
    vec_env.reset()
    env = vec_env.envs[0]

    configure_debug_renderer(
        render_mode=args.render_mode,
        texture_budget=args.texture_budget,
        dlss_quality=args.dlss_quality,
        spp=args.spp,
        total_spp=args.total_spp,
    )
    print(f"renderer : {args.render_mode} "
          f"(spp={args.spp}, totalSpp={args.total_spp})", flush=True)

    cam = og.sim.viewer_camera
    assert cam is not None, (
        "og.sim.viewer_camera is None -- macro.render_viewer_camera must be True "
        "BEFORE the simulator launches")
    cam.image_width, cam.image_height = args.width, args.height_px
    if args.match_detect_fov:
        # detect.py uses a 60-degree HALF-angle; aperture = 2 * f * tan(half_angle)
        cam.horizontal_aperture = 2.0 * float(cam.focal_length) * math.tan(math.radians(60.0))

    # Where to look: the robot's own position, so the frame contains the workspace.
    robot = env.robots[0]
    centre = robot.get_position_orientation()[0].clone()
    centre[2] = 0.0

    def shoot(yaw: float, pitch: float = -0.15):
        pos = th.tensor([float(centre[0]) + args.radius * math.cos(yaw),
                         float(centre[1]) + args.radius * math.sin(yaw),
                         args.height])
        # look back toward the centre
        cam.set_position_orientation(position=pos, orientation=look_quat(yaw + math.pi,
                                                                        pitch))
        for _ in range(args.render_iters):
            og.sim.render()       # no physics step: the scene is static, and repeated
                                  # renders let the denoiser converge
        raw = cam.get_obs()[0]["rgb"]
        arr = raw[..., :3]
        arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
        # `ascontiguousarray` is not decoration: the slice above is a view with a
        # non-contiguous last stride, and PIL fails on it with the opaque
        # `SystemError: tile cannot extend outside image`.
        return np.ascontiguousarray(arr, dtype=np.uint8)

    frame = shoot(0.0)
    if frame.size == 0:
        raise SystemExit(
            "renderer returned an EMPTY frame. Check the log for "
            "`optixDenoiserCreate ... Internal error`; if present, the OptiX denoiser "
            "failed to initialise and must stay disabled.")
    print(f"frame    : shape={frame.shape} dtype={frame.dtype} "
          f"min={frame.min()} max={frame.max()}", flush=True)
    png = os.path.join(args.out, f"{args.activity}_initial.png")
    imageio.imwrite(png, frame)
    print(f"wrote {png}  {frame.shape}", flush=True)

    mp4 = os.path.join(args.out, f"{args.activity}_orbit.mp4")
    with imageio.get_writer(mp4, fps=args.fps) as w:
        for i in range(args.frames):
            w.append_data(shoot(2.0 * math.pi * i / args.frames))
            if (i + 1) % 30 == 0:
                print(f"  frame {i + 1}/{args.frames}", flush=True)
    print(f"wrote {mp4}", flush=True)

    # --- the check that matters: are these the same camera? --------------------------
    K = cam.intrinsic_matrix
    fx_render = float(K[0][0]) if not hasattr(K, "shape") else float(K[0, 0])
    from rlinf.envs.behavior import detect as D
    from rlinf.envs.behavior import viewpoint as V
    fx_detect = (D.IMAGE_W / 2) / math.tan(V.FOV_HALF_ANGLE_RAD)
    print("\n--- camera model ---")
    print(f"rendered  : {args.width}x{args.height_px}  fx={fx_render:.2f}  "
          f"hFOV={2 * math.degrees(math.atan(args.width / (2 * fx_render))):.1f} deg")
    print(f"detect.py : {D.IMAGE_W}x{D.IMAGE_H}  fx={fx_detect:.2f}  "
          f"hFOV={2 * math.degrees(V.FOV_HALF_ANGLE_RAD):.1f} deg")
    print("MATCH" if abs(2 * math.degrees(math.atan(args.width / (2 * fx_render)))
                         - 2 * math.degrees(V.FOV_HALF_ANGLE_RAD)) < 1.0
          else "MISMATCH -- projected boxes and pixels describe different cameras")

    og.shutdown()


if __name__ == "__main__":
    main()
