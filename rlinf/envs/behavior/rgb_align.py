"""Is the rendered camera the SAME camera `detect.py` projects through?

Run before trusting any multimodal-vs-text comparison. `detect` shows the policy boxes
computed by our own pinhole in `viewpoint.py`/`detect.py`; an RGB agent sees pixels from
Kit's camera. If the two disagree, the two agent classes are looking at different worlds
and every comparison between them is confounded -- silently, because each looks plausible
on its own.

The test is numeric, not visual. Ground truth is the renderer's own `bbox_2d_loose` (the
modality that, like our projection, bounds an object's full extent rather than its visible
silhouette); the prediction is `detect._project` over the same pose and extent.

MEASURED 2026-08-21, house_double_floor_lower, 5 yaws, ~490 object observations:

    intrinsics   renderer fx=fy=92.3760 cx=160 cy=120  ==  ours, exactly
    pose         readback delta 0.00 rad at every yaw
    projection   median centre error 1.0-1.5 px on a 320x240 image
    IoU          median 0.57-0.64

The centre error is the alignment claim, and it passes. The IoU is NOT 1.0 and must not be
read as misalignment: the renderer bounds the object's true oriented extent while we
project a world-axis-aligned box from `o.aabb`, so the two differ in SIZE while sharing a
centre. Occlusion clipping differs too.

TWO THINGS THAT WILL COST A DAY IF FORGOTTEN:

1. **The render lags the camera.** With 2 `og.sim.step()` calls after
   `set_position_orientation`, the boxes still describe the PREVIOUS pose -- which reads
   as an IoU that is excellent at some yaws and ~0 at others, and looks exactly like a
   wrong yaw convention. It is not. 8 steps suffices here; always read the pose back.
2. **Set the aperture, not an FOV.** `horizontal_aperture = 2 * focal_length * tan(60deg)`
   at 320x240 reproduces our pinhole exactly; there is no FOV setter.

Usage:  python -m rlinf.envs.behavior.rgb_align [scene_model]
"""

import math, sys
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from scipy.spatial.transform import Rotation as R

gm.HEADLESS = True; gm.ENABLE_OBJECT_STATES = True; gm.RENDER_VIEWER_CAMERA = True
W, H = 320, 240
HALF = math.radians(60.0)
CAM_Z = 1.2
SCENE = sys.argv[1] if len(sys.argv) > 1 else "house_double_floor_lower"


def _wrapd(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cam_quat(yaw, pitch):
    """USD camera: looks down -Z, +Y up, +X right."""
    f = np.array([math.cos(yaw) * math.cos(pitch),
                  math.sin(yaw) * math.cos(pitch),
                  math.sin(pitch)], dtype=float)
    f /= np.linalg.norm(f)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    M = np.column_stack([right, up, -f])          # camera axes in world frame
    return R.from_matrix(M).as_quat()             # xyzw


cfg = {"scene": {"type": "InteractiveTraversableScene", "scene_model": SCENE},
       "robots": [], "task": {"type": "DummyTask"}}
env = og.Environment(configs=cfg)
cam = og.sim.viewer_camera
cam.image_width, cam.image_height = W, H
cam.horizontal_aperture = 2.0 * float(cam.focal_length) * math.tan(HALF)
cam.add_modality("bbox_2d_loose"); cam.add_modality("rgb")
for _ in range(3): og.sim.step()

from rlinf.envs.behavior.detect import _project
from rlinf.envs.behavior.viewpoint import Viewpoint

scene = og.sim.scenes[0]
objs = [o for o in scene.objects if hasattr(o, "aabb")]
px = float(np.mean([float(o.get_position_orientation()[0][0]) for o in objs]))
py = float(np.mean([float(o.get_position_orientation()[0][1]) for o in objs]))

print("intrinsics match:", np.allclose(np.array(cam.intrinsic_matrix)[0, 0], (W/2)/math.tan(HALF)))
rows = []
for yaw in (0.0, 0.7, 1.9, 3.4, 5.0):
    cam.set_position_orientation(position=[px, py, CAM_Z], orientation=cam_quat(yaw, 0.0))
    for _ in range(8): og.sim.step()          # let the render catch up with the pose
    # read the pose BACK and recover the yaw the camera actually has, so a stale or
    # mis-set orientation shows up as a number instead of as a bad IoU
    _p, _q = cam.get_position_orientation()
    Rm = R.from_quat(np.array(_q, dtype=float)).as_matrix()
    fwd = -Rm[:, 2]
    yaw_actual = math.atan2(fwd[1], fwd[0])
    print(f"  yaw set={yaw:5.2f}  readback={yaw_actual:5.2f}  delta={_wrapd(yaw_actual - yaw):+5.2f}")
    obs, info = cam.get_obs()
    id2lab = (info or {}).get("bbox_2d_loose", {})
    rb = {}
    for rec in obs["bbox_2d_loose"]:
        lab = id2lab.get(int(rec[0])) or id2lab.get(str(int(rec[0])))
        nm = lab.get("class") if isinstance(lab, dict) else lab
        b = (float(rec[1]), float(rec[2]), float(rec[3]), float(rec[4]))
        if b[2] > b[0] and b[3] > b[1]:
            rb.setdefault(str(nm), []).append(b)
    view = Viewpoint(x=px, y=py, yaw=yaw, pitch=0.0)
    sc, cerr = [], []
    for o in objs:
        lo, hi = o.aabb
        centre = tuple(float(lo[i] + hi[i]) / 2.0 for i in range(3))
        ext = tuple(float(hi[i] - lo[i]) / 2.0 for i in range(3))
        pr = _project(view, centre, ext)
        if pr is None: continue
        cand = rb.get(o.category, [])
        if cand:
            best_c = max(cand, key=lambda c: iou(pr[0], c))
            sc.append(iou(pr[0], best_c))
            pc = ((pr[0][0]+pr[0][2])/2.0, (pr[0][1]+pr[0][3])/2.0)
            rc = ((best_c[0]+best_c[2])/2.0, (best_c[1]+best_c[3])/2.0)
            cerr.append(math.hypot(pc[0]-rc[0], pc[1]-rc[1]))
    if sc:
        rows.append((yaw, len(sc), float(np.median(sc)), float(np.mean(sc))))
        print(f"  yaw={yaw:4.1f}  n={len(sc):3d}  median IoU={np.median(sc):.3f}  "
              f"median centre err={np.median(cerr):5.1f} px")
if rows:
    print(f"\nOVERALL median-of-medians IoU: {np.median([r[2] for r in rows]):.3f}")
og.shutdown()
