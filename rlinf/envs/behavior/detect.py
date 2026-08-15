"""Detection-style perception: what is in view, not what is relevant.

`observe()` has until now returned the BDDL object scope -- median 9 objects out of the
260 in a real BEHAVIOR scene, and precisely the 9 the task needs. That is CaP-X's
**S1 (privileged)** tier, which they keep as a *reasoning upper bound* rather than a
headline, and it removes the entire object-identification problem before the episode
starts. See `0815 - interface audit`.

This module supplies the S1.5 replacement designed in `0815 - selection design`:

  * every scene object whose projected box lands in the camera frustum is reported,
    scope member or not, so the task objects arrive mixed with the furniture;
  * each detection carries a **view-local** handle, a 2D box, a category and a score --
    and no instance index, so identity across frames is the policy's problem;
  * selection resolves by box overlap and **refuses when the decision is not clean**:
    ambiguous when two candidates compete, stale when the handle predates a camera
    move, nothing-there when the region is empty.

WHAT IS STILL PRIVILEGED, and it must be said every time this is quoted: boxes are
projected from ground-truth poses, so there is no misdetection, no false positive and
no missed object; categories are always correct; and `score` is a geometric quantity
(how much of the frustum the box fills), not a detector's confidence. This is a
simulated perception interface, not a perception system. Noise -- drop rate, label
swap, box jitter -- is a staged upgrade measured against this baseline, kept separate
so "identification is hard" never gets confounded with "our noise model is hard".
"""
from __future__ import annotations

import functools
import glob
import json
import math
import os
from dataclasses import dataclass

# Architecture a manipulation-oriented segmentation stack conventionally filters out
# (COCO's "stuff" rather than "things"). Floors are kept: BDDL uses `floor.n.01` as a
# real support and the policy has to be able to select one.
STUFF_CATEGORIES = frozenset({
    "walls", "ceilings", "downlight", "garden_light", "lawn_light",
    "ceiling_light", "wall_light", "roof", "window", "door_frame",
})

IMAGE_W, IMAGE_H = 320, 240      # virtual sensor; boxes are reported in these pixels
MIN_EXTENT_M = 0.05
IOU_MATCH = 0.30                 # a query box must overlap a candidate at least this much
AMBIGUOUS_MARGIN = 0.15          # two candidates this close in IoU are not distinguishable

SCENE_ROOT = "/data/behavior-data/behavior-1k-assets/scenes"


@dataclass
class Detection:
    det: str                     # view-local handle, e.g. "d3"
    bbox: tuple                  # (x0, y0, x1, y1) in image pixels
    category: str
    score: float
    key: str                     # RESOLVES TO: sim/scope identity. Never sent to the policy.
    depth: float                 # camera-space distance, for occlusion ordering


@functools.lru_cache(maxsize=8)
def scene_furniture(scene_model: str) -> tuple:
    """Every non-`stuff` object in a BEHAVIOR scene as (category, pos, extent).

    These are the distractors, and they are REAL -- read from the shipped scene json,
    not invented. Only the task objects' positions may come from a generated layout.
    """
    hits = glob.glob(os.path.join(SCENE_ROOT, scene_model, "json",
                                  f"{scene_model}_best.json"))
    if not hits:
        return ()
    d = json.load(open(hits[0]))
    reg = d.get("state", {}).get("registry", {}).get("object_registry", {})
    init = d.get("objects_info", {}).get("init_info", {})
    out = []
    for name, info in init.items():
        args = info.get("args", {})
        cat = args.get("category")
        if not cat or cat in STUFF_CATEGORIES:
            continue
        pos = (reg.get(name) or {}).get("root_link", {}).get("pos")
        if not pos:
            continue
        # `scale` is a per-axis factor, not a size, and the shipped json carries no
        # bbox. Using it directly as metres is crude and deliberately so: overlap and
        # occlusion only need RELATIVE size to be roughly right, and a wrong absolute
        # scale shifts every box together. Flagged rather than hidden.
        sc = args.get("scale") or [0.3, 0.3, 0.3]
        ext = tuple(max(float(v) / 2.0, MIN_EXTENT_M) for v in sc)
        out.append((cat, (float(pos[0]), float(pos[1]), float(pos[2])), ext))
    return tuple(out)


def _project(view, pos, ext) -> tuple | None:
    """AABB -> 2D box in the virtual image, or None if it is not in front of the camera.

    A pinhole with the same horizontal half-angle the FoV gate uses, so `in_view` and
    a non-empty projection agree. Corners are projected and bounded rather than
    projecting the centre and guessing a size, which is what makes a wide object near
    the frame edge behave correctly.
    """
    from rlinf.envs.behavior.viewpoint import (
        CAMERA_HEIGHT_M,
        FOV_HALF_ANGLE_RAD,
    )

    fx = (IMAGE_W / 2) / math.tan(FOV_HALF_ANGLE_RAD)
    cy_pitch = view.pitch
    cos_y, sin_y = math.cos(-view.yaw), math.sin(-view.yaw)
    cos_p, sin_p = math.cos(-cy_pitch), math.sin(-cy_pitch)

    xs, ys, depths = [], [], []
    for dx in (-ext[0], ext[0]):
        for dy in (-ext[1], ext[1]):
            for dz in (-ext[2], ext[2]):
                wx = pos[0] + dx - view.x
                wy = pos[1] + dy - view.y
                wz = pos[2] + dz - CAMERA_HEIGHT_M
                # world -> camera: yaw about z, then pitch about the camera's right axis
                cx = wx * cos_y - wy * sin_y
                cyy = wx * sin_y + wy * cos_y
                fwd = cx * cos_p - wz * sin_p
                up = cx * sin_p + wz * cos_p
                if fwd <= 0.05:
                    continue
                xs.append(IMAGE_W / 2 - fx * cyy / fwd)
                ys.append(IMAGE_H / 2 - fx * up / fwd)
                depths.append(fwd)
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(float(IMAGE_W), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(IMAGE_H), max(ys))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (int(x0), int(y0), int(x1), int(y1)), min(depths)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect(view, entries) -> list:
    """Project @entries -> visible detections, nearest first, occluded ones dropped.

    @entries is an iterable of (key, category, pos, extent). The key is how the caller
    maps a detection back to a simulator entity and is never exposed.

    Occlusion is a coarse but real test: a farther box whose area is almost entirely
    covered by nearer boxes is not reported. That is what makes discovery depend on
    where you stand rather than on a gate someone has to describe.
    """
    from rlinf.envs.behavior.viewpoint import FOV_RANGE_M

    projected = []
    for key, cat, pos, ext in entries:
        # Range first: the frustum alone would report the far wall of the house, and
        # the first run of this did exactly that -- 224 detections from one viewpoint.
        if math.hypot(pos[0] - view.x, pos[1] - view.y) > FOV_RANGE_M:
            continue
        pr = _project(view, pos, ext)
        if pr is None:
            continue
        box, depth = pr
        projected.append((depth, key, cat, box))
    projected.sort()

    out, occluders = [], []
    for i, (depth, key, cat, box) in enumerate(projected):
        covered = max((_iou(box, nb) for nb in occluders), default=0.0)
        if covered > 0.85:
            continue                    # hidden behind something nearer
        area = (box[2] - box[0]) * (box[3] - box[1])
        out.append(Detection(
            det=f"d{len(out) + 1}", bbox=box, category=cat,
            score=round(min(1.0, 0.35 + 0.65 * math.sqrt(area / (IMAGE_W * IMAGE_H))), 2),
            key=key, depth=round(depth, 2)))
        occluders.append(box)
    return out


def resolve(query: str, detections: list):
    """Selection. Returns (Detection, None) or (None, refusal_reason).

    @query is either a view-local handle (`"d3"`) or a box (`"118,96,171,148"`).
    Refusals are the point of this function -- see `0815 - selection design`. A
    selection mechanism that always succeeds is the object-name oracle wearing a
    different syntax.
    """
    q = (query or "").strip()
    if not q:
        return None, "no selection given"

    by_det = {d.det: d for d in detections}
    if q in by_det:
        return by_det[q], None
    if q.startswith("d") and q[1:].isdigit():
        return None, f"{q} is not in the current view; observe again"

    parts = [p for p in q.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) != 4:
        return None, ("selection must be a detection handle from the current observation "
                      "(e.g. d3) or a box x0,y0,x1,y1")
    try:
        box = tuple(float(p) for p in parts)
    except ValueError:
        return None, "box coordinates must be numbers"

    scored = sorted(((_iou(box, d.bbox), d) for d in detections),
                    key=lambda t: -t[0])
    if not scored or scored[0][0] < IOU_MATCH:
        return None, "nothing in view matches that region"
    if len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUOUS_MARGIN:
        cats = {scored[0][1].category, scored[1][1].category}
        return None, (f"ambiguous: {len(scored)} objects overlap that region "
                      f"({', '.join(sorted(cats))}); move or look closer")
    return scored[0][1], None
