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

# TRUE per-model bounding boxes, extracted from the asset USDs' `ig:nativeBB` -- the
# same attribute OmniGibson reads via `DatasetObject.native_bbox`. Nothing here is
# estimated or nominal.
#
# An earlier version used one hardcoded 15 cm half-extent for every task object, then a
# hand-written per-category table. Both were inventions and both were wrong in the same
# direction: a refrigerator, a floor and a sheet of plywood modelled at the same size
# are invisible across a room and are dropped by the occlusion test the moment anything
# stands in front of them. Per-MODEL matters too, not just per-category --
# bottle__of__detergent_1 and _2 in one activity are models qjkmhq and gkpmii with
# different boxes.
#
# Extraction: `scratchpad/extract_bbox.py`. Assets are Fernet-encrypted usdz; the key
# ships with the dataset and decryption needs only `cryptography`, so no Kit boot.
# 8662 models, 1829 categories, 0 failures.
NATIVE_BBOX_PATH = "/data/behavior-data/native_bbox.json"


@functools.lru_cache(maxsize=1)
def _native_bbox() -> dict:
    with open(NATIVE_BBOX_PATH) as f:
        return json.load(f)


@functools.lru_cache(maxsize=4)
def scope_models(instance_path: str) -> dict:
    """BDDL scope name -> asset model id, for one task instance.

    Chain: the instance's sibling `*_template.json` carries `metadata.task.inst_to_name`
    (scope -> scene object name) and `objects_info.init_info[name].args.model`. Both are
    written by the sampler, so this is recorded data rather than a guess.
    """
    # The template is PER ACTIVITY and lives one directory up; the tro_state files are
    # per instance inside `<scene>_task_<activity>_instances/`. Measured on the shipped
    # tree: 37 templates against 301 instances in one scene. Looking for a sibling
    # returns nothing, which is why an earlier version reported 0 of 31 official
    # instances as having a model map when in fact all of them do.
    tmpl = instance_path.replace("-tro_state.json", ".json")
    if not os.path.exists(tmpl):
        base = os.path.basename(tmpl)
        parent = os.path.dirname(os.path.dirname(instance_path))
        cands = sorted(glob.glob(os.path.join(parent, base)))
        if not cands:
            # instance index differs from the template's (…_0_187_ vs …_0_0_)
            head = base.split("_task_")[0] + "_task_" + \
                base.split("_task_")[1].rsplit("_", 2)[0]
            cands = sorted(glob.glob(os.path.join(parent, head + "_*_template.json")))
        if not cands:
            return {}
        tmpl = cands[0]
    with open(tmpl) as f:
        d = json.load(f)
    i2n = d.get("metadata", {}).get("task", {}).get("inst_to_name") or {}
    init = d.get("objects_info", {}).get("init_info", {})
    out = {}
    for inst, name in i2n.items():
        model = (init.get(name) or {}).get("args", {}).get("model")
        if model:
            out[inst] = model
    return out


def extent_for_model(model: str):
    """Half-extent from the asset's own native bbox, or None if the model is unknown."""
    bb = _native_bbox()["by_model"].get(model)
    return tuple(max(v / 2.0, MIN_EXTENT_M) for v in bb["bbox"]) if bb else None


def offset_for_model(model: str) -> tuple:
    """Recorded base-link -> bbox-centre offset, `ig:offsetBaseLink`.

    A scene or instance file records the BASE LINK pose, not the bbox centre. 89% of
    the 8662 assets have a non-zero offset and some are metres (`floors` is
    (6.72, 8.44, 0)), so treating the recorded pose as the centre puts every box in
    the wrong place. This is data we already had and were not using.
    """
    bb = _native_bbox()["by_model"].get(model)
    return tuple(bb["offset"]) if bb and "offset" in bb else (0.0, 0.0, 0.0)


def extent_for_category(category: str):
    """Half-extent from the per-category median of real model boxes, or None.

    Used only where a model id is genuinely unrecorded. Still measured data -- the
    median of that category's real assets -- never a hand-written number.
    """
    bb = _native_bbox()["by_category"].get(category)
    return tuple(max(v / 2.0, MIN_EXTENT_M) for v in bb) if bb else None


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
    not invented. Only the task objects' positions may come from a sampled instance.
    """
    hits = glob.glob(os.path.join(SCENE_ROOT, scene_model, "json",
                                  f"{scene_model}_best.json"))
    if not hits:
        return ()
    with open(hits[0]) as f:
        d = json.load(f)
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
        # TRUE size: the asset's own native bbox for THIS model, times the instance's
        # per-axis scale. `scale` alone is a factor, not a size.
        base = _native_bbox()["by_model"].get(str(args.get("model")))
        if base is None:
            continue                  # unknown model: omit rather than invent a size
        sc = args.get("scale") or [1.0, 1.0, 1.0]
        ext = tuple(max(float(b) * float(v) / 2.0, MIN_EXTENT_M)
                    for b, v in zip(base["bbox"], sc))
        off = base.get("offset") or (0.0, 0.0, 0.0)
        centre = tuple(float(pos[k]) + float(off[k]) * float(sc[k]) for k in range(3))
        out.append((cat, centre, ext))
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
                # CLAMP to the near plane rather than dropping corners behind the
                # camera. Dropping them silently collapses the box of any object that
                # straddles the camera: standing on a 12 m floor, the four corners in
                # front all sit at the same depth, the projected y-extent becomes a
                # line, and the floor you are standing on reports as invisible. That
                # cost 68 points of solvability before it was found.
                near = max(fwd, 0.05)
                xs.append(IMAGE_W / 2 - fx * cyy / near)
                ys.append(IMAGE_H / 2 - fx * up / near)
                depths.append(near)
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(float(IMAGE_W), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(IMAGE_H), max(ys))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (int(x0), int(y0), int(x1), int(y1)), min(depths)


def _frac_covered(a, b) -> float:
    """Fraction of box @a that box @b covers. Asymmetric, unlike IoU -- which is the
    point: a small object behind a large one is hidden, a large one behind a small one
    is not."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return ((ix1 - ix0) * (iy1 - iy0) / area_a) if area_a > 0 else 0.0


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
    from rlinf.envs.behavior.viewpoint import CAMERA_HEIGHT_M, FOV_RANGE_M

    projected = []
    for key, cat, pos, ext in entries:
        # Range to the NEAREST PART of the object, not to its centroid. A centroid
        # test drops anything large whose middle is far away -- measured: with proper
        # extents it made every `floor` invisible from the floor you were standing on,
        # and solvability fell from 77% to 9%.
        radius = math.hypot(ext[0], ext[1])
        if math.hypot(pos[0] - view.x, pos[1] - view.y) - radius > FOV_RANGE_M:
            continue
        pr = _project(view, pos, ext)
        if pr is None:
            continue
        box, depth = pr
        # A near-horizontal surface at or below foot level -- a floor, a lawn, a
        # driveway -- cannot hide what rests ON it, however near and large its
        # projection is. Without this the floor you are standing on is the nearest,
        # biggest box in the frame and occludes the entire scene: measured, it took
        # `bringing_in_wood` to 0 of 5 objects found with every object plainly in
        # range. Decided by geometry, not by category name.
        flat = (ext[2] <= 0.25 and max(ext[0], ext[1]) >= 1.0
                and pos[2] + ext[2] < CAMERA_HEIGHT_M - 0.5)
        projected.append((depth, key, cat, box, flat))
    projected.sort()

    out, occluders = [], []
    for depth, key, cat, box, flat in projected:
        # Occlusion is measured as the fraction of THIS box covered by a nearer one,
        # not as IoU. IoU is symmetric, so two large coplanar surfaces -- two floor
        # tiles of a single room, say -- score 1.0 against each other and the second
        # was being reported as hidden behind the first. Measured: this is why
        # `floor_1` was missing from almost every failed activity once true asset
        # sizes made floor boxes large.
        #
        # A nearer object of the SAME category is also not an occluder here: it is the
        # same surface continuing, and BDDL scopes one of them.
        # PARTIAL OCCLUSION. A segmentation stack returns a partially hidden object
        # with a smaller mask and lower confidence; it does not silently omit it. All
        # or nothing was our simplification, and it was measurably too harsh -- with
        # true asset sizes it removed most task objects from view. Only near-total
        # cover drops a detection now; anything less shrinks the reported box and the
        # score, which is both more faithful and still a real cost to the policy,
        # since a sliver of a box is harder to select against and easier to confuse.
        covered = max((_frac_covered(box, nb) for nb, ncat in occluders
                       if ncat != cat), default=0.0)
        if covered > 0.95:
            continue                    # essentially entirely hidden
        area = (box[2] - box[0]) * (box[3] - box[1]) * (1.0 - covered)
        out.append(Detection(
            det=f"d{len(out) + 1}", bbox=box, category=cat,
            score=round(min(1.0, (0.35 + 0.65 * math.sqrt(
                max(area, 1) / (IMAGE_W * IMAGE_H))) * (1.0 - covered)), 2),
            key=key, depth=round(depth, 2)))
        if not flat:
            occluders.append((box, cat))
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
