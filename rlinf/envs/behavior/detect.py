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

`detect` changes TWO things at once against `fov` -- the scene's furniture starts being
reported, and objects stop being addressable by name -- so the measured -0.677 is a
joint effect. `scene_furniture` therefore feeds two consumers now: this module's
projection, and `symbolic_world`'s `fov_distract`, which reports the same furniture
BY NAME. See `0816 - separating distractors from reference resolution`.
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
STUFF_CATEGORIES = frozenset(
    {
        "walls",
        "ceilings",
        "downlight",
        "garden_light",
        "lawn_light",
        "ceiling_light",
        "wall_light",
        "roof",
        "window",
        "door_frame",
    }
)

IMAGE_W, IMAGE_H = 320, 240  # virtual sensor; boxes are reported in these pixels
MIN_EXTENT_M = 0.05
IOU_MATCH = 0.30  # a query box must overlap a candidate at least this much
AMBIGUOUS_MARGIN = 0.15  # two candidates this close in IoU are not distinguishable

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


@functools.lru_cache(maxsize=4)
def _load_native_bbox(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _native_bbox() -> dict:
    """Load the bbox artifact selected for this asset release.

    Both paths are environment-selectable because old and synchronized v3.9 assets
    coexist on the same host. Caching by resolved path prevents one process from
    accidentally reusing the old artifact after an explicit source switch.
    """
    return _load_native_bbox(native_bbox_path())


def native_bbox_path() -> str:
    """The version-pinned bbox artifact selected for this process."""
    return os.environ.get("BEHAVIOR_NATIVE_BBOX_PATH", NATIVE_BBOX_PATH)


def _scene_root() -> str:
    return os.environ.get("BEHAVIOR_ASSET_SCENE_ROOT", SCENE_ROOT)


@functools.lru_cache(maxsize=4)
def _template_data(instance_path: str) -> dict:
    """Load the activity template paired with one task-instance snapshot."""
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
            head = (
                base.split("_task_")[0]
                + "_task_"
                + base.split("_task_")[1].rsplit("_", 2)[0]
            )
            cands = sorted(glob.glob(os.path.join(parent, head + "_*_template.json")))
        if not cands:
            return {}
        tmpl = cands[0]
    with open(tmpl) as f:
        return json.load(f)


@functools.lru_cache(maxsize=4)
def scope_assets(instance_path: str) -> dict:
    """BDDL scope name -> concrete asset model and per-axis instance scale.

    Chain: the instance's activity template carries ``inst_to_name`` and the scene
    registry's ``init_info[name].args``. Both are sampler output.
    """
    d = _template_data(instance_path)
    i2n = d.get("metadata", {}).get("task", {}).get("inst_to_name") or {}
    init = d.get("objects_info", {}).get("init_info", {})
    out = {}
    for inst, name in i2n.items():
        args = (init.get(name) or {}).get("args", {})
        model = args.get("model")
        if model:
            scale = args.get("scale") or (1.0, 1.0, 1.0)
            out[inst] = {
                "model": str(model),
                "scale": tuple(float(value) for value in scale),
            }
    return out


def scope_models(instance_path: str) -> dict:
    """Backward-compatible scope -> model view over :func:`scope_assets`."""
    return {
        scope_name: asset["model"]
        for scope_name, asset in scope_assets(instance_path).items()
    }


@functools.lru_cache(maxsize=4)
def scope_fixed_base(instance_path: str) -> frozenset[str]:
    """Task scope objects whose concrete template instance is actually fixed-base.

    BDDL's ``sceneObject`` taxonomy is not a mobility flag: movable glasses and food
    processors carry it too.  ``args.fixed_base`` in the paired sampler template is
    the instance-level fact used by OmniGibson.
    """
    data = _template_data(instance_path)
    mapping = data.get("metadata", {}).get("task", {}).get("inst_to_name") or {}
    init = data.get("objects_info", {}).get("init_info", {})
    return frozenset(
        scope_name
        for scope_name, concrete_name in mapping.items()
        if bool(((init.get(concrete_name) or {}).get("args") or {}).get("fixed_base"))
    )


@functools.lru_cache(maxsize=4)
def scope_scene_names(instance_path: str) -> frozenset[str]:
    """Concrete scene registry objects already represented by task-scope entries."""
    d = _template_data(instance_path)
    mapping = d.get("metadata", {}).get("task", {}).get("inst_to_name") or {}
    return frozenset(str(name) for name in mapping.values() if name != "robot")


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


def world_bbox(
    base_pos: tuple,
    local_half_extent: tuple,
    *,
    local_offset: tuple = (0.0, 0.0, 0.0),
    scale: tuple = (1.0, 1.0, 1.0),
    orientation: tuple = (0.0, 0.0, 0.0, 1.0),
) -> tuple[tuple, tuple]:
    """Transform an asset-local bbox into a conservative world-axis AABB.

    BEHAVIOR records ``root_link.pos`` and an xyzw ``root_link.ori`` while
    ``ig:nativeBB`` and ``ig:offsetBaseLink`` are asset-local.  Scaling without
    rotating these quantities moves a rotated object's centre to the wrong side of
    its base link and leaves long objects' x/y dimensions on the wrong axes.
    """
    x, y, z, w = (float(value) for value in orientation)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    scaled_offset = tuple(float(local_offset[k]) * float(scale[k]) for k in range(3))
    scaled_half_extent = tuple(
        float(local_half_extent[k]) * abs(float(scale[k])) for k in range(3)
    )
    centre = tuple(
        float(base_pos[row])
        + sum(rotation[row][col] * scaled_offset[col] for col in range(3))
        for row in range(3)
    )
    extent = tuple(
        max(
            sum(abs(rotation[row][col]) * scaled_half_extent[col] for col in range(3)),
            MIN_EXTENT_M,
        )
        for row in range(3)
    )
    return centre, extent


def extent_for_category(category: str):
    """Half-extent from the per-category median of real model boxes, or None.

    Used only where a model id is genuinely unrecorded. Still measured data -- the
    median of that category's real assets -- never a hand-written number.
    """
    bb = _native_bbox()["by_category"].get(category)
    return tuple(max(v / 2.0, MIN_EXTENT_M) for v in bb) if bb else None


@dataclass
class Detection:
    det: str  # view-local handle, e.g. "d3"
    bbox: tuple  # (x0, y0, x1, y1) in image pixels
    category: str
    score: float
    key: str  # RESOLVES TO: sim/scope identity. Never sent to the policy.
    depth: float  # camera-space distance, for occlusion ordering


@dataclass(frozen=True)
class SceneObject:
    """One piece of scene furniture, entirely as recorded in the scene json."""

    name: str  # scene registry name, e.g. "bottom_cabinet_pkdnbu_0"
    category: str
    pos: tuple  # bbox centre = root_link pos + scaled offset
    extent: tuple  # half-extent
    room: str | None  # `args.in_rooms[0]`
    openable: bool  # the asset has articulated joints
    is_open: bool  # any joint away from rest, see JOINT_OPEN_EPS


# A scene object's `joint_pos` is recorded in the scene json. `_best.json` is a
# settled scene, so a shut cabinet reads ~0.002 rather than exactly 0. OmniGibson's
# own `Open` compares against a FRACTION of each joint's range, which we would have to
# read from the asset; an absolute threshold is a simplification, and it is named as
# one. It is still a measurement of a recorded quantity, not an invented state.
JOINT_OPEN_EPS = 0.05


@functools.lru_cache(maxsize=8)
def scene_furniture(scene_model: str) -> tuple:
    """Every non-`stuff` object in a BEHAVIOR scene, as `SceneObject`s.

    These are the distractors, and they are REAL -- read from the shipped scene json,
    not invented. Only the task objects' positions may come from a sampled instance.
    """
    hits = glob.glob(
        os.path.join(_scene_root(), scene_model, "json", f"{scene_model}_best.json")
    )
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
        st = reg.get(name) or {}
        root_link = st.get("root_link", {})
        pos = root_link.get("pos")
        if not pos:
            continue
        # TRUE size: the asset's own native bbox for THIS model, times the instance's
        # per-axis scale. `scale` alone is a factor, not a size.
        base = _native_bbox()["by_model"].get(str(args.get("model")))
        if base is None:
            continue  # unknown model: omit rather than invent a size
        sc = args.get("scale") or [1.0, 1.0, 1.0]
        local_ext = tuple(max(float(b) / 2.0, MIN_EXTENT_M) for b in base["bbox"])
        off = base.get("offset") or (0.0, 0.0, 0.0)
        centre, ext = world_bbox(
            tuple(float(value) for value in pos),
            local_ext,
            local_offset=tuple(float(value) for value in off),
            scale=tuple(float(value) for value in sc),
            orientation=tuple(
                float(value) for value in root_link.get("ori", (0, 0, 0, 1))
            ),
        )
        jp = st.get("joint_pos") or []
        rooms = args.get("in_rooms") or []
        out.append(
            SceneObject(
                name=name,
                category=cat,
                pos=centre,
                extent=ext,
                room=str(rooms[0]) if rooms else None,
                openable=bool(jp),
                is_open=any(abs(float(v)) > JOINT_OPEN_EPS for v in jp),
            )
        )
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

    # Sort projected boxes by their centre depth. Using the nearest of eight AABB
    # corners made every large cabinet/table win the painter's ordering across its
    # entire projected rectangle, even where that near corner was nowhere near the
    # small target ray. Centre depth is still an analytic approximation, but it does
    # not systematically turn large conservative AABBs into foreground billboards.
    centre_wx = pos[0] - view.x
    centre_wy = pos[1] - view.y
    centre_wz = pos[2] - CAMERA_HEIGHT_M
    centre_cx = centre_wx * cos_y - centre_wy * sin_y
    centre_fwd = centre_cx * cos_p - centre_wz * sin_p

    xs, ys, raw_depths = [], [], []
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
                raw_depths.append(fwd)
                # CLAMP to the near plane rather than dropping corners behind the
                # camera. Dropping them silently collapses the box of any object that
                # straddles the camera: standing on a 12 m floor, the four corners in
                # front all sit at the same depth, the projected y-extent becomes a
                # line, and the floor you are standing on reports as invisible. That
                # cost 68 points of solvability before it was found.
                near = max(fwd, 0.05)
                xs.append(IMAGE_W / 2 - fx * cyy / near)
                ys.append(IMAGE_H / 2 - fx * up / near)
    # Clamping is only valid for a box that STRADDLES the near plane. A box whose
    # eight corners are behind the camera is not visible; clamping all of them turns
    # it into a giant mirrored rectangle and lets a backwards-facing sweep "find" it.
    if not xs or max(raw_depths) <= 0.05:
        return None
    x0, x1 = max(0.0, min(xs)), min(float(IMAGE_W), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(IMAGE_H), max(ys))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (int(x0), int(y0), int(x1), int(y1)), max(centre_fwd, 0.05)


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


def _audit_mark(audit, key: str, event: str) -> None:
    """Accumulate detector-stage evidence without changing detector output."""
    if audit is not None:
        audit.setdefault(key, set()).add(event)


def detect(view, entries, audit=None, occlusion_exempt_pairs=frozenset()) -> list:
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
        _audit_mark(audit, key, "within_range")
        pr = _project(view, pos, ext)
        if pr is None:
            continue
        _audit_mark(audit, key, "projectable")
        box, depth = pr
        # A near-horizontal surface at or below foot level -- a floor, a lawn, a
        # driveway -- cannot hide what rests ON it, however near and large its
        # projection is. Without this the floor you are standing on is the nearest,
        # biggest box in the frame and occludes the entire scene: measured, it took
        # `bringing_in_wood` to 0 of 5 objects found with every object plainly in
        # range. Decided by geometry, not by category name.
        flat = (
            ext[2] <= 0.25
            and max(ext[0], ext[1]) >= 1.0
            and pos[2] + ext[2] < CAMERA_HEIGHT_M - 0.5
        )
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
        coverage = [
            (_frac_covered(box, nearer_box), nearer_key)
            for nearer_box, nearer_cat, nearer_key in occluders
            if nearer_cat != cat and (key, nearer_key) not in occlusion_exempt_pairs
        ]
        covered, occluder_key = max(coverage, default=(0.0, ""))
        if covered > 0.95:
            _audit_mark(audit, key, "occluded")
            _audit_mark(audit, key, f"occluded_by:{occluder_key}")
            continue  # essentially entirely hidden
        area = (box[2] - box[0]) * (box[3] - box[1]) * (1.0 - covered)
        out.append(
            Detection(
                det=f"d{len(out) + 1}",
                bbox=box,
                category=cat,
                score=round(
                    min(
                        1.0,
                        (0.35 + 0.65 * math.sqrt(max(area, 1) / (IMAGE_W * IMAGE_H)))
                        * (1.0 - covered),
                    ),
                    2,
                ),
                key=key,
                depth=round(depth, 2),
            )
        )
        _audit_mark(audit, key, "visible")
        if not flat:
            occluders.append((box, cat, key))
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
    # A POINT, which is what an open-vocabulary pointer (Molmo-style) emits and what a
    # human clicking the render produces. Resolved to the SMALLEST detection containing
    # it: with nested boxes -- a mug on a table, both under the cursor -- the smaller is
    # almost always the intended referent, and picking the larger would silently make
    # every click select the furniture.
    if len(parts) == 2:
        try:
            qx, qy = float(parts[0]), float(parts[1])
        except ValueError:
            return None, "point coordinates must be numbers"
        hits = [
            d
            for d in detections
            if d.bbox[0] <= qx <= d.bbox[2] and d.bbox[1] <= qy <= d.bbox[3]
        ]
        if not hits:
            return None, "nothing in view is at that point"
        hits.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        if len(hits) > 1:
            a0 = (hits[0].bbox[2] - hits[0].bbox[0]) * (
                hits[0].bbox[3] - hits[0].bbox[1]
            )
            a1 = (hits[1].bbox[2] - hits[1].bbox[0]) * (
                hits[1].bbox[3] - hits[1].bbox[1]
            )
            # Two candidates of near-identical size under one point are genuinely
            # ambiguous; refusing is the honest answer, per 0815 selection design R4.
            if a1 <= a0 * (1.0 + AMBIGUOUS_MARGIN):
                cats = {hits[0].category, hits[1].category}
                return None, (
                    f"ambiguous: {len(hits)} objects lie under that point "
                    f"({', '.join(sorted(cats))}); move or look closer"
                )
        return hits[0], None
    if len(parts) != 4:
        return None, (
            "selection must be a detection handle from the current observation "
            "(e.g. d3), a point x,y, or a box x0,y0,x1,y1"
        )
    try:
        box = tuple(float(p) for p in parts)
    except ValueError:
        return None, "box coordinates must be numbers"

    scored = sorted(((_iou(box, d.bbox), d) for d in detections), key=lambda t: -t[0])
    if not scored or scored[0][0] < IOU_MATCH:
        return None, "nothing in view matches that region"
    if len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUOUS_MARGIN:
        cats = {scored[0][1].category, scored[1][1].category}
        return None, (
            f"ambiguous: {len(scored)} objects overlap that region "
            f"({', '.join(sorted(cats))}); move or look closer"
        )
    return scored[0][1], None
