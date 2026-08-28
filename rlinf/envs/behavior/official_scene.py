"""Fail-closed scene composition for official BEHAVIOR 2026 recordings."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_inventory(path: str | Path) -> dict[str, Any]:
    """Return the immutable object/category inventory authored in a scene JSON."""
    template = Path(path)
    with template.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    init_info = raw.get("objects_info", {}).get("init_info", {})
    if not isinstance(init_info, dict) or not init_info:
        raise ValueError(f"scene template has no objects_info.init_info: {template}")
    categories = Counter()
    dataset_objects = 0
    for info in init_info.values():
        if info.get("class_name") == "DatasetObject":
            dataset_objects += 1
        category = info.get("args", {}).get("category")
        if category:
            categories[str(category)] += 1
    return {
        "path": str(template),
        "sha256": _sha256(template),
        "object_count": len(init_info),
        "dataset_object_count": dataset_objects,
        "categories": dict(sorted(categories.items())),
    }


def runtime_scene_inventory(scene) -> dict[str, Any]:
    """Describe objects actually instantiated after room filtering."""
    objects = list(scene.objects)
    categories = Counter(
        str(obj.category)
        for obj in objects
        if getattr(obj, "category", None) is not None
    )
    dataset_objects = sum(type(obj).__name__ == "DatasetObject" for obj in objects)
    return {
        "object_count": len(objects),
        "dataset_object_count": dataset_objects,
        "categories": dict(sorted(categories.items())),
    }


@dataclass(frozen=True)
class OfficialSceneContract:
    """Pinned template and room subset used by the official 2026 replay path."""

    activity: str
    scene_model: str
    room_instances: tuple[str, ...]
    metadata_path: str
    full_template: dict[str, Any]
    partial_room_template: dict[str, Any]

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["room_instances"] = list(self.room_instances)
        return value


def _challenge_root(instance_dir: Path) -> Path:
    for candidate in (instance_dir, *instance_dir.parents):
        if candidate.name == "2026-challenge-task-instances":
            return candidate
    raise ValueError(
        "official_rooms requires an instance directory inside "
        f"2026-challenge-task-instances, got {instance_dir}"
    )


def _room_instances(activity: str, metadata_path: Path) -> tuple[str, ...]:
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.reader(stream) if len(row) >= 3 and row[1] == activity
        ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one exact B100_task_misc row for {activity!r}, got {len(matches)}"
        )
    rooms = tuple(room.strip() for room in matches[0][2].splitlines() if room.strip())
    if not rooms or len(set(rooms)) != len(rooms):
        raise ValueError(f"invalid official room instances for {activity!r}: {rooms!r}")
    return rooms


def resolve_official_scene_contract(
    *,
    activity: str,
    scene_model: str,
    instance_dir: str | Path,
    activity_definition_id: int = 0,
    activity_instance_id: int = 0,
) -> OfficialSceneContract:
    """Resolve the exact full template plus official room-instance filter."""
    instance_dir = Path(instance_dir).resolve()
    challenge_root = _challenge_root(instance_dir)
    metadata_path = challenge_root / "metadata" / "B100_task_misc.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"official room metadata is missing: {metadata_path}")

    json_dir = challenge_root / "scenes" / scene_model / "json"
    stem = (
        f"{scene_model}_task_{activity}_{activity_definition_id}_"
        f"{activity_instance_id}_template"
    )
    full_template = json_dir / f"{stem}.json"
    partial_template = json_dir / f"{stem}-partial_rooms.json"
    for path in (full_template, partial_template):
        if not path.is_file():
            raise FileNotFoundError(f"official scene template is missing: {path}")

    return OfficialSceneContract(
        activity=activity,
        scene_model=scene_model,
        room_instances=_room_instances(activity, metadata_path),
        metadata_path=str(metadata_path),
        full_template=template_inventory(full_template),
        partial_room_template=template_inventory(partial_template),
    )


__all__ = [
    "OfficialSceneContract",
    "resolve_official_scene_contract",
    "runtime_scene_inventory",
    "template_inventory",
]
