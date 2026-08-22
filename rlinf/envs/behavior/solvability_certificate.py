"""Pure helpers for merging primitive and privileged visibility audits."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def classify_object(
    primitive_events: Sequence[str],
    tour_events: Sequence[str],
) -> str:
    """Assign one mutually exclusive outcome to a required task object."""
    primitive = set(primitive_events)
    tour = set(tour_events)
    if "nonvisual_substance" in tour:
        return "nonvisual_substance"
    if "missing_pose" in tour:
        return "missing_pose"
    if "missing_extent" in tour:
        return "missing_extent"
    if (
        "closed_container" in tour
        and "within_range" not in tour
        and "projectable" not in tour
    ):
        return "sealed_by_closed_container"
    if "visible" in tour:
        return (
            "found_by_primitive" if "visible" in primitive else "primitive_search_miss"
        )
    if "within_range" not in tour:
        return "outside_tour_range"
    if "projectable" not in tour:
        return "never_projectable"
    if "occluded" in tour:
        return "near_total_occlusion"
    return "tour_miss_unclassified"


def build_activity_certificate(
    activity: str,
    primitive: Mapping,
    tour: Mapping,
) -> dict:
    """Merge two searches and return a count-conserving activity certificate."""
    targets = sorted(set(primitive["target_objects"]) | set(tour["target_objects"]))
    primitive_audit = primitive.get("object_events", {})
    tour_audit = tour.get("object_events", {})
    objects = []
    for name in targets:
        primitive_events = sorted(primitive_audit.get(name, ()))
        tour_events = sorted(tour_audit.get(name, ()))
        objects.append(
            {
                "object": name,
                "outcome": classify_object(primitive_events, tour_events),
                "primitive_events": primitive_events,
                "tour_events": tour_events,
            }
        )
    counts = Counter(row["outcome"] for row in objects)
    if sum(counts.values()) != len(targets):
        raise AssertionError("object taxonomy does not conserve its denominator")
    return {
        "activity": activity,
        "source": tour.get("source", primitive.get("source")),
        "instance_source": tour.get(
            "instance_source", primitive.get("instance_source")
        ),
        "bddl_release": tour.get("bddl_release", primitive.get("bddl_release")),
        "asset_release": tour.get("asset_release", primitive.get("asset_release")),
        "instance_path": tour.get("instance_path", primitive.get("instance_path")),
        "targets": len(targets),
        "primitive_steps": primitive["steps"],
        "tour_steps": tour["steps"],
        "primitive_complete": primitive["complete"],
        "tour_complete": tour["complete"],
        "outcome_counts": dict(sorted(counts.items())),
        "objects": objects,
    }


__all__ = ["build_activity_certificate", "classify_object"]
