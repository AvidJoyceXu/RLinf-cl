"""Versioned BEHAVIOR task-instance sources and fail-closed selection.

The repository has three geometrically different populations on disk: the official
2025 challenge instances, the official 2026 instances released with BEHAVIOR-1K
v3.9.1, and locally sampled v3.7-era instances.  They must not be treated as one
anonymous search path: a BDDL scope from one release plus poses or model IDs from
another produces plausible-looking but invalid perception results.

This module is deliberately simulator-free so source policy can be regression-tested
without importing OmniGibson.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

DATA_ROOT = os.environ.get("BEHAVIOR_DATA_ROOT", "/data/behavior-data")
DEFAULT_SIM_SOURCE = "2025-official"
DEFAULT_LAYOUT_SOURCES = "2026-v3.9.1,2025-official"


@dataclass(frozen=True)
class InstanceSource:
    """One task-instance population and the release contract it carries."""

    name: str
    root: str
    bddl_release: str
    asset_release: str
    official: bool
    has_task_templates: bool
    detect_eligible: bool


def source_catalog(data_root: str | None = None) -> dict[str, InstanceSource]:
    """Return the known source catalog rooted at ``data_root``."""
    root = data_root or DATA_ROOT
    return {
        "2026-v3.9.1": InstanceSource(
            name="2026-v3.9.1",
            root=os.path.join(
                root,
                "releases/behavior-1k-v3.9.1/2026-challenge-task-instances/scenes",
            ),
            bddl_release="v3.9.1",
            asset_release="3.9.0",
            official=True,
            has_task_templates=True,
            detect_eligible=True,
        ),
        "2025-official": InstanceSource(
            name="2025-official",
            root=os.path.join(root, "2025-challenge-task-instances/scenes"),
            # The activity-level scope audit matches both the installed v3.7 BDDL
            # and v3.9.1. Simulator load is still required before quoting results.
            bddl_release="v3.7/v3.9.1-audited",
            asset_release="3.7-origin; 3.9.0-models-present",
            official=True,
            has_task_templates=True,
            detect_eligible=True,
        ),
        "local-v3.7": InstanceSource(
            name="local-v3.7",
            root=os.path.join(root, "behavior-1k-assets/scenes"),
            bddl_release="v3.7",
            asset_release="3.7.0rc23",
            official=False,
            # The current local files are tro_state-only. Without a template the
            # scope-name -> concrete model mapping cannot be recovered reliably.
            has_task_templates=False,
            detect_eligible=False,
        ),
    }


def selected_layout_sources(
    selection: str | None = None,
    *,
    data_root: str | None = None,
) -> tuple[InstanceSource, ...]:
    """Resolve an ordered, explicit source list for symbolic geometry lookup.

    Local samples are opt-in. Unknown names fail instead of silently falling back to
    an arbitrary directory.
    """
    raw = selection
    if raw is None:
        raw = os.environ.get("BEHAVIOR_INSTANCE_SOURCES", DEFAULT_LAYOUT_SOURCES)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("BEHAVIOR_INSTANCE_SOURCES selected no sources")
    catalog = source_catalog(data_root)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(
            f"unknown BEHAVIOR instance source(s): {unknown}; "
            f"known={sorted(catalog)}"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate BEHAVIOR instance source(s): {names}")
    return tuple(catalog[name] for name in names)


def simulator_source(
    name: str | None = None,
    *,
    data_root: str | None = None,
) -> InstanceSource:
    """Resolve the single source a booted simulator is allowed to consume."""
    selected = name or os.environ.get("BEHAVIOR_INSTANCE_SOURCE", DEFAULT_SIM_SOURCE)
    catalog = source_catalog(data_root)
    if selected not in catalog:
        raise ValueError(
            f"unknown BEHAVIOR_INSTANCE_SOURCE={selected!r}; known={sorted(catalog)}"
        )
    return catalog[selected]


@dataclass(frozen=True)
class InstanceLocation:
    """Resolved activity directory with provenance attached."""

    source: InstanceSource
    scene: str
    directory: str


def resolve_activity_location(
    activity: str,
    *,
    source_name: str | None = None,
    scene: str | None = None,
    data_root: str | None = None,
) -> InstanceLocation:
    """Find one activity under exactly one selected simulator source."""
    source = simulator_source(source_name, data_root=data_root)
    scenes = [scene] if scene else sorted(glob.glob(os.path.join(source.root, "*")))
    for scene_path in scenes:
        scene_name = scene or os.path.basename(scene_path)
        directory = os.path.join(
            source.root,
            scene_name,
            "json",
            f"{scene_name}_task_{activity}_instances",
        )
        if os.path.isdir(directory):
            return InstanceLocation(source, scene_name, directory)
    raise ValueError(
        f"no instance dir for activity={activity!r} under source={source.name!r} "
        f"root={source.root!r} scene={scene!r}"
    )


def assert_detect_eligible(source: InstanceSource) -> None:
    """Reject sources that cannot bind task scope names to asset model IDs."""
    if not source.detect_eligible or not source.has_task_templates:
        raise ValueError(
            f"instance source {source.name!r} is not detect-eligible: paired "
            "*_template.json files with scope-to-model bindings are required; "
            "regenerate the local samples with the synchronized release"
        )


__all__ = [
    "DEFAULT_LAYOUT_SOURCES",
    "DEFAULT_SIM_SOURCE",
    "InstanceLocation",
    "InstanceSource",
    "assert_detect_eligible",
    "resolve_activity_location",
    "selected_layout_sources",
    "simulator_source",
    "source_catalog",
]
