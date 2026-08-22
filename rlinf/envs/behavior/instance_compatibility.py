"""Simulator-free compatibility checks for BDDL scopes and sampled instances."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable, Optional

IGNORED_TRO_KEYS = {"robot_poses"}


def expand_problem_wildcards(problem: str, concrete_scope: Iterable[str]) -> str:
    """Expand v3.9 scene selectors using the paired template's concrete scope.

    ``category.n.01_*`` is a placeholder for additional scene objects beyond the
    explicitly required minimum. The official sampler expands it before creating
    ``inst_to_name``; the v3.7 symbolic evaluator does not know that syntax, so the
    synchronized compatibility bridge performs the same textual expansion from the
    authoritative template scope.
    """
    lines = problem.splitlines(keepends=True)
    concrete = set(concrete_scope)
    expansions: dict[str, list[str]] = {}

    for index, line in enumerate(lines):
        if "*" not in line or " - " not in line:
            continue
        declaration, synset = line.strip().split(" - ", 1)
        instances = declaration.split()
        wildcard = next((name for name in instances if name.endswith("_*")), None)
        if wildcard is None:
            continue
        explicit = set(instances) - {wildcard}
        candidates = sorted(
            name for name in concrete if name.startswith(f"{synset}_")
        )
        extras = [name for name in candidates if name not in explicit]
        expansions[wildcard] = extras
        lines[index] = line.replace(wildcard, " ".join(extras))

    if not expansions:
        return problem

    out = []
    for line in lines:
        wildcard = next((name for name in expansions if name in line), None)
        if wildcard is None:
            out.append(line)
            continue
        if " - " in line:
            # Already replaced in the first pass; this branch is defensive for an
            # unusual declaration containing two wildcard selectors.
            out.append(line.replace(wildcard, " ".join(expansions[wildcard])))
            continue
        stripped = line.strip()
        if not stripped.startswith("(inroom "):
            raise ValueError(f"unsupported wildcard use: {stripped}")
        out.extend(line.replace(wildcard, name) for name in expansions[wildcard])
    return "".join(out)


def template_path_for(tro_state_path: str) -> Path:
    """Return the template paired with a ``*-tro_state.json`` snapshot."""
    path = Path(tro_state_path)
    sibling = Path(str(path).replace("-tro_state.json", ".json"))
    if sibling.exists():
        return sibling

    # Official challenge snapshots are per-instance children of an ``*_instances``
    # directory, while one activity template lives in its parent. Its sampled index
    # may differ (e.g. snapshot ``_0_281`` against template ``_0_0``).
    parent = path.parent.parent
    base = sibling.name
    candidates = sorted(glob.glob(os.path.join(parent, base)))
    if not candidates and "_task_" in base:
        scene, task = base.split("_task_", 1)
        head = scene + "_task_" + task.rsplit("_", 2)[0]
        candidates = sorted(glob.glob(os.path.join(parent, head + "_*_template.json")))
    return Path(candidates[0]) if candidates else sibling


def instance_scope(tro_state_path: str) -> set[str]:
    """Read the declared scope, preferring the template over pose-bearing state.

    The template's ``inst_to_name`` is authoritative: a tro-state may omit metadata
    or non-pose-bearing entities, while the template records what task was sampled.
    """
    template_path = template_path_for(tro_state_path)
    if template_path.exists():
        with template_path.open() as stream:
            template = json.load(stream)
        mapping = template.get("metadata", {}).get("task", {}).get("inst_to_name")
        if mapping is not None:
            return set(mapping)

    with Path(tro_state_path).open() as stream:
        tro_state = json.load(stream)
    return set(tro_state) - IGNORED_TRO_KEYS


def scope_mismatch(
    expected_scope: Iterable[str],
    tro_state_path: str,
    ignored_scope: Iterable[str] = (),
) -> Optional[dict[str, list[str]]]:
    """Return incompatible concrete names, or ``None`` for a semantic match.

    Scene-object declarations ending in ``_*`` are selectors, not concrete objects.
    Templates may omit them or expand them to any number of numbered instances. Names
    in ``ignored_scope`` are also excluded; callers use this for BDDL future objects,
    which correctly do not exist in an initial geometric snapshot.
    """
    # Agent pose is sampled separately in ``robot_poses`` and is deliberately absent
    # from ``inst_to_name``. It is not evidence for or against task-object provenance.
    ignored = set(ignored_scope)
    ignored_wildcard_prefixes = {name[:-1] for name in ignored if name.endswith("_*")}
    declared = {
        name
        for name in expected_scope
        if not name.startswith("agent.n.01_") and name not in ignored
    }
    wildcard_prefixes = {name[:-1] for name in declared if name.endswith("_*")}
    expected = {name for name in declared if not name.endswith("_*")}
    actual = {
        name
        for name in instance_scope(tro_state_path)
        if not name.startswith("agent.n.01_")
        and name not in ignored
        and not any(name.startswith(prefix) for prefix in ignored_wildcard_prefixes)
        and not any(name.startswith(prefix) for prefix in wildcard_prefixes)
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return None
    return {"missing": missing, "extra": extra}
