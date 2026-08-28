"""Audit whether same-looking detect handles require hidden BDDL identity.

The detector intentionally withholds scope names.  Two detections with the same
category and public state are therefore indistinguishable for text planning (score
and ``dN`` are selection tokens, not semantic identities).  This audit distinguishes
benign multi-solution cases from real interface defects by swapping the two scope
objects in the reference goal option and checking whether BDDL accepts the swapped
option as another fully-ground goal alternative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlinf.envs.behavior.symbolic_world import SymbolicWorld


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _swap(value: Any, left: str, right: str) -> Any:
    if isinstance(value, list):
        return [_swap(item, left, right) for item in value]
    if value == left:
        return right
    if value == right:
        return left
    return value


def _canonical_option(option: list) -> tuple[Any, ...]:
    """Canonicalize a BDDL conjunction without reordering atom arguments."""
    return tuple(sorted((_freeze(atom) for atom in option), key=repr))


def goal_interchangeable(
    goal_options: list[list], left: str, right: str, *, reference_index: int = 0
) -> bool:
    """Whether swapping two objects preserves an accepted complete goal option."""
    accepted = {_canonical_option(option) for option in goal_options}
    swapped = _swap(goal_options[reference_index], left, right)
    return _canonical_option(swapped) in accepted


def public_signature(detection: dict[str, Any]) -> tuple[Any, ...]:
    """Semantic detector fields visible to a text policy.

    Confidence and handle values remain intentionally excluded: they can choose a
    box, but do not tell the policy which BDDL individual that box represents.
    """
    return (
        detection.get("category"),
        _freeze(detection.get("states") or {}),
        detection.get("track"),
    )


def selector_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract selector alternatives from one reference trace without BDDL imports."""
    summary = document["summary"]
    trace = document["trace"]
    selected_scopes = {
        str(resolution["scope"]) for resolution in summary.get("handle_resolutions", [])
    }
    rows = []
    for resolution in summary.get("handle_resolutions", []):
        selected_turn = int(resolution["turn"])
        preceding = trace[selected_turn - 2] if selected_turn >= 2 else None
        result = (preceding or {}).get("result") or {}
        detections = result.get("detections", [])
        selected = next(
            (
                detection
                for detection in detections
                if detection.get("det") == resolution.get("handle")
            ),
            None,
        )
        handle_to_scope = {
            str(item.get("handle")): str(item.get("scope"))
            for item in ((preceding or {}).get("audit") or {}).get(
                "scope_detections", []
            )
        }
        if selected is None:
            rows.append(
                {
                    "turn": selected_turn,
                    "tool": resolution["tool"],
                    "selected_scope": str(resolution["scope"]),
                    "missing_selected_handle": True,
                    "same_public_signature_scopes": [],
                }
            )
            continue
        signature = public_signature(selected)
        alternatives = sorted(
            {
                handle_to_scope[str(detection.get("det"))]
                for detection in detections
                if public_signature(detection) == signature
                and str(detection.get("det")) in handle_to_scope
            }
        )
        selected_scope = str(resolution["scope"])
        rows.append(
            {
                "turn": selected_turn,
                "tool": resolution["tool"],
                "selected_scope": selected_scope,
                "missing_selected_handle": False,
                "public_signature": _freeze(selected),
                "same_public_signature_scopes": alternatives,
                "unselected_peers": sorted(set(alternatives) - selected_scopes),
            }
        )
    return rows


def audit_document(
    document: dict[str, Any], goal_options: list[list], *, source: str = ""
) -> dict[str, Any]:
    """Classify all same-public-signature selectors in one episode."""
    selectors = selector_rows(document)
    ambiguous = []
    for selector in selectors:
        selected = selector["selected_scope"]
        peers = [
            peer
            for peer in selector["same_public_signature_scopes"]
            if peer != selected
        ]
        if not peers:
            continue
        symmetric = [
            peer for peer in peers if goal_interchangeable(goal_options, selected, peer)
        ]
        identity_sensitive = sorted(set(peers) - set(symmetric))
        ambiguous.append(
            {
                **selector,
                "goal_symmetric_peers": symmetric,
                "identity_sensitive_peers": identity_sensitive,
            }
        )
    return {
        "source": source,
        "activity": document["summary"]["activity"],
        "split": document["summary"]["split"],
        "selectors": len(selectors),
        "missing_selected_handles": sum(
            row["missing_selected_handle"] for row in selectors
        ),
        "same_public_signature_ambiguous": len(ambiguous),
        "fully_goal_symmetric": sum(
            not row["identity_sensitive_peers"] for row in ambiguous
        ),
        "identity_sensitive": sum(
            bool(row["identity_sensitive_peers"]) for row in ambiguous
        ),
        "ambiguous_selectors": ambiguous,
    }


def audit_directory(path: Path) -> dict[str, Any]:
    episodes = []
    for trace_path in sorted(path.glob("*.json")):
        with trace_path.open() as stream:
            document = json.load(stream)
        summary = document["summary"]
        if not summary.get("policy_legal") or not summary.get("proper_completion"):
            continue
        world = SymbolicWorld(
            summary["activity"],
            instance_source_selection=summary.get("instance_source"),
        )
        episodes.append(
            audit_document(
                document,
                world.ground_goal_options(),
                source=trace_path.name,
            )
        )
    if not episodes:
        raise ValueError(f"no eligible reference trace JSON files found under {path}")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
        keys = (
            "selectors",
            "missing_selected_handles",
            "same_public_signature_ambiguous",
            "fully_goal_symmetric",
            "identity_sensitive",
        )
        return {
            "episodes": len(rows),
            **{key: sum(row[key] for row in rows) for key in keys},
        }

    by_split = {
        split: aggregate([row for row in episodes if row["split"] == split])
        for split in sorted({row["split"] for row in episodes})
    }
    return {
        "schema": "behavior-detect-goal-identity-v1",
        "public_equivalence": "same category and states; handle/score ignored",
        "goal_equivalence": (
            "swapping the selected and peer scope in reference ground option 0 "
            "produces any BDDL-accepted ground goal option"
        ),
        "eligible": aggregate(episodes),
        "by_split": by_split,
        "episodes": sorted(episodes, key=lambda row: (row["split"], row["activity"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit_directory(args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("eligible", "by_split")}, indent=2))


if __name__ == "__main__":
    main()
