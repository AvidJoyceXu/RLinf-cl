"""Sim-free turn-budget policy shared by BEHAVIOR benchmark launchers.

The turn ceiling dominates the measured result, so choosing it is part of an
episode's experimental condition. This module keeps the rule pure and testable:
callers receive both the effective limit and the metadata required to reproduce it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class TurnBudget:
    """Effective episode budget and the inputs that produced it."""

    max_turns: int
    budget_source: str
    oracle_steps: int | None
    budget_k: float | None
    budget_cap: int | None
    requested_max_turns: int
    capped: bool

    def record(self) -> dict:
        """JSON-serializable metadata to store beside an episode."""
        return asdict(self)


def turn_budget_for(
    activity: str,
    *,
    flat_max_turns: int,
    oracle_steps: Mapping[str, int] | None = None,
    budget_k: float = 2.0,
    budget_cap: int | None = 0,
    allow_missing_oracle: bool = False,
) -> TurnBudget:
    """Return a flat or oracle-relative budget for one activity.

    Missing oracle entries fail closed by default. Silently mixing a flat fallback
    into an oracle-relative run creates episodes scored under different rules. A
    caller may opt into that fallback, but the resulting record says so explicitly.
    """
    if flat_max_turns < 1:
        raise ValueError(f"flat_max_turns must be >= 1, got {flat_max_turns}")
    # Hydra represents an intentionally disabled optional cap as ``null``.  Treat
    # that identically to the CLI's historical ``0`` sentinel before comparing or
    # recording it.  Keeping this normalization in the shared policy prevents the
    # API, standalone runner, and RL agent loop from drifting apart.
    normalized_cap = 0 if budget_cap is None else int(budget_cap)
    if budget_k <= 0:
        raise ValueError(f"budget_k must be > 0, got {budget_k}")
    if normalized_cap < 0:
        raise ValueError(f"budget_cap must be >= 0, got {budget_cap}")

    if not oracle_steps:
        return TurnBudget(
            max_turns=flat_max_turns,
            budget_source="flat",
            oracle_steps=None,
            budget_k=None,
            budget_cap=None,
            requested_max_turns=flat_max_turns,
            capped=False,
        )

    base = oracle_steps.get(activity)
    if base is None:
        if not allow_missing_oracle:
            raise KeyError(
                f"no oracle step count for {activity!r}; restrict the activity set "
                "to certified episodes or pass allow_missing_oracle=True"
            )
        return TurnBudget(
            max_turns=flat_max_turns,
            budget_source="flat_fallback",
            oracle_steps=None,
            budget_k=budget_k,
            budget_cap=normalized_cap or None,
            requested_max_turns=flat_max_turns,
            capped=False,
        )

    base = int(base)
    if base < 1:
        raise ValueError(f"oracle step count for {activity!r} must be >= 1, got {base}")
    requested = max(1, math.ceil(budget_k * base))
    effective = min(requested, normalized_cap) if normalized_cap else requested
    return TurnBudget(
        max_turns=effective,
        budget_source="oracle_relative",
        oracle_steps=base,
        budget_k=budget_k,
        budget_cap=normalized_cap or None,
        requested_max_turns=requested,
        capped=effective < requested,
    )


__all__ = ["TurnBudget", "turn_budget_for"]
