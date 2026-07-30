# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Reward for the BEHAVIOR semantic tool-call agent.

The reward keys off the *real* BDDL goal predicate evaluation reported by the
tool worker (SemanticACI.goal_status), never off a leaked answer. Each tool
turn carries a ``meta`` dict shaped by BehaviorToolWorker:

    meta = {
        "num_satisfied": int,      # |satisfied atoms| after this turn
        "num_goal": int,           # |satisfied| + |unsatisfied| (constant)
        "is_success": bool,        # all atoms satisfied AND at least one exists
    }

Two shaping modes (config ``reward.type``):

* ``terminal`` — one scalar: +success_reward if the episode ended by calling
  ``end_task`` on a satisfied goal; a small negative if it never called
  ``end_task`` (budget-exhausted); 0 for a wrong/incomplete end. Mirrors the EQA
  terminal reward, and has the same failure mode under sampling (a cautious
  policy that rarely satisfies the full goal floors every group sample -> zero
  advantage variance), which is exactly what ``staged`` fixes.

* ``staged`` — dominant terminal PLUS a small process credit for the fraction of
  goal atoms satisfied at the end (partial progress). Budget-exhausted rollouts
  that satisfied *some* atoms earn graded reward, restoring within-group variance
  for GRPO. Terminal stays dominant (|success_reward| >> coverage_weight).
"""
from __future__ import annotations

from typing import Any


def _final_meta(tool_trace: list[dict]) -> dict:
    """The goal_status meta from the last tool turn that carried one."""
    for turn in reversed(tool_trace or []):
        meta = turn.get("meta") or {}
        if "num_goal" in meta:
            return meta
    return {}


def _ended_with_end_task(tool_trace: list[dict]) -> bool:
    return bool(tool_trace) and tool_trace[-1].get("name") == "end_task"


def compute_score(
    tool_trace: list[dict],
    success_reward: float = 1.0,
    no_end_penalty: float = -0.5,
) -> float:
    """Terminal BDDL reward.

    +success_reward  : ended via end_task AND the goal is satisfied.
     0.0             : ended via end_task but goal NOT satisfied (wrong stop).
     no_end_penalty  : never called end_task (ran out of turn budget).
    """
    meta = _final_meta(tool_trace)
    ended = _ended_with_end_task(tool_trace)
    if not ended:
        return float(no_end_penalty)
    return float(success_reward) if bool(meta.get("is_success")) else 0.0


def compute_staged_rewards(
    tool_trace: list[dict],
    success_reward: float = 1.0,
    no_end_penalty: float = -0.5,
    coverage_weight: float = 0.3,
) -> list[float]:
    """Per-turn reward list (terminal dominant + final-coverage shaping).

    Returned list is aligned to ``tool_trace``; the trajectory RETURN is the sum.
    The agent loop assigns the trajectory return uniformly across turns (the
    Megatron trajectory-advantage packer requires one reward per trajectory), so
    only the sum matters here — but we keep it per-turn so token-level credit is a
    later config flip, not a rewrite.
    """
    rewards = [0.0 for _ in (tool_trace or [])]
    if not tool_trace:
        return rewards

    meta = _final_meta(tool_trace)
    num_goal = int(meta.get("num_goal", 0) or 0)
    num_satisfied = int(meta.get("num_satisfied", 0) or 0)
    coverage = (num_satisfied / num_goal) if num_goal > 0 else 0.0

    # Process credit: fraction of atoms satisfied at the end, on the last turn.
    rewards[-1] += float(coverage_weight) * float(coverage)

    # Terminal (dominant), on the last turn.
    rewards[-1] += compute_score(
        tool_trace, success_reward=success_reward, no_end_penalty=no_end_penalty
    )
    return rewards


def episode_is_success(tool_trace: list[dict]) -> bool:
    return bool(_final_meta(tool_trace).get("is_success"))


def episode_properly_ended(tool_trace: list[dict]) -> bool:
    return _ended_with_end_task(tool_trace)


__all__ = [
    "compute_score",
    "compute_staged_rewards",
    "episode_is_success",
    "episode_properly_ended",
]
