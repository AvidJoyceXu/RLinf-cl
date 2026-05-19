# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""EQA reward v0 — terminal correctness + step penalty + observation bonus.

Shape and constants follow code-doc/0511 - agentic RL for EQA - ACI and
pipeline decisions.md §4. Per-turn shaping is wired but trivially zero in M2;
the agent loop attaches `compute_score`'s scalar to every turn's reward_score.
"""
from __future__ import annotations

import random
import re
from typing import Iterable

from omegaconf import DictConfig


_ANSWER_RE = re.compile(r'"answer"\s*:\s*"?([A-Da-d])"?')


def extract_submitted_letter(response_text: str, traj_info: dict | None = None) -> str | None:
    """Pull the submitted MCQ letter from either the trajectory or the raw text."""
    if traj_info:
        for entry in reversed(traj_info.get("tool_trace") or []):
            if entry.get("name") == "submit_answer":
                ans = (entry.get("arguments") or {}).get("answer")
                if isinstance(ans, str) and ans.upper() in {"A", "B", "C", "D"}:
                    return ans.upper()
    m = list(_ANSWER_RE.finditer(response_text or ""))
    if m:
        return m[-1].group(1).upper()
    return None


def compute_score(
    response_text: str,
    answer: str,
    traj_info: dict | None = None,
    *,
    terminal_correct: float = 1.0,
    terminal_wrong: float = 0.0,
    budget_exhausted: float = -0.5,
    step_penalty: float = -0.01,
    observe_bonus: float = 0.05,
    observe_bonus_cap: float = 1.0,
) -> float:
    submitted = extract_submitted_letter(response_text, traj_info)
    steps = int((traj_info or {}).get("steps", 0))
    unique_obs = int((traj_info or {}).get("unique_observed_objects", 0))

    if submitted is None:
        terminal = budget_exhausted
    elif submitted == str(answer).upper():
        terminal = terminal_correct
    else:
        terminal = terminal_wrong

    obs_term = min(observe_bonus * unique_obs, observe_bonus_cap)
    return float(terminal + step_penalty * steps + obs_term)


class EQAReward:
    """Wrapper that satisfies the rule-based reward registry contract.
    TODO: 这里只能access response，无法访问`trajectory_info`。因此只有terminal reward, 没有obs_reward

    The reward worker passes `response` (list[str]) and `reference` (list[list[str]]
    or list[str]). For EQA each `reference[i]` is the ground-truth MCQ letter; the
    response is the concatenated LLM text for that trajectory. Per-trajectory
    `traj_info` is not threaded through the rule-based reward path, so this class
    operates on response_text alone — agent-loop-side `compute_score` (called from
    EQAAgentLoopWorker.post_process_query) handles the shaping with full traj info.
    """

    def __init__(self, config: DictConfig):
        self.scale = float(config.get("reward_scale", 1.0))
        self.random_print_percent = float(config.get("random_print_percent", 0.01))

    def get_reward(
        self, response: list[str], reference: list[Iterable[str]] | list[str]
    ) -> list[float]:
        rewards = []
        for resp, ref in zip(response, reference):
            answer = ref[0] if isinstance(ref, (list, tuple)) else ref
            r = compute_score(resp, str(answer), traj_info=None) * self.scale
            if random.random() < self.random_print_percent:
                print(f"[EQAReward] gt={answer} reward={r:.3f}")
            rewards.append(r)
        return rewards


__all__ = ["EQAReward", "compute_score", "extract_submitted_letter"]
