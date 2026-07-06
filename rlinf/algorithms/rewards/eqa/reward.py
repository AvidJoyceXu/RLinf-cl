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
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
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


def _question_text(question: str, options: dict | None) -> str:
    """Lowercased Q+options blob for cheap substring relevance checks (dep-free)."""
    text = (question or "").lower()
    if options:
        text += " " + " ".join(str(v).lower() for v in options.values())
    return text


def _inspected_relevant(new_classes: Iterable[str], q_text: str) -> bool:
    return any(c and str(c).lower() in q_text for c in new_classes)


def compute_staged_rewards(
    tool_trace: list[dict],
    answer: str,
    question: str,
    options: dict | None = None,
    *,
    w_cov: float = 0.3,
    w_obj: float = 0.05,
    obj_cap: float = 1.0,
    w_inspect: float = 0.2,
    step_penalty: float = 0.01,
    term_correct: float = 1.0,
    term_wrong: float = 0.0,
    term_nosubmit: float = -0.5,
) -> list[float]:
    """One reward per turn. Intermediate turns get *process* credit (coverage gain,
    newly-observed objects, a relevant inspect); the terminal turn additionally gets
    the dominant correctness reward. Process signals come from the cumulative
    `meta` (= episode.traj_info()) attached to each tool_trace entry; we credit only
    the per-turn delta so a signal is rewarded once.

    Terminal stays dominant (|±1.0| ≫ shaping) so correctness is never overridden;
    shaping only guides exploration / breaks ties — the flat-curve fix (C4).
    """
    rewards: list[float] = []
    prev_cov, prev_obj = 0.0, 0
    prev_inspected: set[str] = set()
    q_text = _question_text(question, options)
    for e in tool_trace:
        m = e.get("meta") or {}
        cov = float(m.get("coverage", 0.0) or 0.0)
        obj = int(m.get("unique_observed_objects", 0) or 0)
        r = w_cov * max(0.0, cov - prev_cov)                              # exploration
        r += min(w_obj * max(0, obj - prev_obj), obj_cap)                 # new objects
        inspected = set(m.get("inspected_classes") or [])
        new_inspected = inspected - prev_inspected
        if e.get("name") == "inspect" and _inspected_relevant(new_inspected, q_text):
            r += w_inspect                                               # answerability
        r -= step_penalty                                                # efficiency
        rewards.append(r)
        prev_cov, prev_obj, prev_inspected = cov, obj, inspected
    submitted = extract_submitted_letter("", {"tool_trace": tool_trace})
    if submitted is None:
        term = term_nosubmit
    elif submitted == str(answer).upper():
        term = term_correct
    else:
        term = term_wrong
    if rewards:
        rewards[-1] += term
    return rewards


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

    def __init__(self, config: "DictConfig"):
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


__all__ = [
    "EQAReward",
    "compute_score",
    "compute_staged_rewards",
    "extract_submitted_letter",
]
