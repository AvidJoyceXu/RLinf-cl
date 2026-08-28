# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""BehaviorAgentLoopWorker — online GRPO agent loop for BEHAVIOR task planning.

    pre_process_query()      lease an env server, reset an instance, build the prompt
            |
    generate_llm_response()  <-+   one <tool_call> per turn
            |                  |
    generate_tool_response() --+   dispatch through BehaviorToolWorker
            |
    post_process_query()     release the lease, score with the REAL BDDL goal status

Mirrors ``EQAAgentLoopWorker`` (same ``MultiAgentLoopWorker`` contract, same
session-keyed tool routing) with three BEHAVIOR-specific differences:

* the prompt is built by ``sft_build.build_prompt_messages`` and the tool schema by
  ``sft_build.tool_schemas`` — the *same functions* the SFT cold-start used, so the
  RL rollout distribution matches the one the policy was tuned on rather than
  merely resembling it;
* the goal atoms are instance-resolved, so they arrive from the env at
  ``session_start`` instead of riding in the dataset row;
* reward comes from ``algorithms/rewards/behavior.py``, which reads the real BDDL
  ``goal_status`` snapshot each tool response carries in ``meta_info``.

Nothing here is a leaked answer: the prompt states the goal (the task spec), the
observation reports state, and the reward is the simulator's own predicate
evaluation. See ``code-doc/human-plan/0710 ... §4`` for the honesty red lines.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.algorithms.rewards.behavior import (
    compute_score,
    compute_staged_rewards,
    episode_is_success,
    episode_properly_ended,
)
from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.envs.behavior.episode_contract import NUDGE
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


def _parse_episode_spec(answer: Any) -> dict:
    """Recover the episode spec {activity, scene, instance_id} from the dataset row.

    The loader does NOT hand back what the JSONL stored. `ReasoningDataset` wraps a
    string answer in a list (`reasoning.py:224`: "if answer is a string, convert it to
    a list"), so `answer` arrives as `['{"activity": ...}']`, never as the bare str.
    An earlier version tested `isinstance(answer, str)` and fell through to
    `dict(answer)`, which tried to read a list of one 89-char string as key/value
    pairs:

        ValueError: dictionary update sequence element #0 has length 89; 2 is required

    Accept every shape the loader can produce rather than the one shape the file has,
    and fail with the offending value rather than a length arithmetic error.
    """
    if isinstance(answer, (list, tuple)):
        if len(answer) != 1:
            raise ValueError(
                f"episode spec must be a single entry, got {len(answer)}: {answer!r}"
            )
        answer = answer[0]
    if isinstance(answer, str):
        answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise TypeError(
            f"episode spec must be a dict, got {type(answer).__name__}: {answer!r}"
        )
    missing = {"activity"} - answer.keys()
    if missing:
        raise ValueError(f"episode spec is missing {sorted(missing)}: {answer!r}")
    return answer


def _native_feedback_ids(
    tokenizer,
    feedback_role: str,
    feedback_text: str,
) -> list[int]:
    """Render a model/feedback boundary with the tokenizer's native chat format.

    The current prompt already ends at an assistant generation header and the
    generated token IDs follow it. Render an *empty* assistant message to derive
    only the suffix: assistant terminator, feedback wrapper, and next generation
    header. Re-tokenizing decoded model text is incorrect because arbitrary sampled
    token sequences need not survive decode/encode round trips bit-for-bit.
    """
    if feedback_role not in {"tool", "user"}:
        raise ValueError(f"feedback_role must be tool or user, got {feedback_role!r}")
    anchor = [{"role": "user", "content": "context-boundary-anchor"}]
    prompt_ids = tokenizer.apply_chat_template(
        anchor,
        tools=[],
        tokenize=True,
        add_generation_prompt=True,
    )
    rendered_ids = tokenizer.apply_chat_template(
        [
            *anchor,
            {"role": "assistant", "content": ""},
            {"role": feedback_role, "content": feedback_text},
        ],
        tools=[],
        tokenize=True,
        add_generation_prompt=True,
    )
    expected_prefix = list(prompt_ids)
    if list(rendered_ids[: len(expected_prefix)]) != expected_prefix:
        raise RuntimeError(
            "tokenizer chat template cannot derive a content-independent native "
            "feedback suffix"
        )
    return list(rendered_ids[len(expected_prefix) :])


class BehaviorAgentLoopWorker(MultiAgentLoopWorker):
    """Agent loop for BEHAVIOR semantic tool-call planning.

    Each dataset row's ``answer`` column is a JSON episode spec:
    ``{"activity": ..., "scene": ..., "instance_id": ...}`` — see
    ``rlinf/envs/behavior/build_rl_dataset.py``.
    """

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.runner.seq_length)
        self.max_new_tokens = int(
            self.cfg.algorithm.get("sampling_params", {}).get("max_new_tokens", 512)
        )
        if self.max_new_tokens <= 0 or self.max_new_tokens >= max_total_len:
            raise ValueError(
                "algorithm.sampling_params.max_new_tokens must be positive and "
                f"below runner.seq_length, got {self.max_new_tokens}/{max_total_len}"
            )
        self.max_context_len = max_total_len - self.max_new_tokens
        self.obs_mode = str(cfg.agentloop.get("obs_mode", "full"))
        self.selection_mode = str(cfg.agentloop.get("selection_mode", "handle"))
        if self.selection_mode not in ("handle", "bbox"):
            raise ValueError(
                "agentloop.selection_mode must be 'handle' or 'bbox', got "
                f"{self.selection_mode!r}"
            )
        # "nl"    -> the goal reaches the policy as a natural-language sentence
        # "atoms" -> as ground BDDL atoms (the ablation, and what SFT step 120 saw)
        self.goal_format = str(cfg.agentloop.get("goal_format", "nl"))
        assert self.goal_format in ("nl", "atoms"), self.goal_format
        self.trace_dump_dir: str | None = cfg.agentloop.get("trace_dump_dir", None)
        self.budget_k = cfg.agentloop.get("budget_k", None)
        self.budget_cap = cfg.agentloop.get("budget_cap", None)
        self.max_nudges = int(cfg.agentloop.get("max_nudges", 3))
        if self.max_nudges < 0:
            raise ValueError(
                f"agentloop.max_nudges must be nonnegative, got {self.max_nudges}"
            )
        self.context_format = str(cfg.agentloop.get("context_format", "native"))
        if self.context_format not in {"native", "flat"}:
            raise ValueError(
                "agentloop.context_format must be 'native' or 'flat', got "
                f"{self.context_format!r}"
            )

        reward_cfg = cfg.get("reward", {}) or {}
        self.reward_type = str(reward_cfg.get("type", "staged"))
        shaping = reward_cfg.get("shaping", {}) or {}
        from omegaconf import OmegaConf

        self.reward_shaping = (
            OmegaConf.to_container(shaping, resolve=True)
            if OmegaConf.is_config(shaping)
            else dict(shaping)
        )

        assert self.toolcall_parser is not None, (
            "BehaviorAgentLoopWorker requires agentloop.toolcall_parser "
            "(use 'eqa-qwen' — it is a format-level Qwen <tool_call> parser, "
            "not an EQA-specific one)"
        )
        if self.cfg.runner.task_type != "reasoning_eval":
            assert self.cfg.algorithm.recompute_logprobs, (
                "tool responses are spliced into the context between turns, which "
                "re-tokenizes the prefix — logprobs must be recomputed"
            )

    # ------------------------------------------------------------------ helpers

    def _tool_schemas(self) -> list[dict]:
        """Imported lazily: sft_build is sim-free, but keep the import local so
        the agent loop never grows a hard dependency on the env package.

        Camera-controlled observation modes add viewpoint actions. Using the core
        17-tool schema here while the API baseline used ``all_schemas`` made the two
        policies incomparable and left an RL policy unable to search at all.
        """
        from rlinf.envs.behavior.sft_build import all_schemas

        return list(
            all_schemas(
                self.obs_mode,
                selection_mode=self.selection_mode,
            )
        )

    def _dump_trace(self, record: dict) -> None:
        if not self.trace_dump_dir:
            return
        os.makedirs(self.trace_dump_dir, exist_ok=True)
        path = os.path.join(self.trace_dump_dir, f"trace_pid{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _tool_channel(self):
        return self.tool_channel_info_map[self.tool_name_map["observe"]]

    # --------------------------------------------------------------- lifecycle

    async def pre_process_query(
        self, prompt_ids: list[int], answer: str
    ) -> tuple[list[int], dict[str, Any]]:
        from rlinf.envs.behavior import sft_build as sb

        spec = _parse_episode_spec(answer)
        episode_id = uuid4().hex

        tool_channel_info = await self._tool_channel()
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(
                session_id=episode_id,
                request_type="session_start",
                tool_name=None,
                tool_args=spec,
            ),
            async_op=True,
        ).async_wait()
        started: ToolChannelResponse = await self.tool_worker_output_channel.get(
            episode_id, async_op=True
        ).async_wait()

        # Surface a failed session_start AS ITSELF. `_handle_one` reports failures as
        # ToolChannelResponse(success=False, result="<ExcType>: <msg>") -- a string,
        # not a dict. Coercing that to {} and then asserting on a missing key
        # reported "the env server returned no goal_nl ... the server predates
        # nl_goal.py" when the truth was an exception during env boot: a real error
        # replaced by a misleading one pointing at a component that no longer exists.
        if not started.success:
            raise RuntimeError(
                f"session_start failed for {spec['activity']} "
                f"(instance {spec.get('instance_id')}): {started.result}"
            )
        if not isinstance(started.result, dict):
            raise TypeError(
                f"session_start returned {type(started.result).__name__}, expected "
                f"dict: {started.result!r}"
            )

        info = started.result
        goal_lines = list(info.get("goal_lines") or [])
        goal_nl = info.get("goal_nl")
        # A reset whose goal already holds would score a free success. Flag it so
        # post_process can zero the trajectory instead of rewarding a no-op.
        contaminated = bool(info.get("contaminated"))

        from rlinf.envs.behavior.turn_budget import turn_budget_for

        oracle_turns = spec.get("oracle_turns")
        oracle_table = (
            {spec["activity"]: int(oracle_turns)} if oracle_turns is not None else None
        )
        budget = turn_budget_for(
            spec["activity"],
            flat_max_turns=int(self.cfg.agentloop.max_turns),
            oracle_steps=oracle_table,
            budget_k=float(self.budget_k) if self.budget_k is not None else None,
            budget_cap=int(self.budget_cap) if self.budget_cap else None,
            allow_missing_oracle=False,
        )

        if self.goal_format == "nl":
            assert goal_nl, (
                f"agentloop.goal_format='nl' but the tool worker returned no goal_nl "
                f"for {spec['activity']}. session_start succeeded, so this means "
                f"BehaviorEnv.start() omitted the key -- check nl_goal.render_goal_nl "
                f"for this activity. Keys returned: {sorted(info)}"
            )
        messages = sb.build_prompt_messages(
            spec["activity"],
            goal_lines,
            info.get("obs_mode", self.obs_mode),
            goal_nl=goal_nl if self.goal_format == "nl" else None,
            selection_mode=info.get("selection_mode", self.selection_mode),
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self._tool_schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )
        built_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(built_ids) > self.max_prompt_len:
            raise ValueError(
                f"rendered prompt has {len(built_ids)} tokens, above "
                f"data.max_prompt_length={self.max_prompt_len}; refusing to truncate"
            )
        if len(built_ids) > self.max_context_len:
            raise ValueError(
                f"rendered prompt has {len(built_ids)} tokens but the generation "
                f"context cap is {self.max_context_len}"
            )
        prompt_ids = built_ids

        ctx = {
            "episode_id": episode_id,
            "activity": spec["activity"],
            "instance_id": info.get("instance_id", spec.get("instance_id")),
            "instance_source": info.get("instance_source", spec.get("instance_source")),
            "geometry_instance_key": info.get(
                "geometry_instance_key", spec.get("geometry_instance_key")
            ),
            "goal_lines": goal_lines,
            "contaminated": contaminated,
            "turn": 0,
            "all_llm_response_ids": [],
            "problem_prompt_ids": copy.deepcopy(prompt_ids),
            "tool_trace": [],
            "encoded_turns": [],
            "context_compactions": 0,
            "budget": budget.record(),
            "max_turns": budget.max_turns,
            "consecutive_nudges": 0,
            "nudge_count": 0,
            "stop_reason": None,
        }
        return prompt_ids, ctx

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> MultiAgentLoopOutput:
        episode_id = generate_context["episode_id"]
        tool_channel_info = await self._tool_channel()
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(
                session_id=episode_id,
                request_type="session_end",
                tool_name=None,
                tool_args=None,
            ),
            async_op=True,
        ).async_wait()
        try:
            _ = await self.tool_worker_output_channel.get(
                episode_id, async_op=True
            ).async_wait()
        except Exception:
            pass

        tool_trace = generate_context["tool_trace"]
        if generate_context["contaminated"]:
            reward_score = 0.0  # pre-satisfied goal: no credit
        elif self.reward_type == "staged":
            # Trajectory-level GRPO needs ONE reward per trajectory (the Megatron
            # packer asserts every turn of a packed trajectory shares it), so the
            # trajectory RETURN is assigned uniformly across turns. Shaping still
            # works — more goal coverage means a higher return, which is what
            # restores within-group variance.
            reward_score = float(
                sum(compute_staged_rewards(tool_trace, **self.reward_shaping))
            )
        else:
            reward_score = float(compute_score(tool_trace, **self.reward_shaping))

        for single_turn_output in output.single_turn_outputs:
            single_turn_output.reward_score = reward_score

        success = (not generate_context["contaminated"]) and episode_is_success(
            tool_trace
        )
        properly_ended = episode_properly_ended(tool_trace)
        proper_completion = bool(success and properly_ended)
        if properly_ended and output.single_turn_outputs:
            output.single_turn_outputs[-1].is_end = True

        final_meta = next(
            (
                t.get("meta") or {}
                for t in reversed(tool_trace)
                if (t.get("meta") or {}).get("num_goal") is not None
            ),
            {},
        )
        top_level_coverage = final_meta.get("num_satisfied", 0) / max(
            final_meta.get("num_goal", 0) or 1, 1
        )
        coverage = float(final_meta.get("atom_coverage", top_level_coverage))

        output.extra_fields.update(
            {
                "success": bool(success),
                "properly_ended": bool(properly_ended),
                "proper_completion": proper_completion,
                "budget_exhausted": bool(
                    generate_context["turn"] >= generate_context["max_turns"]
                    and not properly_ended
                ),
                "budget": generate_context["budget"],
                "context_compactions": generate_context["context_compactions"],
                "context_format": self.context_format,
                "nudge_count": generate_context["nudge_count"],
                "stop_reason": generate_context["stop_reason"] or "unknown",
                "coverage": float(coverage),
                "top_level_coverage": float(top_level_coverage),
                "initial_atom_coverage": float(
                    final_meta.get("initial_atom_coverage", 0.0)
                ),
                "llm_reward": reward_score,
                "tool_trace": tool_trace,
                "response_text": self.tokenizer.decode(
                    generate_context["all_llm_response_ids"]
                ),
                "prompt_text": self.tokenizer.decode(
                    generate_context.get("problem_prompt_ids", [])
                ),
                "turns": [
                    {
                        "input": self.tokenizer.decode(t.prompt_ids),
                        "output": self.tokenizer.decode(t.response_ids),
                    }
                    for t in output.single_turn_outputs
                ],
            }
        )

        self._dump_trace(
            {
                "episode_id": episode_id,
                "activity": generate_context["activity"],
                "instance_id": generate_context["instance_id"],
                "instance_source": generate_context["instance_source"],
                "geometry_instance_key": generate_context["geometry_instance_key"],
                "goal_lines": generate_context["goal_lines"],
                "contaminated": generate_context["contaminated"],
                "num_turns": generate_context["turn"],
                "n_steps": generate_context["turn"],
                "model_turns": len(output.single_turn_outputs),
                "tool_call_rate": generate_context["turn"]
                / max(len(output.single_turn_outputs), 1),
                "context_compactions": generate_context["context_compactions"],
                "context_format": self.context_format,
                "nudge_count": generate_context["nudge_count"],
                "budget": generate_context["budget"],
                **generate_context["budget"],
                "stop_reason": generate_context["stop_reason"] or "unknown",
                "reward": reward_score,
                "success": bool(success),
                "proper_completion": proper_completion,
                "coverage": float(coverage),
                "top_level_coverage": float(top_level_coverage),
                "initial_atom_coverage": float(
                    final_meta.get("initial_atom_coverage", 0.0)
                ),
                "tool_trace": tool_trace,
            }
        )
        return output

    # ----------------------------------------------------------------- generate

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        if generate_context["turn"] >= generate_context["max_turns"]:
            generate_context["stop_reason"] = "max_turns"
            return False, None, None, None

        generate_result = await self.generate(
            turn_prompt_ids,
            sampling_params={"max_new_tokens": self.max_new_tokens},
        )
        llm_response_ids = generate_result["output_ids"][: self.max_new_tokens]
        llm_response_text = self.tokenizer.decode(llm_response_ids)

        # The rollout engine stops on "</tool_call>" and trims the stop string;
        # re-attach it (or cut after the first call) so the parser sees a balanced
        # block and a chatty turn cannot smuggle in a second tool call.
        if "</tool_call>" in llm_response_text:
            llm_response_text = (
                llm_response_text.split("</tool_call>")[0] + "</tool_call>"
            )
            llm_response_ids = self.tokenizer.encode(
                llm_response_text, add_special_tokens=False
            )
        elif "<tool_call>" in llm_response_text:
            llm_response_text = llm_response_text + "</tool_call>"
            llm_response_ids = self.tokenizer.encode(
                llm_response_text, add_special_tokens=False
            )

        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
        )
        generate_context["all_llm_response_ids"] += llm_response_ids

        if len(llm_response_ids) >= self.max_new_tokens:
            generate_context["stop_reason"] = "generation_length"
            generate_context["tool_trace"].append(
                {
                    "name": "__format__",
                    "arguments": {},
                    "result": {"ok": False, "reason": "generation_length"},
                    "meta": {},
                    "policy_error": "generation_length",
                }
            )
            return False, None, None, llm_output
        return True, llm_response_ids, llm_response_text, llm_output

    async def generate_tool_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
        llm_response_ids,
        llm_response_text,
    ):
        _, tool_requests = await self.toolcall_parser(llm_response_text)
        if not tool_requests:
            generate_context["consecutive_nudges"] += 1
            generate_context["nudge_count"] += 1
            generate_context["tool_trace"].append(
                {
                    "name": "__format__",
                    "arguments": {},
                    "result": {"ok": False, "reason": "no_tool_call"},
                    "meta": {},
                    "policy_error": "no_tool_call",
                }
            )
            if generate_context["consecutive_nudges"] > self.max_nudges:
                generate_context["stop_reason"] = "no_tool_call"
                return False, None
            return self._append_context_turn(
                generate_context,
                problem_prompt_ids,
                llm_response_ids,
                llm_response_text,
                NUDGE,
                "user",
            )
        generate_context["consecutive_nudges"] = 0

        # One tool per turn is the ACI contract the SFT data established; honour
        # only the first call if the model emits several.
        tr: ToolRequest = tool_requests[0]
        tr.arguments = dict(tr.arguments or {})
        tr.arguments["_episode_id"] = generate_context["episode_id"]
        tool_response = await self.tool_call_session(tr, generate_context)
        extra_calls = max(len(tool_requests) - 1, 0)
        if extra_calls and generate_context["tool_trace"]:
            generate_context["tool_trace"][-1]["format_violations"] = extra_calls
        generate_context["turn"] += 1

        is_continue, next_turn_prompt_ids = self._append_context_turn(
            generate_context,
            problem_prompt_ids,
            llm_response_ids,
            llm_response_text,
            tool_response.text,
            "tool",
        )
        if not is_continue:
            return False, None
        if self.print_outputs:
            trace_prints.append(
                {
                    "prompt": self.tokenizer.decode(turn_prompt_ids),
                    "generate": llm_response_text,
                    "tool_resp": tool_response.text,
                }
            )
        if (
            generate_context["tool_trace"]
            and generate_context["tool_trace"][-1].get("name") == "end_task"
        ):
            generate_context["stop_reason"] = "end_task"
            return False, None
        return True, next_turn_prompt_ids

    def _append_context_turn(
        self,
        generate_context: dict[str, Any],
        problem_prompt_ids: list[int],
        response_ids: list[int],
        response_text: str,
        feedback_text: str,
        feedback_role: str,
    ) -> tuple[bool, list[int] | None]:
        """Append one complete decision/feedback pair and compact at its boundary."""
        from spatialcode.sft_format import compact_complete_turns

        if self.context_format == "native":
            feedback_ids = _native_feedback_ids(
                self.tokenizer,
                feedback_role,
                feedback_text,
            )
        else:
            # Compatibility for the historical flat-continuation SFT. Base Qwen
            # policies need ``native`` so tool-role boundaries match pretraining.
            feedback_ids = self.tokenizer.encode(
                feedback_text, add_special_tokens=False
            )

        generate_context["encoded_turns"].append(
            (list(response_ids), list(feedback_ids))
        )
        try:
            next_prompt, first_kept = compact_complete_turns(
                problem_prompt_ids,
                generate_context["encoded_turns"],
                self.max_context_len,
            )
        except ValueError as exc:
            generate_context["stop_reason"] = "context_overflow"
            generate_context["tool_trace"].append(
                {
                    "name": "__harness__",
                    "arguments": {},
                    "result": {"ok": False, "reason": f"context_overflow:{exc}"},
                    "meta": {},
                }
            )
            return False, None
        if first_kept:
            generate_context["context_compactions"] += 1
        return True, next_prompt

    # ------------------------------------------------------- session-aware call

    async def tool_call_session(
        self, tool_request: ToolRequest, generate_context: dict[str, Any]
    ) -> ToolResponse:
        worker_name = self.tool_name_map.get(tool_request.name)
        if worker_name is None:
            # Hallucinated tool: report it like any other failed call so the
            # policy can recover, and record it for the misconception catalog.
            payload = {"ok": False, "reason": f"unknown_tool:{tool_request.name}"}
            generate_context["tool_trace"].append(
                {
                    "name": tool_request.name,
                    "arguments": tool_request.arguments,
                    "result": payload,
                    "meta": {},
                    "policy_error": "unknown_tool",
                }
            )
            return ToolResponse(text=json.dumps(payload))

        tool_channel_info = self.tool_channel_info_map[worker_name]
        session_id = generate_context["episode_id"]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(
                session_id=session_id,
                request_type="execute",
                tool_name=tool_request.name,
                tool_args=tool_request.arguments,
            ),
            async_op=True,
        ).async_wait()
        channel_response: ToolChannelResponse = (
            await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
        )
        if not channel_response.success and not isinstance(
            channel_response.result, dict
        ):
            raise RuntimeError(
                f"tool worker failed for {tool_request.name}: {channel_response.result}"
            )

        trace_turn = {
            "name": tool_request.name,
            "arguments": {
                k: v for k, v in tool_request.arguments.items() if k != "_episode_id"
            },
            "result": channel_response.result,
            "meta": channel_response.meta_info,
        }
        reason = str((channel_response.result or {}).get("reason", ""))
        if reason.startswith("bad_arguments:"):
            trace_turn["policy_error"] = "bad_arguments"
        generate_context["tool_trace"].append(trace_turn)
        text = (
            json.dumps(channel_response.result)
            if isinstance(channel_response.result, (list, dict))
            else str(channel_response.result)
        )
        return ToolResponse(text=text)

    # --------------------------------------------------------- eval bookkeeping

    def gen_extra_fields(self, task_results, answer):
        if self.is_eval:
            return (
                None,
                {
                    "llm_reward": [
                        r.extra_fields.get("llm_reward", 0.0) for r in task_results
                    ],
                    "success": [
                        r.extra_fields.get("success", False) for r in task_results
                    ],
                    "proper_completion": [
                        r.extra_fields.get("proper_completion", False)
                        for r in task_results
                    ],
                    "coverage": [
                        r.extra_fields.get("coverage", 0.0) for r in task_results
                    ],
                    "properly_ended": [
                        r.extra_fields.get("properly_ended", False)
                        for r in task_results
                    ],
                    "budget_exhausted": [
                        r.extra_fields.get("budget_exhausted", False)
                        for r in task_results
                    ],
                    "budget": [r.extra_fields.get("budget", {}) for r in task_results],
                    "stop_reason": [
                        r.extra_fields.get("stop_reason", "unknown")
                        for r in task_results
                    ],
                    "context_compactions": [
                        r.extra_fields.get("context_compactions", 0)
                        for r in task_results
                    ],
                    "tool_trace": [
                        r.extra_fields.get("tool_trace", []) for r in task_results
                    ],
                    "response_text": [
                        r.extra_fields.get("response_text", "") for r in task_results
                    ],
                },
                {"answer": answer},
                {},
            )

        idx_to_sub_traj = []
        for r in task_results:
            for _ in r.single_turn_outputs:
                idx_to_sub_traj.append(0)
        extra_fields_traj = {
            "success": [
                bool(r.extra_fields.get("success", False)) for r in task_results
            ],
            "proper_completion": [
                bool(r.extra_fields.get("proper_completion", False))
                for r in task_results
            ],
            "properly_ended": [
                bool(r.extra_fields.get("properly_ended", False)) for r in task_results
            ],
            "coverage": [
                float(r.extra_fields.get("coverage", 0.0)) for r in task_results
            ],
        }
        return None, extra_fields_traj, None, {"idx_to_sub_traj": idx_to_sub_traj}

    def get_rollout_metrics(self, rollout_result) -> dict:
        """Success / end / coverage rates under the ``rollout/`` namespace.

        ``rollout/coverage`` is the dial to watch on iteration 1: if success is
        flat at zero it is the only signal that says whether the policy is making
        partial progress or nothing at all."""
        if self.is_eval:
            return {}
        eft = rollout_result.extra_fields_traj or {}
        success = eft.get("success") or []
        proper = eft.get("proper_completion") or []
        ended = eft.get("properly_ended") or []
        coverage = eft.get("coverage") or []
        n = len(success)
        if n == 0:
            return {}
        return {
            "__mean__/rollout/success_rate": (
                float(sum(1 for s in success if s)),
                float(n),
            ),
            "__mean__/rollout/proper_completion_rate": (
                float(sum(1 for value in proper if value)),
                float(n),
            ),
            "__mean__/rollout/end_task_rate": (
                float(sum(1 for e in ended if e)),
                float(n),
            ),
            "__mean__/rollout/goal_coverage": (float(sum(coverage)), float(n)),
        }


__all__ = ["BehaviorAgentLoopWorker"]
