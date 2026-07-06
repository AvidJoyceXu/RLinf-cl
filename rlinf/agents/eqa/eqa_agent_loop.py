# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""EQAAgentLoopWorker — agent loop for HM-EQA with Habitat-Sim tools.
NOTE: 整个系统的智能体循环控制器，负责协调 LLM 生成、工具调用和奖励计算

pre_process_query()          初始化 episode，建立会话
        ↓
generate_llm_response()  ←──┐  LLM 生成一次工具调用
        ↓                   │
generate_tool_response() ───┘  执行工具，判断是否继续
        ↓
post_process_query()         关闭 episode，计算奖励

Mirrors searchr1 (rlinf/agents/searchr1/searchr1_agent_loop.py) but:
  * tool calls share a session_id == episode_id so the tool worker can route
    each request to the right Habitat episode handle;
  * an optional `scripted_policy` mode bypasses `self.generate` to emit canned
    tool-call tokens — drives end-to-end smoke runs at M2 before training.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.algorithms.rewards.eqa import (
    compute_score,
    compute_staged_rewards,
    extract_submitted_letter,
)
from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


_SCRIPTED_TURNS = [
    '<tool_call>{"name": "observe", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "submit_answer", "arguments": {"answer": "A"}}</tool_call>',
]


_SYSTEM_PROMPT = (
    "You are an embodied agent exploring a 3D indoor scene to answer a "
    "multiple-choice question. You act over multiple turns by calling exactly "
    "one tool per turn. First explore the scene (e.g. `observe`, `go_to_object`, "
    "`look_at`, `query_object`) to gather evidence, recording findings with "
    "`wm_write_node` and recalling them with `wm_query`. When confident, call "
    "`submit_answer` with the letter (A/B/C/D) of the correct option."
)


def _eqa_tool_schemas() -> list[dict]:
    """Tool schemas for the Qwen native `tools=` chat template; matches the
    eqa-qwen parser / HabitatEQAToolWorker contract. Imported lazily so the
    agent-loop worker does not hard-depend on spatialcode at module import."""
    from spatialcode.embodied.eqa_episode import tool_schemas

    return list(tool_schemas())


def _build_prompt_messages(spec: dict) -> list[dict]:
    options = spec.get("options") or {}
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = f"Question: {spec.get('question', '')}\nOptions:\n{options_text}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class EQAAgentLoopWorker(MultiAgentLoopWorker):
    """Agent loop for the EQA task — Qwen2.5-VL emits <tool_call> JSON.

    `answer` field of each dataset row is expected to be a JSON string carrying
    the full EpisodeSpec (scene_id, scene_glb, dataset_config, navmesh, wm_path,
    init_xz, init_yaw, question, answer, options). The HM-EQA loader serializes
    rows in this shape — see spatialcode/embodied/hm_eqa_loader.py.
    """

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.runner.seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)
        self.scripted_policy: bool = bool(cfg.agentloop.get("scripted_policy", False))
        # When set, every finished episode is appended (as one JSON line) to
        # {trace_dump_dir}/trace_pid{pid}.jsonl for offline inspection of the
        # model's skill sequence / reward. Off (None) by default.
        self.trace_dump_dir: str | None = cfg.agentloop.get("trace_dump_dir", None)
        # Reward shaping: `terminal` (compute_score, one scalar on every turn) or
        # `staged` (compute_staged_rewards, per-turn process credit + dominant
        # terminal). The `− staged` ablation is a one-line config flip (doc 01 §3.3).
        reward_cfg = cfg.get("reward", {}) or {}
        self.reward_type = str(reward_cfg.get("type", "terminal"))
        shaping = reward_cfg.get("shaping", {}) or {}
        from omegaconf import OmegaConf
        self.reward_shaping = (
            OmegaConf.to_container(shaping, resolve=True)
            if OmegaConf.is_config(shaping) else dict(shaping)
        )

        assert self.toolcall_parser is not None, (
            "EQAAgentLoopWorker requires agentloop.toolcall_parser (e.g. 'eqa-qwen')"
        )
        if self.cfg.runner.task_type != "reasoning_eval" and not self.scripted_policy:
            assert self.cfg.algorithm.recompute_logprobs, (
                "EQA agent must use recompute_logprobs (tool insertions re-tokenize)"
            )

    def _dump_trace(self, record: dict) -> None:
        if not self.trace_dump_dir:
            return
        os.makedirs(self.trace_dump_dir, exist_ok=True)
        path = os.path.join(self.trace_dump_dir, f"trace_pid{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- lifecycle

    async def pre_process_query(
        self, prompt_ids: list[int], answer: str
    ) -> tuple[list[int], dict[str, Any]]:
        spec_dict = _parse_spec(answer)
        episode_id = uuid4().hex

        # Open the episode on the tool worker before any tool dispatch.
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map["observe"]]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(
                session_id=episode_id,
                request_type="session_start",
                tool_name=None,
                tool_args=spec_dict,
            ),
            async_op=True,
        ).async_wait()
        _ = await self.tool_worker_output_channel.get(
            episode_id, async_op=True
        ).async_wait()

        # The dataset row's `prompt` is just the bare question. For real
        # generation the model needs the system prompt + tool schema + options,
        # so rebuild the prompt from the spec and chat-template it. (Scripted
        # policy ignores prompt content, so this is harmless there too.) The
        # built prompt must fit within max_prompt_len — bump data.max_prompt_length
        # in the config if the tool schema gets truncated.
        messages = _build_prompt_messages(spec_dict)
        # Pass tools via the Qwen2.5 native template — it injects the canonical
        # <tool_call>{"name":...,"arguments":...}</tool_call> format the model is
        # tuned to emit and that the eqa-qwen parser expects. (Hand-embedding the
        # schema in the system text made the model improvise a bare `name {args}`
        # format the parser couldn't read.)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=_eqa_tool_schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )
        built_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = built_ids[: self.max_prompt_len]

        ctx = {
            "episode_id": episode_id,
            "answer": str(spec_dict["answer"]),
            "options": dict(spec_dict.get("options") or {}),
            "question": spec_dict.get("question", ""),
            "turn": 0,
            "all_llm_response_ids": [],
            "problem_prompt_ids": copy.deepcopy(prompt_ids),
            "tool_trace": [],
        }
        return prompt_ids, ctx

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> MultiAgentLoopOutput:
        episode_id = generate_context["episode_id"]
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map["observe"]]
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

        response_text = self.tokenizer.decode(generate_context["all_llm_response_ids"])
        tool_trace = generate_context["tool_trace"]
        traj_info = {"tool_trace": tool_trace}

        if self.reward_type == "staged":
            # Per-turn process reward; terminal correctness lands on the last turn.
            staged = compute_staged_rewards(
                tool_trace,
                generate_context["answer"],
                generate_context["question"],
                generate_context["options"],
                **self.reward_shaping,
            )
            # Trajectory-level GRPO (advantage_mode: trajectory) needs ONE reward per
            # trajectory: the Megatron packer (io_struct.pack_traj_batch:1470) asserts
            # every turn of a packed trajectory shares a single reward. Assigning
            # distinct per-turn staged[k] crashed the actor at step 1 (AssertionError).
            # So assign the trajectory RETURN (sum of staged) uniformly to all turns —
            # keeps the shaping benefit (more coverage -> higher return -> restored
            # within-group variance) while satisfying the packer. Per-turn granularity
            # is unused by trajectory advantage; revisit only for token-level credit.
            reward_score = float(sum(staged))         # trajectory return
            for single_turn_output in output.single_turn_outputs:
                single_turn_output.reward_score = reward_score
        else:
            reward_score = compute_score(
                response_text=response_text,
                answer=generate_context["answer"],
                traj_info=traj_info,
            )
            for single_turn_output in output.single_turn_outputs:
                single_turn_output.reward_score = reward_score

        # Phase-1 logging (code-doc/0611 §A.3): interpretable per-trajectory signals.
        # `properly_ended` = the episode terminated by calling submit_answer (vs
        # running out of budget). Mark is_end on the terminal turn so the
        # `fraction_of_samples_properly_ended` metric becomes meaningful — verified
        # it only feeds that cosmetic metric and does NOT gate the loss mask
        # (response_mask is built from prompt/response lengths in io_struct).
        submitted = extract_submitted_letter(response_text, traj_info)
        properly_ended = bool(tool_trace) and tool_trace[-1].get("name") == "submit_answer"
        correct = submitted is not None and submitted == str(generate_context["answer"]).upper()
        if properly_ended and output.single_turn_outputs:
            output.single_turn_outputs[-1].is_end = True
        output.extra_fields["submitted"] = submitted is not None
        output.extra_fields["correct"] = bool(correct)

        output.extra_fields["llm_reward"] = reward_score
        output.extra_fields["response_text"] = response_text
        output.extra_fields["prompt_text"] = self.tokenizer.decode(
            generate_context.get("problem_prompt_ids", [])
        )
        output.extra_fields["turns"] = [
            {
                "input": self.tokenizer.decode(t.prompt_ids),
                "output": self.tokenizer.decode(t.response_ids),
            }
            for t in output.single_turn_outputs
        ]
        output.extra_fields["tool_trace"] = generate_context["tool_trace"]

        self._dump_trace(
            {
                "episode_id": episode_id,
                "question": generate_context["question"],
                "options": generate_context["options"],
                "answer_gt": generate_context["answer"],
                "num_turns": generate_context["turn"],
                "reward": reward_score,
                "response_text": response_text,
                "tool_trace": generate_context["tool_trace"],
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
        if generate_context["turn"] >= self.cfg.agentloop.max_turns:
            return False, None, None, None

        max_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) - len(problem_prompt_ids)
        )

        if self.scripted_policy:
            canned = _SCRIPTED_TURNS[min(generate_context["turn"], len(_SCRIPTED_TURNS) - 1)]
            llm_response_ids: list[int] = self.tokenizer.encode(canned, add_special_tokens=False)
        else:
            generate_result = await self.generate(
                turn_prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
            )
            llm_response_ids = generate_result["output_ids"]

        if len(llm_response_ids) > max_resp_len:
            llm_response_ids = llm_response_ids[:max_resp_len]
        llm_response_text = self.tokenizer.decode(llm_response_ids)

        if "</tool_call>" in llm_response_text:
            llm_response_text = llm_response_text.split("</tool_call>")[0] + "</tool_call>"
            llm_response_ids = self.tokenizer.encode(llm_response_text, add_special_tokens=False)
        elif "<tool_call>" in llm_response_text:
            # Generation stopped on the "</tool_call>" stop string, which the
            # rollout engine trims from the output. Re-append it so the parser's
            # open+close check passes.
            llm_response_text = llm_response_text + "</tool_call>"
            llm_response_ids = self.tokenizer.encode(llm_response_text, add_special_tokens=False)

        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
        )
        generate_context["all_llm_response_ids"] += llm_response_ids

        if len(llm_response_ids) == max_resp_len:
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
            return False, None

        # Dispatch each tool call against the same episode session.
        tool_responses: list[ToolResponse] = []
        for tr in tool_requests:
            tr.arguments = dict(tr.arguments or {})
            tr.arguments["_episode_id"] = generate_context["episode_id"]
            tool_responses.append(await self.tool_call_session(tr, generate_context))

        # Stop after a terminal tool (submit_answer). EQAEpisode.done is reflected
        # in meta_info; we check it on the tool_trace appended below.
        tool_messages = [{"role": "tool", "content": tr.text} for tr in tool_responses]
        tool_response_ids: list[int] = self.tokenizer.encode(
            tool_messages[0]["content"], add_special_tokens=False
        )
        max_tool_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
        )
        if len(tool_response_ids) > max_tool_resp_len:
            return False, None

        next_turn_prompt_ids = turn_prompt_ids + llm_response_ids + tool_response_ids
        if self.print_outputs:
            trace_prints.append(
                {
                    "prompt": self.tokenizer.decode(turn_prompt_ids),
                    "generate": llm_response_text,
                    "tool_resp": tool_messages,
                }
            )
        generate_context["turn"] += 1

        # If the most-recent tool was submit_answer, stop the loop now.
        last_trace = generate_context["tool_trace"][-1] if generate_context["tool_trace"] else {}
        if last_trace.get("name") == "submit_answer":
            return False, None

        return True, next_turn_prompt_ids

    # ------------------------------------------------------- session-aware call

    async def tool_call_session(
        self, tool_request: ToolRequest, generate_context: dict[str, Any]
    ) -> ToolResponse:
        """Run one tool call against the per-query episode session."""
        worker_name = self.tool_name_map.get(tool_request.name)
        if worker_name is None:
            return ToolResponse(text=json.dumps({"error": f"unknown_tool:{tool_request.name}"}))
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
        channel_response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()

        generate_context["tool_trace"].append({
            "name": tool_request.name,
            "arguments": {k: v for k, v in tool_request.arguments.items() if k != "_episode_id"},
            "result": channel_response.result,
            "meta": channel_response.meta_info,
        })
        if isinstance(channel_response.result, (list, dict)):
            text = json.dumps(channel_response.result)
        else:
            text = str(channel_response.result)
        return ToolResponse(text=text)

    # ----------------------------------------------------------- eval bookkeeping

    def gen_extra_fields(self, task_results, answer):
        if self.is_eval:
            llm_rewards, response_texts, prompt_texts, turns_list, traces = [], [], [], [], []
            for r in task_results:
                llm_rewards.append(r.extra_fields.get("llm_reward", 0.0))
                response_texts.append(r.extra_fields.get("response_text", ""))
                prompt_texts.append(r.extra_fields.get("prompt_text", ""))
                turns_list.append(r.extra_fields.get("turns", []))
                traces.append(r.extra_fields.get("tool_trace", []))
            return (
                None,
                {
                    "llm_reward": llm_rewards,
                    "response_text": response_texts,
                    "prompt_text": prompt_texts,
                    "turns": turns_list,
                    "tool_trace": traces,
                },
                {"answer": answer},
                {},
            )

        idx_to_sub_traj = []
        for r in task_results:
            for _ in r.single_turn_outputs:
                idx_to_sub_traj.append(0)
        # Per-trajectory correctness/submit flags → consumed by get_rollout_metrics
        # to log rollout/accuracy + rollout/submit_rate (code-doc/0611 §A.3).
        extra_fields_traj = {
            "correct": [bool(r.extra_fields.get("correct", False)) for r in task_results],
            "submitted": [bool(r.extra_fields.get("submitted", False)) for r in task_results],
        }
        return None, extra_fields_traj, None, {"idx_to_sub_traj": idx_to_sub_traj}

    def get_rollout_metrics(self, rollout_result) -> dict:
        """Aggregate interpretable EQA rollout metrics (accuracy, submit rate).

        Keys use the `__mean__/` stat prefix the agent-loop `post_process_metric`
        understands: each value is a (numerator, group_size) pair, summed across
        groups/workers into an overall ratio under the `rollout/` namespace.
        """
        if self.is_eval:
            return {}
        eft = rollout_result.extra_fields_traj or {}
        correct = eft.get("correct") or []
        submitted = eft.get("submitted") or []
        n = len(submitted)
        if n == 0:
            return {}
        return {
            "__mean__/rollout/accuracy": (float(sum(1 for c in correct if c)), float(n)),
            "__mean__/rollout/submit_rate": (float(sum(1 for s in submitted if s)), float(n)),
        }


def _parse_spec(answer: str) -> dict:
    """The dataset row's `answer` column carries the JSON-encoded EpisodeSpec."""
    if isinstance(answer, dict):
        return answer
    return json.loads(answer)


__all__ = ["EQAAgentLoopWorker"]
