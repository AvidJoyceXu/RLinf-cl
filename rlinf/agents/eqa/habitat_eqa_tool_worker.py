# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""HabitatEQAToolWorker — in-process tool worker for the EQA agent.

Owns a pool of EQAEpisode handles (one Habitat-Sim process each). Tool calls
carry an `_episode_id` field in their arguments dict; the worker looks up the
episode and forwards the (name, args) pair to its dispatcher.

Lifecycle of one episode:
  1. EQAAgentLoopWorker.pre_process_query sends a "session_start" request with
     an EpisodeSpec payload; this worker resets a pooled episode and tags it
     with a session_id.
  2. Subsequent "execute" requests dispatch tools against that episode.
  3. A "session_end" request frees the episode back to the pool. (Optional —
     pre_process_query of the next query will free + reset implicitly.)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
)
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker

# Import the spatialcode embodied backend. Inside the rlinf-eqa docker image the
# repo is bind-mounted at /workspace/spatialcode/embodied (see
# code-doc/0512 - RLinf install for EQA/INSTALL.md §1.4); `/workspace` must be
# on PYTHONPATH for this import to resolve.
from spatialcode.embodied.eqa_episode import EQAEpisode, EpisodeSpec  # noqa: E402


class HabitatEQAToolWorker(ToolWorker):
    """In-process tool worker dispatching to a pool of EQAEpisode handles."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.tools.habitat_eqa
        self.pool_size: int = int(tcfg.get("pool_size", 1))
        self.render_rgb: bool = bool(tcfg.get("render_rgb", False))
        # Vision backend for the inspect() skill (B.1). "stub" = no VLM (text-only
        # ablation / CI); "qwen2.5-vl" = local Qwen2.5-VL-7B.
        self.vision_backend: str = str(tcfg.get("vision_backend", "stub"))
        self._episodes: dict[str, EQAEpisode] = {}
        self._idle: list[EQAEpisode] = []
        self.request_processor_task: asyncio.Task | None = None

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        super().init_worker(input_channel, output_channel)
        self._idle = [EQAEpisode(render_rgb=self.render_rgb, vision_backend=self.vision_backend) for _ in range(self.pool_size)]

    def start_server(self):
        loop = asyncio.get_running_loop()
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()
        for ep in list(self._episodes.values()) + list(self._idle):
            ep.close()
        self._episodes.clear()
        self._idle.clear()

    async def _process_requests(self):
        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            asyncio.create_task(self._handle_one(request))

    async def _handle_one(self, request: ToolChannelRequest):
        try:
            if request.request_type == "session_start":
                spec = _spec_from_args(request.tool_args)
                episode = self._acquire_episode(request.session_id)
                pose = episode.reset(spec)
                response = ToolChannelResponse(success=True, result={"pose": pose})
            elif request.request_type == "session_end":
                self._release_episode(request.session_id)
                response = ToolChannelResponse(success=True, result={"ok": True})
            elif request.request_type == "execute":
                episode = self._episodes.get(request.session_id)
                if episode is None:
                    response = ToolChannelResponse(success=False, result="unknown_session")
                else:
                    args = dict(request.tool_args or {})
                    args.pop("_episode_id", None)        # consumed by routing, not by dispatch
                    result = episode.dispatch(request.tool_name or "", args)
                    response = ToolChannelResponse(
                        success="error" not in result,
                        result=result,
                        meta_info=episode.traj_info(),
                    )
            else:
                response = ToolChannelResponse(success=False, result=f"unknown_request:{request.request_type}")
        except Exception as e:                            # noqa: BLE001
            response = ToolChannelResponse(success=False, result=f"{type(e).__name__}:{e}")

        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    # ---------------------------------------------------------------- pool mgmt

    def _acquire_episode(self, session_id: str) -> EQAEpisode:
        if session_id in self._episodes:
            return self._episodes[session_id]
        if not self._idle:
            # Pool exhausted — block until one frees up. M2 scope: just expand
            # the pool by one rather than block the event loop. Revisit at M3.
            self._idle.append(EQAEpisode(render_rgb=self.render_rgb, vision_backend=self.vision_backend))
        episode = self._idle.pop()
        self._episodes[session_id] = episode
        return episode

    def _release_episode(self, session_id: str) -> None:
        episode = self._episodes.pop(session_id, None)
        if episode is not None:
            episode.close()
            self._idle.append(episode)


def _spec_from_args(args: Any) -> EpisodeSpec:
    if isinstance(args, str):
        args = json.loads(args)
    args = dict(args or {})
    return EpisodeSpec(
        scene_id=args["scene_id"],
        scene_glb=args["scene_glb"],
        dataset_config=args["dataset_config"],
        navmesh=args.get("navmesh"),
        wm_path=args["wm_path"],
        init_xz=tuple(args["init_xz"]) if args.get("init_xz") else None,
        init_yaw=float(args.get("init_yaw", 0.0)),
        question=args["question"],
        answer=str(args["answer"]),
        options=dict(args.get("options") or {}),
    )


__all__ = ["HabitatEQAToolWorker"]
