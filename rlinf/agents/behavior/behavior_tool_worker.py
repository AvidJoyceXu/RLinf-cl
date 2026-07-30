# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""BehaviorToolWorker — RLinf tool worker for the BEHAVIOR semantic-tool ACI.

Structural counterpart of ``HabitatEQAToolWorker``, with one difference that the
venv split forces: EQA holds ``EQAEpisode`` objects *in process*, while OmniGibson
cannot be imported here at all (IsaacSim pins py3.10; this worker runs in the
py3.11 trainer venv beside SGLang and Megatron). So each episode is leased from a
pool of ``rlinf/envs/behavior/env_server.py`` processes over localhost HTTP.

Everything else the agent framework sees is identical to the EQA worker: a
``session_start`` / ``execute`` / ``session_end`` request triple keyed by
``session_id``, with ``meta_info`` carrying the real BDDL goal status the reward
module scores off.

Pool semantics: one server holds exactly one booted activity and serves one
session at a time, so a lease is (activity → server) and concurrency is bounded by
how many servers are up. ``group_size`` GRPO samples of the same prompt therefore
serialize on one server unless several servers are launched for that activity;
size the launcher accordingly (``scripts/launch_behavior_env_servers.sh``).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
)
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class BehaviorToolWorker(ToolWorker):
    """Dispatches semantic-tool calls to a pool of BEHAVIOR env servers."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.tools.behavior
        # {activity: [base_url, ...]} — several servers per activity give the
        # group_size samples somewhere to run in parallel.
        self.servers: dict[str, list[str]] = {
            str(a): [str(u) for u in urls]
            for a, urls in (tcfg.get("servers") or {}).items()
        }
        self.request_timeout: float = float(tcfg.get("request_timeout_s", 300.0))
        self.lease_timeout: float = float(tcfg.get("lease_timeout_s", 600.0))

        self._free: dict[str, asyncio.Queue] = {}
        self._leased: dict[str, tuple[str, str]] = {}   # session_id -> (url, remote_sid)
        self._session: aiohttp.ClientSession | None = None
        self.request_processor_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- lifecycle

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        super().init_worker(input_channel, output_channel)
        assert self.servers, (
            "tools.behavior.servers is empty — launch env servers first with "
            "scripts/launch_behavior_env_servers.sh and point the config at them"
        )

    def start_server(self):
        loop = asyncio.get_running_loop()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout))
        for activity, urls in self.servers.items():
            q: asyncio.Queue = asyncio.Queue()
            for u in urls:
                q.put_nowait(u.rstrip("/"))
            self._free[activity] = q
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()
        if self._session is not None:
            asyncio.create_task(self._session.close())
            self._session = None
        self._leased.clear()
        self._free.clear()

    async def _process_requests(self):
        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            asyncio.create_task(self._handle_one(request))

    # ------------------------------------------------------------------ routing

    async def _handle_one(self, request: ToolChannelRequest):
        try:
            if request.request_type == "session_start":
                response = await self._start(request)
            elif request.request_type == "session_end":
                response = await self._end(request)
            elif request.request_type == "execute":
                response = await self._execute(request)
            else:
                response = ToolChannelResponse(
                    success=False, result=f"unknown_request:{request.request_type}")
        except asyncio.TimeoutError:
            # A hung server must not wedge the episode forever; the agent loop
            # sees a failed tool response and the trajectory ends with the
            # budget-exhausted reward rather than blocking the whole rollout.
            await self._release(request.session_id)
            response = ToolChannelResponse(success=False, result="env_timeout")
        except Exception as e:                                    # noqa: BLE001
            await self._release(request.session_id)
            response = ToolChannelResponse(success=False, result=f"{type(e).__name__}:{e}")

        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def _start(self, request: ToolChannelRequest) -> ToolChannelResponse:
        spec = request.tool_args
        if isinstance(spec, str):
            spec = json.loads(spec)
        spec = dict(spec or {})
        activity = spec["activity"]
        if activity not in self._free:
            return ToolChannelResponse(
                success=False, result=f"no_env_server_for_activity:{activity}")

        url = await asyncio.wait_for(self._free[activity].get(), self.lease_timeout)
        try:
            body = await self._post(f"{url}/session/start",
                                    {"instance_id": spec.get("instance_id")})
        except Exception:
            self._free[activity].put_nowait(url)      # never leak the lease
            raise
        self._leased[request.session_id] = (url, body["session_id"])
        # `goal_lines` comes back from the env because the ground goal atoms are
        # instance-resolved; keeping them out of the dataset keeps the RL dataset
        # builder sim-free.
        return ToolChannelResponse(success=True, result=body)

    async def _execute(self, request: ToolChannelRequest) -> ToolChannelResponse:
        lease = self._leased.get(request.session_id)
        if lease is None:
            return ToolChannelResponse(success=False, result="unknown_session")
        url, remote_sid = lease
        args = dict(request.tool_args or {})
        args.pop("_episode_id", None)                 # routing key, not a tool arg
        body = await self._post(f"{url}/session/{remote_sid}/tool",
                                {"name": request.tool_name or "", "arguments": args})
        return ToolChannelResponse(
            success=bool(body.get("payload", {}).get("ok", True)),
            result=body.get("payload"),
            meta_info=body.get("meta"),
        )

    async def _end(self, request: ToolChannelRequest) -> ToolChannelResponse:
        await self._release(request.session_id)
        return ToolChannelResponse(success=True, result={"ok": True})

    async def _release(self, session_id: str) -> None:
        lease = self._leased.pop(session_id, None)
        if lease is None:
            return
        url, remote_sid = lease
        try:
            await self._post(f"{url}/session/{remote_sid}/end", {})
        except Exception:
            pass                                      # returning the lease matters more
        for activity, urls in self.servers.items():
            if url in [u.rstrip("/") for u in urls]:
                self._free[activity].put_nowait(url)
                break

    async def _post(self, url: str, payload: dict) -> dict:
        assert self._session is not None, "start_server() not called"
        async with self._session.post(url, json=payload) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"env server {resp.status}: {body}")
            return body


def tool_names() -> list[str]:
    """Names the agent loop registers with the tool router.

    Sourced from the SFT schema so the RL action space cannot silently drift from
    the one the cold-start was trained on."""
    from rlinf.envs.behavior.sft_build import tool_schemas

    return [t["name"] for t in tool_schemas()]


__all__ = ["BehaviorToolWorker", "tool_names"]
