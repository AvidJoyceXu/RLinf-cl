# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""BehaviorToolWorker — RLinf tool worker for the BEHAVIOR semantic-tool ACI.

Holds a ``BehaviorEnv`` **in process** and calls it directly, exactly like
``HabitatEQAToolWorker`` holds its ``EQAEpisode`` objects. There is no HTTP hop.

An earlier version of this file leased episodes from a pool of
``rlinf/envs/behavior/env_server.py`` processes over localhost HTTP, because the
trainer was believed to need py3.11 while IsaacSim pins py3.10. That was wrong: the
trainer stack has a cp310 form for every component, so one venv runs both and the
RPC boundary had nothing to bridge. See
``code-doc/0730 - single-venv container/INSTALL.md`` §1. ``env_server.py`` survives as
a debugging surface (curl one tool call, drive one episode by hand) but is no longer
on the training path.

TWO OMNIGIBSON CONSTRAINTS SHAPE THIS FILE, and neither is about venvs:

* **One activity per process.** OmniGibson locks the ``HEADLESS`` macro at first boot,
  so a process that booted activity A cannot host activity B. This worker boots
  lazily on the first ``session_start`` and *refuses* a second activity rather than
  silently mutating the wrong scene. Multi-activity training therefore needs several
  worker ranks, one activity each, plus routing by activity — not yet implemented, and
  called out in ``_start`` rather than faked.
* **One session at a time.** A concurrent second session would mutate the first's
  scene. ``self._session_lock`` serialises them, so ``group_size`` rollouts of one prompt run
  sequentially within a rank and concurrency comes from rank count.

Every simulator call — boot, reset, tool dispatch — is blocking C++ that can run for
minutes. They go through ``asyncio.to_thread`` because this worker's event loop also
services the request channel; calling them inline would stall the loop and make a slow
reset look like a hung worker. The HTTP version got this for free by having the
simulator in a different process, so it is the one thing that got *harder* here.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
)
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class BehaviorToolWorker(ToolWorker):
    """Dispatches semantic-tool calls to an in-process BEHAVIOR environment."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg.tools.behavior
        self.obs_mode: str = str(tcfg.get("obs_mode", "full"))
        self.instances_per_activity: int = int(tcfg.get("instances_per_activity", 1))
        # Cameras off / partial scene load are the defaults for the same reason as in
        # env_server.py: the semantic ACI reads no pixels, and camera load cost ~13 of
        # ~16 boot minutes on a non-ray-tracing GPU.
        self.rgb: bool = bool(tcfg.get("rgb", False))
        self.partial_scene: bool = bool(tcfg.get("partial_scene", True))

        self._env: Any = None                  # BehaviorEnv, imported lazily
        self._booted_activity: str | None = None
        self._active_session: str | None = None
        # NOT `_lock`: the base Worker class owns `self._lock` (a threading.Lock,
        # used by `_get_collective_group`). Shadowing it with an asyncio.Lock made
        # the base's `with self._lock:` raise `AttributeError: __enter__`, because
        # asyncio.Lock implements __aenter__ and not __enter__ -- which killed the
        # request pump on its first channel recv and hung the whole run silently.
        self._session_lock: asyncio.Lock | None = None
        self.request_processor_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- lifecycle

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        super().init_worker(input_channel, output_channel)

    def start_server(self):
        """Start the request pump.

        Instrumented deliberately. The first run of this path hung with the worker
        idle, an EMPTY asyncio thread, and NOTHING in any log: `create_task` on a
        coroutine that raises immediately swallows the exception, because nobody
        awaits the task, and the agent loop then blocks forever on a tool response
        that will never come. A silent hang is the most expensive failure mode there
        is -- it looks identical to "still booting". So: log both ends, and attach a
        done-callback that surfaces whatever killed the pump.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # start_server may be dispatched off the event-loop thread; fall back
            # rather than dying silently.
            loop = asyncio.get_event_loop()
        self._session_lock = asyncio.Lock()
        self.request_processor_task = loop.create_task(self._process_requests())
        self.request_processor_task.add_done_callback(self._on_pump_exit)
        self.log_info(
            f"BehaviorToolWorker request pump started (loop={id(loop)}); "
            f"OmniGibson boots lazily on the first session_start"
        )

    def _on_pump_exit(self, task: asyncio.Task) -> None:
        """The pump should run for the whole job; any exit is a bug worth shouting about."""
        if task.cancelled():
            self.log_info("BehaviorToolWorker request pump cancelled (shutdown)")
            return
        exc = task.exception()
        if exc is not None:
            self.log_error(
                f"BehaviorToolWorker request pump DIED: {type(exc).__name__}: {exc}\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
        else:
            self.log_error("BehaviorToolWorker request pump exited without an error")

    def stop_server(self):
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()
        if self._env is not None:
            self._env.close()
            self._env = None
        self._booted_activity = None
        self._active_session = None

    async def _process_requests(self):
        self.log_info("BehaviorToolWorker pump: awaiting first request")
        first = True
        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            if first:
                self.log_info(
                    f"BehaviorToolWorker pump: first request {request.request_type} "
                    f"session={request.session_id}"
                )
                first = False
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
        except Exception as e:                                    # noqa: BLE001
            # A failed tool must not wedge the episode: the agent loop sees the
            # failure, the trajectory ends, and the reward module scores it as a
            # non-terminating episode rather than blocking the whole rollout.
            self._release(request.session_id)
            response = ToolChannelResponse(
                success=False, result=f"{type(e).__name__}:{e}")

        await self.output_channel.put(
            response, key=request.session_id, async_op=True
        ).async_wait()

    async def _start(self, request: ToolChannelRequest) -> ToolChannelResponse:
        spec = request.tool_args
        if isinstance(spec, str):
            spec = json.loads(spec)
        spec = dict(spec or {})
        activity = spec["activity"]

        assert self._session_lock is not None, "start_server() not called"
        await self._session_lock.acquire()
        try:
            if self._booted_activity is None:
                self._env = await asyncio.to_thread(self._boot, activity)
                self._booted_activity = activity
            elif activity != self._booted_activity:
                # Refusing beats mutating: OmniGibson cannot rebuild the scene for a
                # different activity in this process, so honouring the request would
                # silently run the wrong task.
                self._session_lock.release()
                return ToolChannelResponse(
                    success=False,
                    result=(
                        f"activity_mismatch: this worker booted "
                        f"{self._booted_activity!r} and OmniGibson locks one activity "
                        f"per process; {activity!r} needs its own worker rank"
                    ),
                )

            body = await asyncio.to_thread(self._env.start, spec.get("instance_id"))
        except BaseException:
            self._session_lock.release()
            raise

        self._active_session = request.session_id
        # `goal_lines` and `goal_nl` both come back from the env: the ground atoms are
        # instance-resolved, which is what keeps the RL dataset builder sim-free. Which
        # one reaches the policy is the agent loop's `goal_format` choice.
        return ToolChannelResponse(success=True, result=body)

    async def _execute(self, request: ToolChannelRequest) -> ToolChannelResponse:
        if self._active_session != request.session_id or self._env is None:
            return ToolChannelResponse(success=False, result="unknown_session")
        args = dict(request.tool_args or {})
        args.pop("_episode_id", None)                 # routing key, not a tool arg
        body = await asyncio.to_thread(
            self._env.call, request.tool_name or "", args)
        return ToolChannelResponse(
            success=bool(body.get("payload", {}).get("ok", True)),
            result=body.get("payload"),
            meta_info=body.get("meta"),
        )

    async def _end(self, request: ToolChannelRequest) -> ToolChannelResponse:
        if self._active_session == request.session_id and self._env is not None:
            await asyncio.to_thread(self._env.end)
        self._release(request.session_id)
        return ToolChannelResponse(success=True, result={"ok": True})

    def _release(self, session_id: str) -> None:
        """Free the single session slot. Idempotent — `_handle_one`'s error path and
        `_end` can both reach it for the same session."""
        if self._active_session != session_id:
            return
        self._active_session = None
        if self._session_lock is not None and self._session_lock.locked():
            self._session_lock.release()

    def _boot(self, activity: str):
        """Blocking: boots Kit and loads the scene. Minutes, not seconds.

        Runs on a worker thread (see the module docstring on `asyncio.to_thread`),
        and OmniGibson/Kit installs a SIGINT handler during boot. CPython allows
        `signal.signal` ONLY from the main thread of the main interpreter, so the
        boot died with:

            ValueError: signal only works in main thread of the main interpreter

        Neutralise `signal.signal` for the duration of the boot rather than moving
        the boot back onto the event loop. Two reasons this is the right trade:
        the offload exists because a multi-minute blocking call would otherwise
        stall the request channel and make a slow boot look like a hung worker; and
        the handler being installed is Kit's Ctrl-C convenience, which is meaningless
        inside a Ray actor that is terminated by `ray.kill`, never by SIGINT.

        Scoped tightly -- restored in `finally`, and only around the boot -- so no
        other component silently loses its handlers.
        """
        import signal as _signal

        from rlinf.envs.behavior.env_server import BehaviorEnv

        real_signal = _signal.signal
        suppressed: list[int] = []

        def _signal_noop(signum, handler):
            suppressed.append(signum)
            return None

        _signal.signal = _signal_noop
        try:
            env = BehaviorEnv(
                activity,
                obs_mode=self.obs_mode,
                instances_per_activity=self.instances_per_activity,
                rgb=self.rgb,
                partial_scene=self.partial_scene,
            )
        finally:
            _signal.signal = real_signal
        if suppressed:
            self.log_info(
                f"BehaviorToolWorker: suppressed {len(suppressed)} signal handler(s) "
                f"{sorted(set(suppressed))} during OmniGibson boot (worker thread)"
            )
        return env


def tool_names() -> list[str]:
    """Names the agent loop registers with the tool router.

    Sourced from the SFT schema so the RL action space cannot silently drift from
    the one the cold-start was trained on."""
    from rlinf.envs.behavior.sft_build import tool_schemas

    return [t["name"] for t in tool_schemas()]


__all__ = ["BehaviorToolWorker", "tool_names"]
