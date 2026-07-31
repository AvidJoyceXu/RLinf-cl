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
import threading
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

        # Booting Kit on a worker thread does not work. Two main-thread assumptions
        # were patchable (signal.signal, a missing event loop); the third was not --
        # Kit's own `Loop` object gets poked by code expecting asyncio's interface,
        # and the resulting per-frame traceback storm (29k errors) turned a 5:59 boot
        # into >28 minutes with OmniGibson's own progress frozen at 00:04:45 while
        # Kit's app clock ran to 1,674,419ms. Three strikes: Kit wants the main
        # thread, so boot there.
        #
        # `activity` therefore has to be known at construction time. That is not a
        # real loss: OmniGibson locks one activity per process, so a worker serves
        # exactly one for its whole life -- the config merely states what was already
        # true. Leave it unset to fall back to the lazy path (fine for a debugging
        # process that is already on the main thread).
        self.activity_cfg: str | None = tcfg.get("activity", None)
        self._env: Any = None                  # BehaviorEnv
        self._booted_activity: str | None = None
        self._active_session: str | None = None
        # NOT `_lock`: the base Worker class owns `self._lock` (a threading.Lock,
        # used by `_get_collective_group`). Shadowing it with an asyncio.Lock made
        # the base's `with self._lock:` raise `AttributeError: __enter__`, because
        # asyncio.Lock implements __aenter__ and not __enter__ -- which killed the
        # request pump on its first channel recv and hung the whole run silently.
        self._session_lock: asyncio.Lock | None = None
        self.request_processor_task: asyncio.Task | None = None

        if self.activity_cfg:
            # Boot on a DEDICATED thread carrying a plain CPython event loop.
            #
            # Kit needs three things at once, and each previous attempt supplied only
            # some of them:
            #   (a) a thread with NO RUNNING loop. Ray's actor __init__ runs on its
            #       asyncio thread where uvloop is live, so Kit's internal
            #       run_until_complete raised "this event loop is already running"
            #       9,774 times.
            #   (b) a CPYTHON loop, not uvloop. sglang/Ray install uvloop's policy, so
            #       `asyncio.new_event_loop()` hands back a uvloop Loop -- which has no
            #       `_ready` / `_check_closed`, the CPython internals Kit reaches into.
            #       Verified: hasattr(uvloop_loop, "_ready") is False, CPython True.
            #       So construct SelectorEventLoop EXPLICITLY, bypassing the policy.
            #   (c) no signal.signal, since it is not the main thread (see _boot).
            #
            # Blocking until the boot finishes is deliberate: no request can arrive
            # before start_server, and a half-booted env is worse than a slow one.
            # Setting a loop on the BOOT THREAD is not enough, and the first attempt
            # proved it: 3.9 MILLION `'Loop' object has no attribute '_ready'`. Kit
            # spawns its own threads and calls asyncio.get_event_loop() there, which
            # goes through the GLOBAL POLICY -- and inside a Ray actor that policy is
            # uvloop's (Ray installs it; sglang alone does not -- verified). So every
            # Kit thread got a uvloop Loop no matter what this thread held.
            #
            # Switch the process policy to CPython's and LEAVE IT. Ray's own loop was
            # created under the old policy and keeps running untouched -- a policy
            # governs loops created AFTER it -- so this only affects loops made from
            # here on, which in this process means Kit's. uvloop is a throughput
            # optimisation for Ray, not a requirement, and this worker's job is to host
            # a simulator that cannot tolerate it.
            if not isinstance(asyncio.get_event_loop_policy(),
                              asyncio.DefaultEventLoopPolicy):
                self.log_info(
                    "BehaviorToolWorker: switching asyncio policy "
                    f"{type(asyncio.get_event_loop_policy()).__module__} -> asyncio "
                    "(Kit reaches into CPython loop internals uvloop does not have)"
                )
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

            box: dict[str, Any] = {}

            def _boot_thread():
                loop = asyncio.SelectorEventLoop()   # NOT new_event_loop(): see (b)
                asyncio.set_event_loop(loop)
                try:
                    box["env"] = self._boot(self.activity_cfg)
                except BaseException as e:           # noqa: BLE001
                    box["err"] = e

            th = threading.Thread(target=_boot_thread, name="omnigibson-boot",
                                  daemon=True)
            th.start()
            th.join()
            if "err" in box:
                raise box["err"]
            self._env = box["env"]
            self._booted_activity = self.activity_cfg

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

        # Kit's async engine calls asyncio.get_event_loop() during startup and then
        # drives it every frame. A bare `to_thread` worker has NO loop, so Kit got
        # None and spewed, once per frame, forever:
        #     AttributeError: 'Loop' object has no attribute '_ready' / '_check_closed'
        #     AttributeError: 'NoneType' object has no attribute 'call_soon_threadsafe'
        # Not cosmetic -- it never stops, and the boot never completes. Give the
        # thread a real loop. It is never run(); Kit only needs a loop OBJECT to
        # schedule onto, and call_soon_threadsafe on a non-running loop just queues.
        # The caller owns loop setup (see __init__): Kit needs a plain CPython loop
        # that is not running, which only the dedicated boot thread can guarantee.
        on_main = threading.current_thread() is threading.main_thread()

        real_signal = _signal.signal
        suppressed: list[int] = []

        def _signal_noop(signum, handler):
            suppressed.append(signum)
            return None

        if not on_main:
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
