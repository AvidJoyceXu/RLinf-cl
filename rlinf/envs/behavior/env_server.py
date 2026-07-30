"""BEHAVIOR env-as-a-service — the OmniGibson half of integrated (online) RL.

Runs in the **py3.10 / openvla-oft** venv (IsaacSim 4.5 + OmniGibson 3.7.1) and
exposes one booted :class:`SemanticACI` over plain HTTP. The **py3.11 / reason**
trainer (SGLang + Megatron + FSDP) drives it through
``rlinf/agents/behavior/behavior_tool_worker.py``.

Why a server instead of an in-process tool worker: the two stacks cannot share an
interpreter (IsaacSim pins py3.10, the SGLang/apex wheels pin py3.11/torch2.6), so
EQA's in-process ``HabitatEQAToolWorker`` pattern is unavailable. This is the
standard agentic-RL answer — env-as-a-service, the same shape Search-R1 uses for
its retrieval server. Crucially it is **not** the ReST^EM split
(:doc:`0716 <../../../../code-doc/human-plan/0716 - GRPO to iterative RFT - design.md>`):
sampling and the policy update stay in one online loop, and the venv boundary
costs one localhost round trip per tool call, not an offline JSONL round trip per
iteration.

Latency budget: tool calls are logic-bound at ~0.08 ms, placement replays a cached
pose at ~13 ms, and a reset is ~1.5 s. HTTP over loopback adds well under a
millisecond — negligible against an LLM turn.

**One activity per process.** OmniGibson locks global macros (``HEADLESS``) at the
first boot, so a second ``boot_env`` in the same process dies with ``Cannot set
attribute HEADLESS in MacroDict``, and ``env.update_task()`` can only re-sample the
*same* activity. Scale by running one server per activity — see
``scripts/launch_behavior_env_servers.sh``.

**One session at a time per server.** There is a single simulator behind this
process; a second concurrent session would mutate the first one's scene. Requests
that arrive while a session is open are refused with 409. Rollout concurrency comes
from the number of servers, which is what ``group_size`` should be sized against.

Launch:
    with-env python -m rlinf.envs.behavior.env_server \\
        --activity picking_up_trash --port 18801 --obs-mode full
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Stdlib HTTP on purpose: the openvla-oft venv is a delicate IsaacSim pin and is
# not worth perturbing with a web framework for four endpoints.


class BehaviorEnv:
    """One booted activity + its expert plan + per-instance pose cache."""

    def __init__(self, activity: str, obs_mode: str = "full",
                 instances_per_activity: int = 0, rgb: bool = False,
                 partial_scene: bool = True, fast_reset: bool = False):
        from omegaconf import OmegaConf
        from omnigibson.envs import VectorEnvironment

        from rlinf.envs.behavior import sft_build as sb
        from rlinf.envs.behavior.harvest_sft import resolve_scene_and_dir
        from rlinf.envs.behavior.nl_goal import render_goal_nl
        from rlinf.envs.behavior.rft_sample import prep_plan
        from rlinf.envs.behavior.semantic_tools import SemanticACI
        from rlinf.envs.behavior.utils import setup_omni_cfg

        self.activity = activity
        self.obs_mode = obs_mode
        self.fast_reset = fast_reset
        self.scene, inst_dir = resolve_scene_and_dir(activity)

        cfg = OmegaConf.load(
            "/workspace/RLinf/examples/embodiment/config/env/behavior_r1pro.yaml")
        OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
        OmegaConf.update(cfg, "omni_config.task.activity_name", activity, force_add=True)
        OmegaConf.update(cfg, "omni_config.scene.scene_model", self.scene, force_add=True)
        if partial_scene:
            # Load only the rooms the activity's BDDL mentions. Without this the
            # whole house is instantiated, and every per-episode `scene.reset()`
            # then walks *all* of it: the restore path calls
            # `set_position_orientation` -> `set_attribute` per prim, and on
            # house_double_floor_lower that turned one reset into >1 h of
            # single-process CPU (measured 2026-07-30, picking_up_trash). Room
            # filtering is upstream's own recommendation for startup cost; it also
            # bounds reset, which is what a rollout pays every episode.
            OmegaConf.update(cfg, "omni_config.scene.partial_scene_load", True,
                             force_add=True)
        if not rgb:
            # The semantic ACI never reads pixels, and loading the R1's head+wrist
            # cameras costs ~10 min of the ~16 min boot: `VisionSensor._post_load`
            # drives Kit's renderer to warm the RTX pipeline, which on a
            # non-ray-tracing GPU (H20/H100/A100) is the slowest thing in the boot.
            # Dropping `rgb` skips sensor creation entirely. Pass --rgb for the
            # vision-augmentation ablation, where the cost is the point.
            OmegaConf.update(cfg, "omni_config.robots.0.obs_modalities",
                             ["proprio"], force_add=True)
        self.vec_env = VectorEnvironment(
            1, OmegaConf.to_container(setup_omni_cfg(cfg), resolve=True))
        self.vec_env.reset()
        self.env = self.vec_env.envs[0]
        self.aci = SemanticACI(self.env, obs_mode=obs_mode)

        task = self.env.task
        self.goal_lines = sb.goal_atoms_to_lines(task.ground_goal_state_options[0])
        # Natural-language goal, rendered from the UNGROUND BDDL so the quantifiers
        # survive. `fallbacks` is surfaced (not swallowed) because a mechanical
        # sentence is a prompt we should not silently train on -- see nl_goal.py.
        self.goal_nl, self.goal_nl_fallbacks = render_goal_nl(activity)
        self.plan, files = prep_plan(task, activity, inst_dir)
        if instances_per_activity:
            files = files[:instances_per_activity]
        self.instances = files
        if not self.instances:
            raise RuntimeError(f"no pre-sampled instances for {activity}")

        self.lock = threading.Lock()
        self.session_id: str | None = None
        self.cached_instance: int | None = None

    # ------------------------------------------------------------------ #
    def start(self, instance_id: int | None) -> dict:
        """Reset to a clean instance and open a session.

        The pose cache is **per instance**: ``build_pose_cache`` records absolute
        placement poses, which only land inside the target on the geometry they
        were sampled on. A boot-scoped cache silently mis-places on every instance
        but the boot one (observed as 0/8 on picking_up_trash).

        It is also memoised per instance, which matters a lot for GRPO: a group is
        ``group_size`` rollouts of the *same* (activity, instance), so only the
        first session on an instance pays the cache build -- the rest reuse poses
        that are identical by construction. Cache building costs one
        ``sample_kinematics`` per place target (documented at 20-78 s each) plus a
        second full ``scene.reset()``; skipping it when the instance has not
        changed removes both."""
        from rlinf.envs.behavior.rft_sample import (
            build_cache_on_instance,
            reset_to_instance,
        )

        if instance_id is None:
            inst = self.instances[0]
        else:
            match = [f for f in self.instances if f.instance_id == instance_id]
            if not match:
                raise KeyError(f"instance {instance_id} not available for {self.activity}")
            inst = match[0]

        # `fast_reset` skips the trailing scene.reset() on the instance load -- the
        # quadratic pose-cache rebuild that dominates reset cost (DEBUG-LOG §5). The
        # `contaminated` field below is the guard: if skipping it left stale state
        # that already satisfies the goal, the episode is flagged and scored zero
        # rather than silently handing the policy a free success.
        reset_to_instance(self.aci, self.env, inst, reset_scene=not self.fast_reset)
        if self.cached_instance != inst.instance_id:
            build_cache_on_instance(self.aci, self.env, self.plan, inst)
            self.cached_instance = inst.instance_id

        self.session_id = uuid4().hex
        return {
            "session_id": self.session_id,
            "activity": self.activity,
            "scene": self.scene,
            "instance_id": inst.instance_id,
            "obs_mode": self.obs_mode,
            # Both goal renderings travel together; which one reaches the policy is
            # the agent loop's `goal_format` choice, so the NL and atoms conditions
            # are the same server and the same episode -- only the prompt differs.
            "goal_lines": self.goal_lines,
            "goal_nl": self.goal_nl,
            "goal_nl_fallbacks": self.goal_nl_fallbacks,
            # A goal already satisfied at reset would hand the policy a free
            # success; the caller must drop such episodes rather than score them.
            "contaminated": bool(self.aci.is_success()),
        }

    def call(self, name: str, arguments: dict) -> dict:
        """Dispatch one tool and return {payload, meta}.

        ``payload`` is the exact JSON the SFT trainer put in the context, so the
        cold-start transfers verbatim. ``meta`` carries the real BDDL goal status
        that ``rlinf/algorithms/rewards/behavior.py`` scores off."""
        from rlinf.envs.behavior.rft_sample import result_payload

        fn = getattr(self.aci, name, None)
        if fn is None or name.startswith("_"):
            return {"payload": {"ok": False, "reason": f"unknown_tool:{name}"},
                    "meta": self._meta(), "terminal": False}
        try:
            res = fn(**(arguments or {}))
        except TypeError as ex:                       # bad/missing arguments
            return {"payload": {"ok": False, "reason": f"bad_arguments: {ex}"},
                    "meta": self._meta(), "terminal": False}
        return {"payload": result_payload(name, res),
                "meta": self._meta(),
                "terminal": name == "end_task"}

    def _meta(self) -> dict:
        """Real BDDL goal evaluation — pure logic over true simulator state."""
        gs = self.aci.goal_status()
        n_sat = len(gs["satisfied"])
        n_goal = n_sat + len(gs["unsatisfied"])
        return {"num_satisfied": n_sat, "num_goal": n_goal,
                "is_success": (len(gs["unsatisfied"]) == 0 and n_sat > 0)}

    def end(self) -> dict:
        self.session_id = None
        return {"ok": True}

    def close(self):
        try:
            self.vec_env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
def make_handler(env: BehaviorEnv):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):            # one line per tool call is far too noisy
            pass

        def _send(self, code: int, body: dict):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not_found"})
            self._send(200, {"ok": True, "activity": env.activity, "scene": env.scene,
                             "obs_mode": env.obs_mode, "busy": env.session_id is not None,
                             "goal_nl": env.goal_nl,
                             "instances": [f.instance_id for f in env.instances]})

        def do_POST(self):
            try:
                parts = [p for p in self.path.split("/") if p]
                body = self._read()

                if parts == ["session", "start"]:
                    # Serialize: one simulator, so a second session would corrupt
                    # the first one's scene rather than run beside it.
                    if not env.lock.acquire(blocking=False):
                        return self._send(409, {"error": "busy"})
                    try:
                        return self._send(200, env.start(body.get("instance_id")))
                    except Exception:
                        env.lock.release()
                        raise

                if len(parts) == 3 and parts[0] == "session":
                    sid, verb = parts[1], parts[2]
                    if sid != env.session_id:
                        return self._send(410, {"error": "stale_session"})
                    if verb == "tool":
                        return self._send(200, env.call(body.get("name", ""),
                                                        body.get("arguments") or {}))
                    if verb == "end":
                        out = env.end()
                        env.lock.release()
                        return self._send(200, out)

                self._send(404, {"error": "not_found"})
            except Exception as ex:
                traceback.print_exc()
                self._send(500, {"error": f"{type(ex).__name__}: {str(ex)[:300]}"})

    return Handler


def serve_blocking(server: ThreadingHTTPServer) -> None:
    """``serve_forever()`` replacement built on a blocking ``accept()``.

    Do not swap this back for ``server.serve_forever()``. Once IsaacSim/Kit has
    booted in this process, the ``selectors``-based poll inside ``serve_forever``
    never reports the listening socket readable: connections land in the accept
    queue and sit there while the main thread sleeps in ``select``, so every
    request hangs with the server looking perfectly healthy.

    Verified 2026-07-29 in the behavior container: an identical stdlib
    ``ThreadingHTTPServer`` answers immediately in a fresh process and stops
    answering in one that has imported OmniGibson — same code, same port. The
    blocking ``accept()`` below sidesteps the selector entirely; ThreadingMixIn
    still gives each request its own thread.
    """
    while True:
        try:
            request, client_address = server.get_request()
        except OSError:
            continue
        try:
            server.process_request(request, client_address)
        except Exception:
            server.handle_error(request, client_address)
            server.shutdown_request(request)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--obs-mode", default="full", choices=["full", "partial"])
    ap.add_argument("--instances-per-activity", type=int, default=0,
                    help="0 = all pre-sampled instances")
    ap.add_argument("--rgb", action="store_true",
                    help="load the robot's cameras (adds ~10 min to boot; only the "
                         "vision-augmentation ablation needs them)")
    ap.add_argument("--full-scene", action="store_true",
                    help="instantiate the whole house instead of the activity's "
                         "rooms (much slower boot AND reset; see __init__)")
    ap.add_argument("--fast-reset", action="store_true",
                    help="skip the trailing scene.reset() on instance load. This is "
                         "the fix for the >90 min reset (DEBUG-LOG §5); verify the "
                         "start state (contaminated flag + observe) before trusting "
                         "a training run with it on.")
    args = ap.parse_args()

    env = BehaviorEnv(args.activity, args.obs_mode, args.instances_per_activity,
                      rgb=args.rgb, partial_scene=not args.full_scene,
                      fast_reset=args.fast_reset)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(env))
    # Boot is ~5 min (shader compile); the trainer polls /health, so announce
    # readiness on a line it can grep rather than making it guess.
    print(f"BEHAVIOR_ENV_SERVER_READY {json.dumps({'activity': args.activity, 'scene': env.scene, 'port': args.port, 'instances': len(env.instances)})}",
          flush=True)
    try:
        serve_blocking(server)              # NOT serve_forever() — see its docstring
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
