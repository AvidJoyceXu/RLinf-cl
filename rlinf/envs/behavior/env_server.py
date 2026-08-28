"""BEHAVIOR env-as-a-service — the OmniGibson half of integrated (online) RL.

Exposes one booted :class:`SemanticACI` over plain HTTP.

**No longer on the training path.** ``BehaviorToolWorker`` now holds a
:class:`BehaviorEnv` in process and calls it directly. The claim that motivated this
server -- "the two stacks cannot share an interpreter (IsaacSim pins py3.10, the
SGLang/apex wheels pin py3.11/torch2.6)" -- was **wrong**: only IsaacSim's cp310 pin
is real, and every trainer component has a cp310 form, so one venv runs both. See
``code-doc/0730 - single-venv container/INSTALL.md`` §1.

This file survives as a **debugging surface**: boot one activity, curl a single tool
call, drive one episode by hand, time resets without standing up Ray and Megatron.
That is genuinely useful and is why it is kept rather than deleted. It is **not** the
ReST^EM split
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

    def __init__(
        self,
        activity: str,
        obs_mode: str = "full",
        instances_per_activity: int = 0,
        rgb: bool = False,
        partial_scene: bool = True,
        fast_reset: bool = False,
        instance_source: str | None = None,
        debug_video_dir: str | None = None,
        debug_video_fps: int = 4,
        debug_render_iters: int = 3,
        scene_profile: str | None = None,
        rgb_manipulation_evidence: bool = False,
    ):
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
        self.debug_video_dir = debug_video_dir
        self.debug_render_iters = debug_render_iters
        self.rgb_manipulation_evidence = bool(rgb_manipulation_evidence)
        self.debug_renderer = {
            "render_mode": "runtime_default",
            "render_iters": self.debug_render_iters,
        }
        if debug_video_dir and not rgb:
            raise ValueError(
                "debug video records the physical R1 head sensor; pass rgb=True/--rgb"
            )
        if self.rgb_manipulation_evidence and not debug_video_dir:
            raise ValueError(
                "rgb_manipulation_evidence is debug evidence and requires "
                "debug_video_dir (and rgb=True)"
            )
        self.instance_source = instance_source or os.environ.get(
            "BEHAVIOR_INSTANCE_SOURCE", "2025-official"
        )
        self.scene, inst_dir = resolve_scene_and_dir(
            activity, source=self.instance_source
        )
        # Preserve the old boolean API, but make formal scene composition explicit.
        # ``task_relevant`` is a fast plumbing ablation; it is never an official
        # BEHAVIOR visual condition. ``official_rooms`` mirrors the 2026 evaluator:
        # load the complete task template, then filter by B100 room instances.
        self.scene_profile = scene_profile or (
            "task_relevant" if partial_scene else "full"
        )
        valid_scene_profiles = {"task_relevant", "official_rooms", "full"}
        if self.scene_profile not in valid_scene_profiles:
            raise ValueError(
                f"unknown scene_profile={self.scene_profile!r}; "
                f"expected one of {sorted(valid_scene_profiles)}"
            )
        self.official_scene_contract = None

        cfg = OmegaConf.load(
            "/workspace/RLinf/examples/embodiment/config/env/behavior_r1pro.yaml"
        )
        OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
        OmegaConf.update(
            cfg, "omni_config.task.activity_name", activity, force_add=True
        )
        OmegaConf.update(
            cfg, "omni_config.scene.scene_model", self.scene, force_add=True
        )
        if self.scene_profile == "task_relevant":
            # `load_task_relevant_only` loads the task-relevant objects plus the
            # building structure and skips every other object in the house. This is
            # the ONLY lever that reduces N, and N is what makes reset expensive:
            # `scene.reset()` -> `load_state` writes every object's pose through USD
            # (`set_attribute`), so cost is linear in scene population and is paid
            # EVERY episode, unlike boot.
            #
            # NOTE (2026-07-30): this replaces `scene.partial_scene_load`, which does
            # not exist in OmniGibson 3.7.1 -- grep finds zero readers. It was added
            # with `force_add=True`, so it silently created a key nothing consumed and
            # the whole house loaded anyway. Earlier claims in INSTALL.md/DEBUG-LOG
            # that it "bounds N" were wrong. If a key here ever looks load-bearing,
            # grep OmniGibson for a reader before believing it.
            #
            # TRADEOFF, and it is not purely a speedup: with non-task objects absent,
            # `observe()` returns fewer candidate objects, which makes the partial-
            # observability condition EASIER than a fully furnished house. The BDDL
            # goal only references task-relevant objects, so scoring is unaffected --
            # but solvability is. Keep this OFF for headline S2 numbers; it is for
            # pipeline work and for ablations where the cost is prohibitive.
            OmegaConf.update(
                cfg, "omni_config.scene.load_task_relevant_only", True, force_add=True
            )
            # The base YAML still carries the pre-v3.9 evaluator switch
            # ``partial_scene_load``.  ``setup_omni_cfg`` interprets that switch via
            # gello's room-selection helpers, while v3.9 has a native scene option
            # for the task-object subset requested above.  Do not activate both
            # mechanisms: apart from requiring an unrelated teleoperation package,
            # they select different subsets of the scene.
            OmegaConf.update(
                cfg, "omni_config.scene.partial_scene_load", False, force_add=True
            )
        elif self.scene_profile == "official_rooms":
            from rlinf.envs.behavior.official_scene import (
                resolve_official_scene_contract,
            )

            self.official_scene_contract = resolve_official_scene_contract(
                activity=activity,
                scene_model=self.scene,
                instance_dir=inst_dir,
            )
            OmegaConf.update(
                cfg,
                "omni_config.scene.scene_file",
                self.official_scene_contract.full_template["path"],
                force_add=True,
            )
            OmegaConf.update(
                cfg, "omni_config.scene.scene_instance", None, force_add=True
            )
            OmegaConf.update(
                cfg,
                "omni_config.scene.load_room_instances",
                list(self.official_scene_contract.room_instances),
                force_add=True,
            )
            OmegaConf.update(
                cfg, "omni_config.scene.load_room_types", None, force_add=True
            )
            OmegaConf.update(
                cfg, "omni_config.scene.load_task_relevant_only", False, force_add=True
            )
            OmegaConf.update(
                cfg, "omni_config.scene.partial_scene_load", False, force_add=True
            )
        else:
            OmegaConf.update(
                cfg, "omni_config.scene.load_room_instances", None, force_add=True
            )
            OmegaConf.update(
                cfg, "omni_config.scene.load_room_types", None, force_add=True
            )
            OmegaConf.update(
                cfg, "omni_config.scene.load_task_relevant_only", False, force_add=True
            )
            OmegaConf.update(
                cfg, "omni_config.scene.partial_scene_load", False, force_add=True
            )
        if not rgb:
            # The semantic ACI never reads pixels, and loading the R1's head+wrist
            # cameras costs ~10 min of the ~16 min boot: `VisionSensor._post_load`
            # drives Kit's renderer to warm the RTX pipeline, which on a
            # non-ray-tracing GPU (H20/H100/A100) is the slowest thing in the boot.
            # Dropping `rgb` skips sensor creation entirely. Pass --rgb for the
            # vision-augmentation ablation, where the cost is the point.
            OmegaConf.update(
                cfg, "omni_config.robots.0.obs_modalities", ["proprio"], force_add=True
            )
        else:
            # Replicator render products are created while the robot sensors load.
            # Resizing a sensor after the first reset requires detaching live
            # annotators and is not safe on Isaac Sim 5.1. Configure the official
            # BEHAVIOR resolutions before the environment is constructed instead.
            for path, value in (
                (
                    "omni_config.robots.0.sensor_config.VisionSensor."
                    "sensor_kwargs.image_height",
                    480,
                ),
                (
                    "omni_config.robots.0.sensor_config.VisionSensor."
                    "sensor_kwargs.image_width",
                    480,
                ),
                (
                    "omni_config.robots.0.sensor_config.zed_link:Camera:0."
                    "sensor_kwargs.image_height",
                    720,
                ),
                (
                    "omni_config.robots.0.sensor_config.zed_link:Camera:0."
                    "sensor_kwargs.image_width",
                    720,
                ),
                (
                    "omni_config.robots.0.sensor_config.zed_link:Camera:0."
                    "sensor_kwargs.horizontal_aperture",
                    40.0,
                ),
            ):
                OmegaConf.update(cfg, path, value, force_add=True)
        self.vec_env = VectorEnvironment(
            1, OmegaConf.to_container(setup_omni_cfg(cfg), resolve=True)
        )
        if rgb:
            from rlinf.envs.behavior.utils import apply_env_wrapper

            apply_env_wrapper(self.vec_env, "rgb")
        self.vec_env.reset()
        self.env = self.vec_env.envs[0]
        self.aci = SemanticACI(self.env, obs_mode=obs_mode)
        self.video_recorder = None
        self.rgb_manipulation = None
        if debug_video_dir:
            from rlinf.envs.behavior.trajectory_video import TrajectoryVideoRecorder

            self.video_recorder = TrajectoryVideoRecorder(
                debug_video_dir, fps=debug_video_fps
            )
        if self.rgb_manipulation_evidence:
            # Keep this import behind the explicit RGB-only gate. TextWorld never
            # imports this module and the ordinary semantic path never creates a
            # physical grasp constraint or captures paired frames.
            from rlinf.envs.behavior.rgb_manipulation import RGBManipulationEvidence

            self.rgb_manipulation = RGBManipulationEvidence(self.env, self.aci)

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
        from rlinf.envs.behavior.expert_planner import plan as build_expert_plan
        from rlinf.envs.behavior.rft_sample import (
            build_cache_on_instance,
            reset_to_instance,
        )

        if instance_id is None:
            inst = self.instances[0]
        else:
            match = [f for f in self.instances if f.instance_id == instance_id]
            if not match:
                have = [f.instance_id for f in self.instances]
                raise KeyError(
                    f"instance {instance_id} not available for {self.activity}: "
                    f"{len(have)} loaded, ids {have[:5]}{'...' if len(have) > 5 else ''}. "
                    f"instances_per_activity truncates to the FIRST N discovered "
                    f"instances while build_rl_dataset samples arbitrary ids -- set "
                    f"tools.behavior.instances_per_activity: 0 to keep all of them."
                )
            inst = match[0]

        # `fast_reset` skips the trailing scene.reset() on the instance load -- the
        # quadratic pose-cache rebuild that dominates reset cost (DEBUG-LOG §5). The
        # `contaminated` field below is the guard: if skipping it left stale state
        # that already satisfies the goal, the episode is flagged and scored zero
        # rather than silently handing the policy a free success.
        # hard=True is OmniGibson's default and costs a doubled full-scene state
        # write (restore -> batch_remove_objects -> removing_objects dumps AND
        # reloads everything even when the remove list is empty). It is only NEEDED
        # when the previous episode changed the object SET -- slice/dice add
        # half-objects, fill/spray instantiate particle systems -- because
        # hard=False ignores objects missing from the initial file. Ask the ACI
        # instead of always paying for it.
        # Phase timing, printed unconditionally. `session/start` has been slow for
        # three different reasons in a row, and each time a py-spy stack was ambiguous
        # about WHICH phase owned the cost -- line attribution cannot separate the
        # reset_to_instance called here from the one inside build_cache_on_instance.
        # Two timers settle in one run what several profiling sessions did not.
        import time as _t

        hard = bool(getattr(self.aci, "object_set_dirty", False))
        if self.rgb_manipulation is not None:
            self.rgb_manipulation.release_before_reset()
        _t0 = _t.time()
        reset_to_instance(
            self.aci, self.env, inst, reset_scene=not self.fast_reset, hard_reset=hard
        )
        self.aci.camera_pitch = 0.0
        d_reset = _t.time() - _t0
        self.aci.object_set_dirty = False

        # v3.9 binds task-scope entities while loading the concrete TRO state.
        # Planning in __init__ is therefore too early for 2026 instances: every
        # wrapped_obj is None and valid goal arguments become ``unresolved-args``.
        # Refresh only after reset/binding and before the per-instance pose cache.
        # This is OmniGibson bookkeeping; TextWorld has its own planner and never
        # constructs this environment.
        self.plan = build_expert_plan(self.env.task)
        print(
            f"[start] plan calls={len(self.plan.calls)} pairs={len(self.plan.pairs)} "
            f"unsupported={len(self.plan.unsupported)}",
            flush=True,
        )

        d_cache = 0.0
        cache_built = False
        if self.cached_instance != inst.instance_id:
            _t0 = _t.time()
            build_cache_on_instance(self.aci, self.env, self.plan, inst)
            d_cache = _t.time() - _t0
            cache_built = True
            self.cached_instance = inst.instance_id
        print(
            f"[start] instance={inst.instance_id} hard={hard} "
            f"reset={d_reset:.1f}s pose_cache={d_cache:.1f}s "
            f"(built={cache_built}) total={d_reset + d_cache:.1f}s",
            flush=True,
        )

        self.session_id = uuid4().hex
        result = {
            "session_id": self.session_id,
            "activity": self.activity,
            "scene": self.scene,
            "instance_id": inst.instance_id,
            "obs_mode": self.obs_mode,
            "instance_source": self.instance_source,
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
        if self.video_recorder is not None:
            result["debug_video"] = self.video_recorder.begin(
                {
                    "session_id": self.session_id,
                    "activity": self.activity,
                    "scene": self.scene,
                    "instance_id": inst.instance_id,
                    "instance_source": self.instance_source,
                    "obs_mode": self.obs_mode,
                    "debug_renderer": self.debug_renderer,
                }
            )
            self._record_debug_frame("session_start", ok=True)
        return result

    def call(self, name: str, arguments: dict) -> dict:
        """Dispatch one tool and return {payload, meta}.

        ``payload`` is the exact JSON the SFT trainer put in the context, so the
        cold-start transfers verbatim. ``meta`` carries the real BDDL goal status
        that ``rlinf/algorithms/rewards/behavior.py`` scores off."""
        from rlinf.envs.behavior.rft_sample import result_payload

        fn = getattr(self.aci, name, None)
        if fn is None or name.startswith("_"):
            payload = {"ok": False, "reason": f"unknown_tool:{name}"}
            self._record_debug_frame(name or "unknown_tool", ok=False)
            return {"payload": payload, "meta": self._meta(), "terminal": False}
        rgb_pair = self.rgb_manipulation is not None and self.rgb_manipulation.handles(
            name
        )
        pre_capture = self._capture_debug_frame() if rgb_pair else None
        control_capture = self._capture_debug_frame() if rgb_pair else None
        context = (
            self.rgb_manipulation.before(name, arguments or {}) if rgb_pair else None
        )
        try:
            res = fn(**(arguments or {}))
        except TypeError as ex:  # bad/missing arguments
            if rgb_pair:
                self.rgb_manipulation.abort(context)
            payload = {"ok": False, "reason": f"bad_arguments: {ex}"}
            if rgb_pair:
                post_capture = self._capture_debug_frame()
                self._append_debug_capture(
                    pre_capture,
                    tool=f"{name}:pre",
                    ok=False,
                    event_metadata={"phase": "pre", "arguments": arguments or {}},
                )
                self._append_debug_capture(
                    post_capture,
                    tool=f"{name}:post",
                    ok=False,
                    event_metadata={
                        "phase": "post",
                        "arguments": arguments or {},
                        "rgb_manipulation": {
                            "physical_ok": False,
                            "physical_reason": "bad_arguments",
                        },
                    },
                )
            else:
                self._record_debug_frame(name, ok=False)
            return {"payload": payload, "meta": self._meta(), "terminal": False}
        except Exception:
            if rgb_pair:
                self.rgb_manipulation.abort(context)
            raise
        if rgb_pair:
            evidence = self.rgb_manipulation.after(context, res)
            post_capture = self._capture_debug_frame()
            evidence["pixel_audit"] = self.rgb_manipulation.pixel_audit(
                before_frame=pre_capture[0],
                control_frame=control_capture[0],
                after_frame=post_capture[0],
                before_camera=pre_capture[1],
                after_camera=post_capture[1],
                evidence=evidence,
            )
            self._append_debug_capture(
                pre_capture,
                tool=f"{name}:pre",
                ok=bool(res.ok),
                event_metadata={
                    "phase": "pre",
                    "arguments": arguments or {},
                    "rgb_manipulation_state": evidence["state_before"],
                },
            )
            self._append_debug_capture(
                post_capture,
                tool=f"{name}:post",
                ok=bool(res.ok),
                event_metadata={
                    "phase": "post",
                    "arguments": arguments or {},
                    "rgb_manipulation": evidence,
                },
            )
        payload = result_payload(name, res)
        if not rgb_pair:
            self._record_debug_frame(name, ok=bool(payload.get("ok", True)))
        return {
            "payload": payload,
            "meta": self._meta(),
            "terminal": name == "end_task",
        }

    def _record_debug_frame(self, tool: str, ok: bool) -> None:
        """Capture the physical R1 head camera after every tool."""
        if self.video_recorder is None:
            return
        capture = self._capture_debug_frame()
        self._append_debug_capture(capture, tool=tool, ok=ok)

    def _capture_debug_frame(self):
        """Render and return one lossless head frame plus camera calibration."""
        if self.video_recorder is None:
            raise RuntimeError("debug frame requested without a video recorder")
        import numpy as np
        import omnigibson as og

        from rlinf.envs.behavior.utils import convert_uint8_rgb

        matches = [
            (name, sensor)
            for name, sensor in self.env.robots[0].sensors.items()
            if "zed_link:Camera:0" in name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "debug video requires exactly one R1 ZED head sensor; "
                f"found {[name for name, _ in matches]!r}"
            )
        sensor_name, cam = matches[0]
        for _ in range(max(1, self.debug_render_iters)):
            og.sim.render()
        raw = cam.get_obs()[0]["rgb"]
        frame = convert_uint8_rgb(raw)
        frame = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
        if frame.size == 0:
            raise RuntimeError("R1 head camera returned an empty RGB frame")
        camera_pos, camera_orn = cam.get_position_orientation()
        try:
            intrinsic = cam.intrinsic_matrix
            intrinsic = (
                intrinsic.detach().cpu().tolist()
                if hasattr(intrinsic, "detach")
                else np.asarray(intrinsic).tolist()
            )
            intrinsic_source = "vision_sensor_intrinsic_matrix"
        except AssertionError:
            # A scene.reset() can transiently leave Replicator's camera-parameter
            # annotator degenerate on Isaac 5.1 even though the physical sensor's
            # RGB render and authored optics are valid. Derive the same pinhole K
            # from those optics instead of making RGB evidence depend on an
            # auxiliary annotator. Record this fallback explicitly in the sidecar.
            width, height = int(cam.image_width), int(cam.image_height)
            focal = float(cam.focal_length)
            horizontal = float(cam.horizontal_aperture)
            vertical = float(
                getattr(cam, "vertical_aperture", horizontal * height / width)
            )
            fx, fy = focal * width / horizontal, focal * height / vertical
            intrinsic = [
                [fx, 0.0, width / 2.0],
                [0.0, fy, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
            intrinsic_source = "pinhole_from_focal_length_and_aperture"
        metadata = {
            "camera_xyz": [round(float(value), 6) for value in camera_pos],
            "camera_orientation_xyzw": [round(float(value), 8) for value in camera_orn],
            "intrinsic": [
                [round(float(value), 6) for value in row] for row in intrinsic
            ],
            "intrinsic_source": intrinsic_source,
            "sensor_name": sensor_name,
            "sensor_prim_path": cam.prim_path,
            "resolution": [int(cam.image_height), int(cam.image_width)],
        }
        return frame, metadata

    def _append_debug_capture(
        self,
        capture,
        *,
        tool: str,
        ok: bool,
        event_metadata: dict | None = None,
    ) -> None:
        if self.video_recorder is None:
            return
        frame, camera_metadata = capture
        metadata = dict(camera_metadata)
        if event_metadata:
            metadata.update(event_metadata)
        self.video_recorder.append(
            frame,
            tool=tool,
            ok=ok,
            event_metadata=metadata,
        )

    def _meta(self) -> dict:
        """Real BDDL goal evaluation — pure logic over true simulator state."""
        gs = self.aci.goal_status()
        n_sat = len(gs["satisfied"])
        n_goal = n_sat + len(gs["unsatisfied"])
        return {
            "num_satisfied": n_sat,
            "num_goal": n_goal,
            "is_success": (len(gs["unsatisfied"]) == 0 and n_sat > 0),
        }

    def end(self) -> dict:
        if self.rgb_manipulation is not None:
            self.rgb_manipulation.release_before_reset()
        video = None
        if self.video_recorder is not None:
            video = self.video_recorder.finish(complete=True, reason="session_end")
        self.session_id = None
        return {"ok": True, "debug_video": video}

    def close(self):
        if self.rgb_manipulation is not None:
            self.rgb_manipulation.release_before_reset()
        if self.video_recorder is not None:
            self.video_recorder.finish(complete=False, reason="environment_closed")
        try:
            self.vec_env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
def make_handler(env: BehaviorEnv):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # one line per tool call is far too noisy
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
            self._send(
                200,
                {
                    "ok": True,
                    "activity": env.activity,
                    "scene": env.scene,
                    "obs_mode": env.obs_mode,
                    "busy": env.session_id is not None,
                    "goal_nl": env.goal_nl,
                    "instances": [f.instance_id for f in env.instances],
                },
            )

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
                        return self._send(
                            200,
                            env.call(body.get("name", ""), body.get("arguments") or {}),
                        )
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
    ap.add_argument("--obs-mode", default="full", choices=["full", "partial", "fov"])
    ap.add_argument(
        "--instances-per-activity",
        type=int,
        default=0,
        help="0 = all pre-sampled instances",
    )
    ap.add_argument(
        "--instance-source",
        default=None,
        choices=["2025-official", "2026-v3.9.1", "local-v3.7"],
        help="one versioned instance population; defaults to "
        "BEHAVIOR_INSTANCE_SOURCE or 2025-official",
    )
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="load the robot's cameras (adds ~10 min to boot; only the "
        "vision-augmentation ablation needs them)",
    )
    ap.add_argument(
        "--full-scene",
        action="store_true",
        help="instantiate the whole house instead of the activity's "
        "rooms (much slower boot AND reset; see __init__)",
    )
    ap.add_argument(
        "--fast-reset",
        action="store_true",
        help="skip the trailing scene.reset() on instance load. This is "
        "the fix for the >90 min reset (DEBUG-LOG §5); verify the "
        "start state (contaminated flag + observe) before trusting "
        "a training run with it on.",
    )
    ap.add_argument(
        "--debug-video-dir",
        default=None,
        help="record one R1 head-camera MP4 plus JSON sidecar per complete "
        "tool trajectory; frames are captured after every tool",
    )
    ap.add_argument("--debug-video-fps", type=int, default=4)
    ap.add_argument("--debug-render-iters", type=int, default=3)
    ap.add_argument(
        "--rgb-manipulation-evidence",
        action="store_true",
        help="RGB debug only: add assisted-grasp synchronization and paired "
        "pre/post manipulation frames; requires --rgb and --debug-video-dir",
    )
    args = ap.parse_args()

    env = BehaviorEnv(
        args.activity,
        args.obs_mode,
        args.instances_per_activity,
        rgb=args.rgb,
        partial_scene=not args.full_scene,
        fast_reset=args.fast_reset,
        instance_source=args.instance_source,
        debug_video_dir=args.debug_video_dir,
        debug_video_fps=args.debug_video_fps,
        debug_render_iters=args.debug_render_iters,
        rgb_manipulation_evidence=args.rgb_manipulation_evidence,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(env))
    # Boot is ~5 min (shader compile); the trainer polls /health, so announce
    # readiness on a line it can grep rather than making it guess.
    print(
        f"BEHAVIOR_ENV_SERVER_READY {json.dumps({'activity': args.activity, 'scene': env.scene, 'port': args.port, 'instances': len(env.instances)})}",
        flush=True,
    )
    try:
        serve_blocking(server)  # NOT serve_forever() — see its docstring
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
