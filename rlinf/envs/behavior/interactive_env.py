"""One BEHAVIOR environment, three kinds of operator.

A human, a text-only agent and a multimodal agent drive the SAME environment through
the SAME 17 tools against the SAME BDDL evaluator. The only thing that differs is how
they point at an object:

    human        clicks the render                 ->  "x,y"
    multimodal   emits a point or a box            ->  "x,y" or "x0,y0,x1,y1"
    text-only    reads handles out of `observe`    ->  "d3"

All three go through `detect.resolve`, so all three can be refused for the same
reasons -- nothing there, ambiguous, stale. That is the point: holding task, scene,
dynamics, scoring and turn budget fixed and varying ONLY the selection channel is what
makes "how much of embodied task success is perception-channel-limited" a measurable
question rather than a rhetorical one.

WHAT IS AUTHORITATIVE. The symbolic world is. Dynamics and success come from BDDL
literals and the real `evaluate_goal_conditions`; the renderer is a VIEWER onto the same
viewpoint, and `rgb_align.py` is what certifies that the pixels and the projected boxes
describe one camera (median centre error 1.0-1.5 px).

HONEST LIMIT, v1: the render shows the scene as SAMPLED. Tool calls mutate symbolic
state -- an object gets held, placed, opened -- and those mutations are NOT yet pushed
back into Kit, so a grasped mug still renders on the table. Text observations are always
correct; pixels are correct for navigation and identification, which is what the
selection experiment needs, and misleading for verifying manipulation. Fix before any
multimodal manipulation result is quoted.

Usage
-----
    # human, text-only rendering of detections (no Kit needed, instant)
    python -m rlinf.envs.behavior.interactive_env --activity picking_up_trash

    # with pixels
    python -m rlinf.envs.behavior.interactive_env --activity picking_up_trash --rgb
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from rlinf.envs.behavior.nl_goal import render_goal_nl
from rlinf.envs.behavior.symbolic_world import SymbolicACI, SymbolicWorld

# WASD, plus the tilt and turn primitives. Kept here so the human CLI and an agent
# front-end cannot drift apart on what the movement vocabulary is.
CAMERA_TOOLS = ("move_ahead", "move_back", "strafe_left", "strafe_right",
                "turn_left", "turn_right", "look_up", "look_down")
KEYMAP = {"w": "move_ahead", "s": "move_back", "a": "strafe_left", "d": "strafe_right",
          "q": "turn_left", "e": "turn_right", "r": "look_up", "f": "look_down"}


class InteractiveEnv:
    """Thin, uniform front-end over `SymbolicACI` for all three operator kinds."""

    def __init__(self, activity: str, obs_mode: str = "detect",
                 trace_path: Optional[str] = None):
        self.world = SymbolicWorld(activity)
        self.aci = SymbolicACI(self.world, obs_mode=obs_mode)
        self.activity = activity
        self.goal_nl, _ = render_goal_nl(activity)
        self.trace_path = trace_path
        self.steps = 0
        self._t0 = time.time()
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            open(trace_path, "w").close()

    # -- observation ---------------------------------------------------- #
    def observe(self) -> dict:
        obs = self.aci.observe()
        obs["goal"] = self.goal_status()
        return obs

    def goal_status(self) -> dict:
        st = self.world.goal_status()
        return {"satisfied": len(st.get("satisfied", [])),
                "total": len(st.get("satisfied", [])) + len(st.get("unsatisfied", [])),
                "success": bool(self.world.is_success())}

    # -- action --------------------------------------------------------- #
    def step(self, tool: str, **args) -> dict:
        """One tool call. `select=` is the uniform selection channel."""
        if "select" in args:
            # Every operator kind lands here: a handle, a point or a box, all resolved
            # by the same function with the same refusals.
            args["name"] = str(args.pop("select"))
        fn = getattr(self.aci, tool, None)
        if fn is None:
            out = {"ok": False, "reason": f"no such tool {tool!r}"}
        else:
            try:
                res = fn(**args) if args else fn()
                out = {"ok": bool(res.ok), "reason": res.reason,
                       "observation": res.observation}
            except TypeError as e:
                out = {"ok": False, "reason": f"bad_arguments: {e}"}
        self.steps += 1
        rec = {"step": self.steps, "tool": tool, "args": args, "ok": out["ok"],
               "reason": out.get("reason", ""), "view": self.aci.view.state(),
               "goal": self.goal_status(), "t": round(time.time() - self._t0, 2)}
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return out


def _render_detections(obs: dict) -> str:
    lines = [f"  view: facing {obs['view']['facing_deg']} deg, "
             f"pitch {obs['view']['pitch_deg']} deg   "
             f"goal {obs['goal']['satisfied']}/{obs['goal']['total']}"]
    dets = obs.get("detections", [])
    if not dets:
        lines.append("  (nothing in view)")
    for d in dets:
        b = d["bbox"]
        lines.append(f"  {d['det']:>4}  {d['category']:<22} "
                     f"bbox=[{b[0]:>3},{b[1]:>3},{b[2]:>3},{b[3]:>3}]  score={d['score']}")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activity", required=True)
    ap.add_argument("--obs-mode", default="detect",
                    choices=["detect", "detect_scope", "fov", "full"])
    ap.add_argument("--trace", default=None, help="write a JSONL trace here")
    args = ap.parse_args()

    env = InteractiveEnv(args.activity, obs_mode=args.obs_mode, trace_path=args.trace)
    print(f"\n{args.activity}   obs_mode={args.obs_mode}")
    print(f"goal: {env.goal_nl}\n")
    print("keys: w/s ahead|back   a/d strafe   q/e turn   r/f look up|down")
    print("      observe | grasp <sel> | place_inside <sel> <sel> | open <sel> | end_task")
    print("      <sel> is d3, or a point '160,120', or a box '10,20,60,80'\n")
    print(_render_detections(env.observe()))

    while True:
        try:
            raw = input("\n> ").strip()
        except EOFError:
            break
        if not raw or raw in ("quit", "exit"):
            break
        if raw in KEYMAP:
            out = env.step(KEYMAP[raw])
        else:
            parts = raw.split()
            tool, rest = parts[0], parts[1:]
            if tool == "observe":
                out = {"ok": True, "reason": ""}
            elif len(rest) == 1:
                out = env.step(tool, select=rest[0])
            elif len(rest) == 2:
                out = env.step(tool, select=rest[0], **{"container": rest[1]})
            else:
                out = env.step(tool)
        if not out["ok"]:
            print(f"  REFUSED: {out['reason']}")
        elif out.get("reason"):
            print(f"  ok: {out['reason']}")
        print(_render_detections(env.observe()))
        if env.goal_status()["success"]:
            print("\n*** GOAL SATISFIED ***")
            break


if __name__ == "__main__":
    _main()
