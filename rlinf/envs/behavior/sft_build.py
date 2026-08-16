"""Convert BEHAVIOR expert trajectories into tool-call SFT rows (M3, step 1).

Turns an :class:`~rlinf.envs.behavior.expert_planner.ExpertTrajectory` (a
privileged planner rollout through :class:`SemanticACI`) into the JSONL row shape
consumed by RLinf's ``EqaToolCallSftDataset`` (``rlinf/data/datasets/vlm.py``):

  {"activity", "task", "success", "num_turns",
   "prompt_messages": [{system}, {user}],           # canonical, rendered once
   "tool_steps": [{"name", "arguments", "result_text"}, ...],
   "unsupported_predicates", "answer"}

and emits the matching **tool-schema sidecar** (``*.tools.json``) so the training
dataset renders the prompt without importing OmniGibson.

Design (kept deliberately sim-free so it unit-tests and dumps the schema without
booting IsaacSim -- it only reads plain attributes off the trajectory object):

  * **observe-first grounding.** Every SFT trajectory starts with one ``observe``
    turn whose (masked) response is the full symbolic observation captured at
    reset -- this is where the policy learns the object *names* the later calls
    ground onto. The single BEHAVIOR tool that consumes new (post-transition)
    object names, ``dice``, takes the *whole* object and slices+dices internally,
    so one initial observe suffices under full observability. Re-observation
    between turns is a documented refinement for the partial-obs study (M4).
  * **response contract.** ``observe`` returns the full observation JSON; every
    action tool returns a compact ``{"ok", "reason"}`` ack; ``end_task`` returns
    ``{"ok", "success"}``. The GRPO agent loop MUST serialize tool responses the
    same way for SFT to be a faithful cold-start (mirrors ``spatialcode.sft_format``
    flat-continuation argument in code-doc/0607 §4.5).
  * **honest labels.** ``success`` and every step's ``ok`` come straight from the
    trajectory (real BDDL ``goal_status`` / real ``ToolResult.ok``). Filtering to
    successful demonstrations is the harvester's job (step 2); this module only
    transcribes.
"""
from __future__ import annotations

import json
from typing import Any


# --------------------------------------------------------------------------- #
# Tool schema -- flat {name, description, parameters} list (matches the EQA
# sidecar shape the dataset feeds to `apply_chat_template(tools=...)`).
# Parameter names MUST match the SemanticACI kwargs the planner emits, so the
# rendered arguments dict validates against the schema.
# --------------------------------------------------------------------------- #
def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_NAME = {"name": {"type": "string", "description": "sim object name, as reported by observe()"}}
_SYS = {"system_name": {"type": "string", "description": "particle/substance system name"}}


def tool_schemas(minimal: bool = False) -> list[dict]:
    """The BEHAVIOR semantic-tool schema. Order is stable (sidecar determinism).

    @minimal strips the procedural clauses from the descriptions -- see
    `_MINIMAL_DESCRIPTIONS`. The parameters are untouched, so the same calls
    validate either way and only the guidance changes.
    """
    out = [
        {"name": "observe",
         "description": "Return a symbolic observation of task-relevant objects "
                        "(names, categories, positions, states, what is held). "
                        "Call this first to learn the object names.",
         "parameters": _obj({}, [])},
        {"name": "go_to",
         "description": "Navigate the robot base next to an object, facing it. "
                        "Required before grasp/open/close/toggle/place/transform.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "grasp",
         "description": "Pick up a nearby object (must be at it via go_to first).",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "release",
         "description": "Release the currently held object.",
         "parameters": _obj(dict(_NAME), [])},
        {"name": "place_on",
         "description": "Place the held object on top of a surface object.",
         "parameters": _obj({**_NAME, "surface": {"type": "string",
                             "description": "sim name of the support surface"}},
                            ["name", "surface"])},
        {"name": "place_inside",
         "description": "Place the held object inside a container object.",
         "parameters": _obj({**_NAME, "container": {"type": "string",
                             "description": "sim name of the container"}},
                            ["name", "container"])},
        {"name": "open",
         "description": "Open a nearby openable container (e.g. fridge, cabinet).",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "close",
         "description": "Close a nearby openable container.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "toggle_on",
         "description": "Turn a nearby toggleable device on.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "toggle_off",
         "description": "Turn a nearby toggleable device off.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "cook",
         "description": "Cook a nearby cookable object.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "spray",
         "description": "Cover a nearby object with a substance system.",
         "parameters": _obj({**_NAME, **_SYS}, ["name", "system_name"])},
        {"name": "uncover",
         "description": "Remove a substance system from a nearby object.",
         "parameters": _obj({**_NAME, **_SYS}, ["name", "system_name"])},
        {"name": "fill",
         "description": "Fill a nearby container with a particle/fluid system.",
         "parameters": _obj({**_NAME, **_SYS}, ["name", "system_name"])},
        {"name": "slice",
         "description": "Slice a nearby sliceable object into its halves.",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "dice",
         "description": "Mince a nearby object into its diced particle system "
                        "(slices first if the object is only sliceable).",
         "parameters": _obj(dict(_NAME), ["name"])},
        {"name": "end_task",
         "description": "Declare the task complete. Call when the goal is achieved.",
         "parameters": _obj({}, [])},
    ]
    if minimal:
        for t in out:
            if t["name"] in _MINIMAL_DESCRIPTIONS:
                t["description"] = _MINIMAL_DESCRIPTIONS[t["name"]]
    return out


# The camera primitives are a SEPARATE group, offered only under `obs_mode=fov`. They
# are not part of the 17 semantic tools: they change no BDDL literal, only what the
# next `observe` reports, so every expert plan and every earlier baseline stays valid.
# Keeping them out of `tool_schemas()` is what guarantees the other modes are
# byte-identical to the runs already reported.
def camera_schemas() -> list[dict]:
    """Viewpoint control. Order is stable, same as `tool_schemas`."""
    return [
        {"name": "turn_left",
         "description": "Rotate the camera 90 degrees to the left, in place.",
         "parameters": _obj({}, [])},
        {"name": "turn_right",
         "description": "Rotate the camera 90 degrees to the right, in place.",
         "parameters": _obj({}, [])},
        {"name": "move_ahead",
         "description": "Step 0.5 m in the direction you are facing.",
         "parameters": _obj({}, [])},
        {"name": "look_down",
         "description": "Tilt the camera 30 degrees down, to see low surfaces "
                        "and the floor.",
         "parameters": _obj({}, [])},
        {"name": "look_up",
         "description": "Tilt the camera 30 degrees up, undoing a look_down.",
         "parameters": _obj({}, [])},
    ]


CAMERA_OBS_MODES = ("fov", "fov_distract", "detect", "detect_scope")


def all_schemas(obs_mode: str = "full", minimal: bool = False) -> list[dict]:
    """Every tool the policy may call under @obs_mode."""
    return (tool_schemas(minimal)
            + (camera_schemas() if obs_mode in CAMERA_OBS_MODES else []))


def dump_tool_schemas(path: str) -> None:
    with open(path, "w") as f:
        json.dump(tool_schemas(), f, indent=1)


# --------------------------------------------------------------------------- #
# Prompt (system + user), rendered ONCE -- mirrors the EQA builder so the SFT
# prompt matches the GRPO rollout exactly.
# --------------------------------------------------------------------------- #
BEHAVIOR_SYSTEM_PROMPT = (
    "You are an embodied household robot completing a long-horizon activity in a "
    "simulated home. You act over multiple turns, calling exactly one tool per "
    "turn. Object names come from the observation, so call `observe` first to "
    "learn what is in the scene and where you are. Then navigate with `go_to` "
    "before interacting (grasp / open / close / toggle / place / transform), and "
    "respect ordering: open a container before placing into it and close it "
    "afterwards if the goal requires it. When every goal condition is met, call "
    "`end_task`."
)

# E1 (brainstorm/260815): the prompt above is a skill.md and we wrote it. Skill1's
# ALFWorld ablation prices procedural knowledge in context at 16.6 points (97.5 ->
# 80.9 without the library), and the prompt above states the control loop, the
# proximity precondition, the one real ordering constraint and the termination rule.
# Every observability condition we then added shipped with a matching hint, so we
# introduced a difficulty and published its solution together, then measured that it
# was not difficult.
#
# The line drawn here is INTERFACE vs DOMAIN. Kept: you are a robot, you call one tool
# per turn -- harness facts the policy cannot infer and which are not what we claim to
# measure. Removed: what to call first, what must precede what, what order containers
# want, when to stop. Those are the plan.
BEHAVIOR_SYSTEM_PROMPT_MINIMAL = (
    "You are an embodied household robot in a simulated home. You act over multiple "
    "turns, calling exactly one tool per turn."
)

# Same cut applied to the tool schema. A description may say what a tool DOES; it may
# not say when to call it or what it requires. "nearby" goes too -- it is the
# proximity precondition restated.
_MINIMAL_DESCRIPTIONS = {
    "observe": "Return a symbolic observation of task-relevant objects (names, "
               "categories, positions, states, what is held).",
    "go_to": "Navigate the robot base next to an object, facing it.",
    "grasp": "Pick up an object.",
    "open": "Open an openable container (e.g. fridge, cabinet).",
    "close": "Close an openable container.",
    "toggle_on": "Turn a toggleable device on.",
    "toggle_off": "Turn a toggleable device off.",
    "cook": "Cook a cookable object.",
    "spray": "Cover an object with a substance system.",
    "uncover": "Remove a substance system from an object.",
    "fill": "Fill a container with a particle/fluid system.",
    "slice": "Slice a sliceable object into its halves.",
    "dice": "Mince an object into its diced particle system.",
    "end_task": "Declare the task complete.",
}



def goal_atoms_to_lines(atoms: Any) -> list[str]:
    """Render BDDL ground goal atoms as readable ``pred(arg, ...)`` lines.

    Sim-free: only reads ``atom.body`` (a nested list like ``["inside", a, b]`` or
    ``["not", ["open", a]]``). Gives the policy the task objective (this is the
    task spec, NOT the answer -- the answer is the tool sequence)."""
    lines = []
    for atom in atoms:
        body = list(getattr(atom, "body", atom))
        neg = ""
        if body and body[0] == "not":
            neg = "not "
            body = list(body[1])
        pred, args = body[0], body[1:]
        lines.append(f"{neg}{pred}({', '.join(str(a) for a in args)})")
    return lines


def build_prompt_messages(activity: str, goal_lines: list[str],
                          obs_mode: str = "full",
                          goal_nl: str | None = None,
                          prompt_mode: str = "guided") -> list[dict]:
    """The one prompt builder, shared by the SFT harvester and the GRPO rollout.

    ``goal_nl`` selects the **natural-language goal condition** (see
    ``nl_goal.py``): the sentence replaces the ground-atom block entirely, so the
    policy has to infer which predicates and objects the instruction denotes
    instead of transliterating atoms into same-named tools. Passing ``None`` keeps
    the ground-atom condition, which is the ablation baseline and what the current
    SFT checkpoint was trained on. Reward is identical either way -- it comes from
    the simulator's BDDL evaluation, not from this text."""
    if goal_nl is not None:
        body = f"You are done when this is true: {goal_nl}"
    else:
        goal_block = "\n".join(f"  - {g}" for g in goal_lines) or "  (none)"
        body = ("Complete the activity so that all of these goal conditions hold:\n"
                f"{goal_block}")
    # The mode word alone does not say what `object` gates on, and a policy that
    # cannot know a shut cupboard has an inside is being tested on guessing, not
    # planning. Stating the rule is not goal leakage -- it says nothing about which
    # objects matter. `full` and `partial` keep a byte-identical prompt so the
    # measured 0801 numbers stay comparable.
    obs_line = f"Observability: {obs_mode}"
    if prompt_mode == "minimal":
        # The per-mode clauses below are the per-condition skill entries: each one
        # names the tool that defeats the gate it describes. Under `minimal` the
        # policy is told WHICH mode it is in and must work out what that implies.
        pass
    elif obs_mode == "object":
        obs_line += (" -- you see only what is in this room and not shut inside a "
                     "closed container; open a container to see what is in it")
    elif obs_mode in CAMERA_OBS_MODES and prompt_mode == "guided":
        # State the INTERFACE, never the contents. A policy cannot infer that objects
        # are addressed by detection handle rather than by name, and testing whether it
        # guesses our calling convention is not the question. It is told nothing about
        # WHICH objects matter, which is the part that is hard and the part no
        # sentence can give away.
        #
        # The four clauses are assembled from shared pieces ON PURPOSE. These modes
        # form a 2x2 (see `SymbolicACI.OBS_MODES`) and the whole experiment is a
        # difference between two of them, so the prompts must differ by exactly the
        # sentence that describes the axis being varied -- and by nothing else. Four
        # independently written paragraphs would have made the prompt a fifth
        # uncontrolled variable.
        SWEEP = ("Use turn_left / turn_right to sweep, move_ahead to approach, "
                 "look_down for low surfaces, then observe again.")
        # The distractor axis.
        MIXED = ("Scene furniture and task objects are both reported and are not "
                 "distinguished.")
        if obs_mode in ("detect", "detect_scope"):
            obs_line += (" -- observe() returns detections in the camera's view: a "
                         "handle (d1, d2, ...), a bounding box, a category and a "
                         "score. ")
            if obs_mode == "detect":
                obs_line += MIXED + " "
            obs_line += ("Refer to an object by its handle, e.g. grasp(name='d3'), or "
                         "by a box 'x0,y0,x1,y1'. A handle is valid only until the "
                         "camera moves; observe again after turning or navigating. "
                         + SWEEP)
        else:
            obs_line += " -- you see only what is in the camera's field of view. "
            if obs_mode == "fov_distract":
                obs_line += MIXED + " "
            obs_line += (SWEEP + " You can only go_to an object you have already "
                         "seen.")
    user = (
        f"Activity: {activity.replace('_', ' ')}\n"
        f"{obs_line}\n"
        f"{body}"
    )
    return [
        {"role": "system",
         "content": (BEHAVIOR_SYSTEM_PROMPT_MINIMAL if prompt_mode == "minimal"
                     else BEHAVIOR_SYSTEM_PROMPT)},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Trajectory -> row
# --------------------------------------------------------------------------- #
def _result_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def trajectory_to_row(traj: Any, initial_obs: dict, goal_lines: list[str],
                      obs_mode: str = "full", goal_nl: str | None = None) -> dict:
    """Transcribe one ExpertTrajectory (duck-typed) into an SFT row.

    ``traj`` needs ``.activity``, ``.task_description``, ``.success``,
    ``.unsupported`` and ``.steps`` (each with ``.tool``, ``.args``, ``.ok``,
    ``.reason``). ``initial_obs`` is ``aci.observe()`` captured at reset, BEFORE
    the planner mutates the scene.

    ``goal_nl`` switches the prompt to the natural-language goal condition. Rows
    already on disk do not need re-harvesting to change it -- the trajectory is
    independent of how the goal was phrased, so ``relabel_sft_nl.py`` rewrites the
    prompt in place.
    """
    tool_steps = [{
        "name": "observe", "arguments": {},
        "result_text": _result_text(initial_obs),
    }]
    for s in traj.steps:
        tool_steps.append({
            "name": s.tool,
            "arguments": dict(s.args or {}),
            "result_text": _result_text({"ok": bool(s.ok), "reason": s.reason}),
        })
    tool_steps.append({
        "name": "end_task", "arguments": {},
        "result_text": _result_text({"ok": bool(traj.success),
                                     "success": bool(traj.success)}),
    })
    return {
        "activity": traj.activity,
        "task": getattr(traj, "task_description", "") or activity_to_task(traj.activity),
        "success": bool(traj.success),
        "num_turns": len(tool_steps),
        "prompt_messages": build_prompt_messages(traj.activity, goal_lines, obs_mode,
                                                 goal_nl=goal_nl),
        "tool_steps": tool_steps,
        "unsupported_predicates": list(getattr(traj, "unsupported", []) or []),
        "answer": "",  # BEHAVIOR reward is BDDL success, not a categorical answer
    }


def activity_to_task(activity: str) -> str:
    return activity.replace("_", " ")


# --------------------------------------------------------------------------- #
# CLI: dump the schema sidecar, and (with no args) run a sim-free self-check.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from types import SimpleNamespace

    if len(sys.argv) > 1:
        dump_tool_schemas(sys.argv[1])
        print(f"wrote {len(tool_schemas())} tool schemas -> {sys.argv[1]}")
        sys.exit(0)

    # Sim-free smoke: fabricate a trajectory + obs and render one row.
    fake = SimpleNamespace(
        activity="carrying_in_groceries", task_description="", success=True,
        unsupported=[],
        steps=[
            SimpleNamespace(tool="go_to", args={"name": "fridge_xyz"}, ok=True, reason="now 0.9m from fridge_xyz"),
            SimpleNamespace(tool="open", args={"name": "fridge_xyz"}, ok=True, reason="Open=True"),
            SimpleNamespace(tool="grasp", args={"name": "bratwurst_1"}, ok=True, reason="holding bratwurst_1"),
            SimpleNamespace(tool="place_inside", args={"name": "bratwurst_1", "container": "fridge_xyz"}, ok=True, reason="Inside=True"),
            SimpleNamespace(tool="close", args={"name": "fridge_xyz"}, ok=True, reason="Open=False"),
        ],
    )
    obs = {"obs_mode": "full", "robot_xy": [1.0, 2.0], "held": None,
           "objects": [{"name": "fridge_xyz", "category": "fridge", "states": {"Open": False}}]}
    goal = goal_atoms_to_lines([
        SimpleNamespace(body=["inside", "bratwurst.n.01_1", "fridge.n.01_1"]),
        SimpleNamespace(body=["not", ["open", "fridge.n.01_1"]]),
    ])
    row = trajectory_to_row(fake, obs, goal)
    assert row["tool_steps"][0]["name"] == "observe"
    assert row["tool_steps"][-1]["name"] == "end_task"
    assert row["num_turns"] == len(fake.steps) + 2
    assert row["prompt_messages"][0]["role"] == "system"
    assert "inside(bratwurst.n.01_1, fridge.n.01_1)" in row["prompt_messages"][1]["content"]
    assert "not open(fridge.n.01_1)" in row["prompt_messages"][1]["content"]
    names = {t["name"] for t in tool_schemas()}
    assert {s.tool for s in fake.steps} <= names, "planner tools not covered by schema"
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print("\nself-check OK: schema covers planner tools; row well-formed")
