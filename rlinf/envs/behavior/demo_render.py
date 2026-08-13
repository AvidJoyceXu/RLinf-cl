"""Render one BEHAVIOR-TextWorld episode as a 2D demo: trajectory, primitives, status.

The Habitat/VLN-CE side of RLinf already has this
(``rlinf/envs/habitat/extensions/utils.py`` draws the agent on a real top-down
navmesh with fog of war). This is the BEHAVIOR counterpart, minus the pixels: no
RGB, no renderer, no GPU -- an episode costs milliseconds, so a demo is a script
run rather than a job.

Three things are shown, and nothing else:

  1. **the movement trajectory** -- where the agent went, in order;
  2. **the primitives it called** -- the tool log, with rejections marked;
  3. **task completion** -- goal conditions ticking off, scored by the real
     ``bddl.activity.evaluate_goal_conditions``.

There is deliberately no reasoning or memory panel: the policy has neither. It
emits a bare ``<tool_call>`` with no thinking trace and carries no state across
turns beyond the message history (see ``0730 - thinking and memory in the
tool-call trajectory.md``). A panel showing "the agent's plan" would be inventing
one.

HONESTY -- what the picture is and is not
-----------------------------------------
**The layout is a schematic, not geometry.** BEHAVIOR-TextWorld has no
coordinates: its state is a set of BDDL ground literals, and ``observe()`` is
careful to report none (inventing xy would be exactly the plausible-looking fake
the project's red lines forbid). So the xy here is *drawn by this file* from the
containment and ``inroom`` structure -- rooms are boxes, an object sits near
whatever supports it. Distances on this canvas mean nothing. What is faithful is
the **topology and its changes**: which room the agent is in, what supports what,
and when that changes.

Everything else on the canvas comes from the environment, unmodified:

  * the tool sequence and each accept/reject is whatever the driver actually did;
  * ``goal conditions`` ticks come from ``world.goal_status()`` -- bddl's own
    evaluator over the unground conditions, the same call the reward uses;
  * ``ground atoms`` ticks come from ``world.holds()`` on the option-0 expansion,
    which is what the prompt's goal block renders. A quantified goal has several
    ground options and only one need hold, so this row is *a* reading of progress
    while the row above it is the score.

The success badge is ``world.is_success()``. It can read FAILED while every
ground atom shows a tick, and that is not a bug -- see above.

Usage
-----
    # the expert -- no GPU, no checkpoint, always runnable
    python -m rlinf.envs.behavior.demo_render --activity picking_up_trash

    # the SFT cold start
    python -m rlinf.envs.behavior.demo_render --activity picking_up_trash \\
        --driver sft --model /data/behavior-data/sft/hf_step120

    # a frontier model over the API (needs ROBOPARTY_API_KEY in the environment)
    python -m rlinf.envs.behavior.demo_render --activity picking_up_trash \\
        --driver api --model chat-fast --goal-format nl

Writes a single self-contained HTML file (no CDN, no external assets) plus, with
``--svg``, a static SVG of the final frame for pasting into a paper.
"""
from __future__ import annotations

import argparse
import html as _html
import inspect
import json
import math
import os
from collections import defaultdict
from typing import Any, Optional

from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.symbolic_world import (
    SymbolicACI,
    SymbolicWorld,
    properties_of,
)
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld

# The canonical 17. Read from the schema rather than retyped, so the recorder can
# never fall out of step with the action space the policy is prompted with.
TOOL_NAMES = tuple(s["name"] for s in sb.tool_schemas())


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
class RecordingACI:
    """Transparent proxy over ``SymbolicACI`` that snapshots state after each tool.

    A proxy rather than a subclass or a patched rollout loop: the drivers
    (``symbolic_expert``, ``rft_sample.rollout_once``, the API harness) all reach
    for tools with ``getattr(aci, name)(**kwargs)``, so wrapping the object means
    every driver is recorded with no change to any of them. That matters for
    honesty as much as for effort -- the demo must show the same episode the
    benchmark would have scored, not a re-run through demo-specific code.
    """

    def __init__(self, aci: SymbolicACI):
        self._aci = aci
        self.world = aci.world
        self.frames: list[dict] = []
        self._snap(None, {}, None)

    def __getattr__(self, item: str):
        attr = getattr(self._aci, item)
        if item not in TOOL_NAMES:
            return attr

        def wrapped(*a, **kw):
            res = attr(*a, **kw)
            self._snap(item, _bind(attr, a, kw), res)
            return res

        return wrapped

    # ------------------------------------------------------------------ #
    def _snap(self, tool: Optional[str], args: dict, res: Any) -> None:
        w = self.world
        gs = w.goal_status()
        n_sat, n_unsat = len(gs["satisfied"]), len(gs["unsatisfied"])

        if res is None:                       # frame 0: the reset state
            ok, reason = None, ""
        elif isinstance(res, dict):           # observe() returns the obs dict
            ok, reason = True, f"{len(res.get('objects', []))} objects visible"
        else:
            ok, reason = bool(res.ok), res.reason or ""

        self.frames.append({
            "tool": tool,
            "args": {k: v for k, v in args.items() if v is not None},
            "ok": ok,
            "reason": reason,
            "at": self._aci._at,
            "held": self._aci._held,
            "supports": {n: list(s) for n in w.scope_names
                         if (s := w.support_of(n))},
            "conds": [i in set(gs["satisfied"])
                      for i in range(n_sat + n_unsat)],
            "atoms": _atom_truth(w),
            "n_sat": n_sat,
            "n_goal": n_sat + n_unsat,
            "success": w.is_success(),
        })


def _bind(fn, a: tuple, kw: dict) -> dict:
    """Name positional arguments, so the log reads ``go_to(name=...)`` either way."""
    try:
        bound = inspect.signature(fn).bind(*a, **kw)
        return dict(bound.arguments)
    except TypeError:
        return dict(kw)


def _atom_truth(world: SymbolicWorld) -> list[bool]:
    # `_split` is symbolic_expert's, imported rather than reimplemented: it is the
    # one place that knows how a negated ground atom is shaped, and two copies of
    # that rule would eventually disagree about what a tick means.
    from rlinf.envs.behavior.symbolic_expert import _split

    out = []
    for atom in world.ground_goal_atoms():
        positive, pred, args = _split(atom)
        out.append(world.holds(pred, args) == positive)
    return out


def _atom_labels(world: SymbolicWorld) -> list[str]:
    from rlinf.envs.behavior.symbolic_expert import _split

    disp = world.to_display
    out = []
    for atom in world.ground_goal_atoms():
        positive, pred, args = _split(atom)
        shown = ", ".join(disp.get(a, a) for a in args)
        out.append(("" if positive else "not ") + f"{pred}({shown})")
    return out


def _cond_labels(world: SymbolicWorld) -> list[str]:
    def flatten(node) -> str:
        if isinstance(node, list):
            return "(" + " ".join(flatten(c) for c in node) + ")"
        return str(node)

    return [flatten(getattr(c, "body", c)) for c in (world.goal_conditions or [])]


# --------------------------------------------------------------------------- #
# layout -- schematic, see the module docstring
# --------------------------------------------------------------------------- #
CELL_W, CELL_H = 148, 104
ROOM_PAD = 26
ROOM_GAP = 34
MARGIN = 22
NO_ROOM = "(no room)"


def _anchors(world: SymbolicWorld) -> list[str]:
    """Objects that get their own fixed slot: rooms' fixed furniture.

    ``inroom`` entries (floors, and whatever BDDL nails down) plus anything the
    property table calls a ``sceneObject``. Everything else is drawn as a satellite
    of whatever supports it, which is what makes ``place_inside`` visible: the
    marker moves to its new container.
    """
    out = []
    for name in world.scope_names:
        if name == world.agent:
            continue
        if name in world.rooms or "sceneObject" in properties_of(name):
            out.append(name)
    return sorted(out)


def build_layout(world: SymbolicWorld) -> dict:
    anchors = _anchors(world)
    by_room: dict[str, list[str]] = defaultdict(list)
    for name in anchors:
        by_room[world.room_of(name) or NO_ROOM].append(name)
    if not by_room:
        by_room[NO_ROOM] = []

    # Room boxes on a grid, all the same size so the picture does not tilt.
    max_cells = max((len(v) for v in by_room.values()), default=1) or 1
    cols_in = max(1, math.ceil(math.sqrt(max_cells)))
    rows_in = max(1, math.ceil(max_cells / cols_in))
    room_w = cols_in * CELL_W + 2 * ROOM_PAD
    room_h = rows_in * CELL_H + 2 * ROOM_PAD + 18      # +18 for the room label

    room_names = sorted(by_room)
    grid_cols = max(1, math.ceil(math.sqrt(len(room_names))))

    rooms, slots = [], {}
    for r, room in enumerate(room_names):
        rx = MARGIN + (r % grid_cols) * (room_w + ROOM_GAP)
        ry = MARGIN + (r // grid_cols) * (room_h + ROOM_GAP)
        rooms.append({"name": room, "x": rx, "y": ry, "w": room_w, "h": room_h})
        for k, name in enumerate(by_room[room]):
            slots[name] = [
                rx + ROOM_PAD + (k % cols_in) * CELL_W + CELL_W / 2,
                ry + ROOM_PAD + 18 + (k // cols_in) * CELL_H + CELL_H / 2,
            ]

    width = MARGIN * 2 + grid_cols * room_w + (grid_cols - 1) * ROOM_GAP
    grid_rows = math.ceil(len(room_names) / grid_cols)
    height = MARGIN * 2 + grid_rows * room_h + (grid_rows - 1) * ROOM_GAP
    return {"rooms": rooms, "slots": slots, "width": width, "height": height}


class RingSlots:
    """Sticky orbit index per object, so nothing drifts unless it actually moved.

    Without this, ring slots are the enumeration order of the current sibling list,
    and picking one can off the floor renumbers the two left behind -- they slide
    across the canvas having done nothing. On an animated demo that reads as
    movement, which is a small lie of exactly the kind this file is trying not to
    tell. An object keeps its index while it keeps its support.
    """

    def __init__(self):
        self._at: dict[str, tuple] = {}

    def index(self, parent: str, kids: list) -> dict:
        used, fresh = set(), []
        for k in kids:
            prev = self._at.get(k)
            if prev and prev[0] == parent and prev[1] not in used:
                used.add(prev[1])
            else:
                fresh.append(k)
        for k in fresh:
            i = 0
            while i in used:
                i += 1
            used.add(i)
            self._at[k] = (parent, i)
        return {k: self._at[k][1] for k in kids}


def frame_positions(world: SymbolicWorld, frame: dict, layout: dict,
                    rings: RingSlots, prev_agent: Optional[list] = None) -> tuple:
    """Object -> xy plus the agent xy, for one frame.

    Objects orbit their **immediate** support, resolved outward from the fixed
    slots. Orbiting the ultimate anchor instead would be simpler and wrong: a can
    placed inside an ashcan that itself stands on the floor would stay ringed around
    the floor, and ``place_inside`` -- the whole point of the episode -- would show
    nothing move.
    """
    slots, supports = layout["slots"], frame["supports"]
    pos = {n: list(xy) for n, xy in slots.items()}
    held = frame["held"]

    children: dict[str, list[str]] = defaultdict(list)
    for name in sorted(world.scope_names):
        if name == world.agent or name in slots or name == held:
            continue
        sup = supports.get(name)
        # A chain through the held object would drag its contents along to the
        # agent; `grasp` clears position anyway, so this is belt and braces.
        if sup and sup[1] != held:
            children[sup[1]].append(name)

    frontier, placed, depth = list(slots), set(slots), 0
    while frontier and depth < 6:
        nxt = []
        for parent in frontier:
            cx, cy = pos[parent]
            kids = [k for k in children.get(parent, []) if k not in placed]
            for name, slot in rings.index(parent, kids).items():
                ring, idx = divmod(slot, 8)
                ang = (idx / 8) * 2 * math.pi + ring * 0.4 + depth * 0.7
                rad = max(17, 34 - depth * 8) + ring * 18
                pos[name] = [cx + rad * math.cos(ang), cy + rad * 0.62 * math.sin(ang)]
                placed.add(name)
                nxt.append(name)
        frontier, depth = nxt, depth + 1

    # The agent, resolved BEFORE the held object is placed: after a grasp the agent
    # is "at" the thing in its own hand, and following that would make it chase its
    # hand across the canvas. Grasping does not move you, so it holds position.
    at = frame["at"]
    if at and at != held and at in pos:
        agent = [pos[at][0], pos[at][1] - 20]
    elif prev_agent is not None:
        agent = list(prev_agent)
    elif layout["rooms"]:
        r = layout["rooms"][0]
        agent = [r["x"] + r["w"] / 2, r["y"] + r["h"] / 2]
    else:
        agent = [layout["width"] / 2, layout["height"] / 2]

    if held:
        pos[held] = [agent[0], agent[1] - 11]

    # Whatever is left has neither slot nor support: BDDL `future` products that do
    # not exist yet, mostly. A strip below the rooms, rather than silently dropped.
    for k, name in enumerate(n for n in sorted(world.scope_names)
                             if n != world.agent and n not in pos):
        pos[name] = [MARGIN + 24 + (k % 8) * 74,
                     layout["height"] + 26 + (k // 8) * 30]
    return pos, agent


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def drive_expert(rec: RecordingACI, world: SymbolicWorld, **_) -> str:
    """The privileged symbolic planner. No model, no GPU -- the demo that always runs."""
    from rlinf.envs.behavior.symbolic_expert import _plan_atom, _split

    parsed = [_split(a) for a in world.ground_goal_atoms()]
    parsed.sort(key=lambda t: t[0], reverse=True)
    for positive, pred, args in parsed:
        if world.holds(pred, args) == positive:
            continue
        for tool, kwargs in _plan_atom(world, positive, pred, args):
            getattr(rec, tool)(**kwargs)
    return "expert (privileged symbolic planner)"


def drive_sft(rec, world, activity, goal_lines, obs_mode, max_turns,
              temperature, model, **_) -> str:
    from rlinf.envs.behavior.rft_sample import load_model, rollout_once

    net, tok = load_model(model)
    rollout_once(net, tok, rec, activity, goal_lines, obs_mode,
                 max_turns, temperature, 1.0)
    return f"SFT checkpoint {os.path.basename(model.rstrip('/'))}"


def drive_api(rec, world, activity, goal_nl, goal_format, obs_mode, max_turns,
              temperature, model, base_url, **_) -> str:
    from openai import OpenAI

    from rlinf.envs.behavior.textworld_api_rollout import Usage, run_episode

    key = os.environ.get("ROBOPARTY_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOPARTY_API_KEY is not set. Export it in the shell that runs this "
            "(deliberately not a flag: flags land in shell history and `ps`).")
    client = OpenAI(api_key=key, base_url=base_url, timeout=120.0, max_retries=3)
    run_episode(client, model, activity, obs_mode, goal_format, max_turns,
                temperature, Usage(), aci=rec)
    return f"zero-shot {model}"


DRIVERS = {"expert": drive_expert, "sft": drive_sft, "api": drive_api}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_demo(activity: str, driver: str = "expert", obs_mode: str = "full",
               goal_format: str = "atoms", max_turns: int = 30,
               temperature: float = 1.0, model: str = "",
               base_url: str = "https://ai-gateway.roboparty.com/v1") -> dict:
    env = BehaviorTextWorld(activity, obs_mode=obs_mode)
    world = SymbolicWorld(activity)
    rec = RecordingACI(SymbolicACI(world, obs_mode=obs_mode))

    goal_lines = env.goal_lines
    label = DRIVERS[driver](
        rec=rec, world=world, activity=activity, goal_lines=goal_lines,
        goal_nl=env.goal_nl, goal_format=goal_format, obs_mode=obs_mode,
        max_turns=max_turns, temperature=temperature, model=model,
        base_url=base_url)

    layout = build_layout(world)
    disp = world.to_display
    goal_objs = {a for atom in world.ground_goal_atoms()
                 for a in _flat_names(atom) if a in world.scope_names}
    touched = {world.from_display.get(v, v)
               for f in rec.frames for v in f["args"].values()
               if isinstance(v, str)}

    out_frames, prev_agent, rings = [], None, RingSlots()
    for i, f in enumerate(rec.frames):
        pos, agent = frame_positions(world, f, layout, rings, prev_agent)
        prev_agent = agent
        objects = []
        for name, xy in sorted(pos.items()):
            kind = ("goal" if name in goal_objs else
                    "touched" if name in touched else "plain")
            objects.append({
                "label": disp.get(name, name),
                "x": round(xy[0], 1), "y": round(xy[1], 1),
                "kind": kind,
                "anchor": name in layout["slots"],
                "held": name == f["held"],
            })
        out_frames.append({
            "i": i,
            "tool": f["tool"],
            "call": _call_str(f["tool"], f["args"]),
            "ok": f["ok"],
            "reason": f["reason"],
            "at": disp.get(f["at"] or "", "") or None,
            "held": disp.get(f["held"] or "", "") or None,
            "agent": [round(v, 1) for v in agent],
            "objects": objects,
            "conds": f["conds"],
            "atoms": f["atoms"],
            "n_sat": f["n_sat"],
            "n_goal": f["n_goal"],
            "success": f["success"],
        })

    return {
        "activity": activity,
        "driver": label,
        "obs_mode": obs_mode,
        "goal_format": goal_format,
        "goal_nl": env.goal_nl,
        "goal_lines": goal_lines,
        "cond_labels": _cond_labels(world),
        "atom_labels": _atom_labels(world),
        "rooms": layout["rooms"],
        "width": layout["width"],
        "height": layout["height"] + 70,      # room for the stray strip
        "frames": out_frames,
        "success": out_frames[-1]["success"],
        "n_steps": len(out_frames) - 1,
    }


def _flat_names(atom) -> list:
    if isinstance(atom, str):
        return [atom]
    return [n for c in atom for n in _flat_names(c)]


def _call_str(tool: Optional[str], args: dict) -> str:
    if tool is None:
        return "reset"
    inner = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{tool}({inner})"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_svg(demo: dict, frame_index: int = -1) -> str:
    """Static SVG of one frame with the trajectory up to it -- for a paper figure."""
    frames = demo["frames"]
    fi = frame_index % len(frames)
    f = frames[fi]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
             f'{demo["width"]} {demo["height"]}" width="{demo["width"]}" '
             f'height="{demo["height"]}" font-family="ui-sans-serif,system-ui,sans-serif">',
             '<rect width="100%" height="100%" fill="#ffffff"/>']
    for room in demo["rooms"]:
        parts.append(
            f'<rect x="{room["x"]}" y="{room["y"]}" width="{room["w"]}" '
            f'height="{room["h"]}" rx="10" fill="#f4f5f7" stroke="#c8ccd4"/>'
            f'<text x="{room["x"] + 10}" y="{room["y"] + 18}" font-size="12" '
            f'fill="#6b7280">{_html.escape(room["name"])}</text>')

    pts = _trajectory(frames, fi)
    if len(pts) > 1:
        d = " ".join(f'{"M" if k == 0 else "L"}{x},{y}' for k, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="2" '
                     f'stroke-dasharray="5 4" opacity="0.75"/>')
        for k, (x, y) in enumerate(pts):
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#2563eb" '
                         f'opacity="0.15"/><text x="{x}" y="{y + 3}" font-size="8" '
                         f'text-anchor="middle" fill="#2563eb">{k}</text>')

    for o in f["objects"]:
        fill = {"goal": "#d97706", "touched": "#2563eb"}.get(o["kind"], "#9ca3af")
        r = 7 if o["anchor"] else 5
        parts.append(f'<circle cx="{o["x"]}" cy="{o["y"]}" r="{r}" fill="{fill}"/>'
                     f'<text x="{o["x"]}" y="{o["y"] - r - 3}" font-size="9" '
                     f'text-anchor="middle" fill="#374151">'
                     f'{_html.escape(o["label"])}</text>')

    ax, ay = f["agent"]
    parts.append(f'<polygon points="{ax},{ay - 11} {ax - 9},{ay + 7} {ax + 9},{ay + 7}" '
                 f'fill="#111827"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _trajectory(frames: list, upto: int) -> list:
    pts = []
    for f in frames[:upto + 1]:
        p = tuple(f["agent"])
        if not pts or pts[-1] != p:
            pts.append(p)
    return pts


def render_html(demo: dict) -> str:
    # The payload is embedded in a <script> block and carries environment-authored
    # strings (tool reasons). A literal "</script>" in one would end the block early,
    # so escape the only two characters that can do it. \u form is still valid JSON
    # and parses back identically.
    data = (json.dumps(demo, ensure_ascii=False)
            .replace("<", "\\u003c").replace("\u2028", "\\u2028"))
    title = f'{demo["activity"]} — BEHAVIOR-TextWorld demo'
    return _TEMPLATE.replace("__TITLE__", _html.escape(title)).replace("__DATA__", data)


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
/* Neutrals carry a slight blue bias toward the accent rather than sitting at pure
   grey, and semantic ok/bad stay separate from it so "satisfied" never reads as
   "selected". No webfont: the CSP blocks font CDNs, and inlining a face as a data
   URI would put a few hundred KB into every generated file for a page that is
   mostly predicates -- which the monospace stack serves better anyway. */
:root{--bg:#fafbfc;--fg:#1f2328;--mut:#667085;--line:#dde2ea;--card:#fff;
      --room:#eef1f6;--roomln:#ccd3e0;--ok:#15803d;--bad:#b91c1c;--acc:#2563eb;
      --goal:#c2680a;--plain:#98a2b3;--code:#eff2f7}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--fg:#e6e8eb;--mut:#98a2b3;
      --line:#232b36;--card:#141922;--room:#182029;--roomln:#2c3644;--ok:#4ade80;
      --bad:#f87171;--acc:#60a5fa;--goal:#fbbf24;--plain:#6b7484;--code:#171e28}}
:root[data-theme=dark]{--bg:#0e1116;--fg:#e6e8eb;--mut:#98a2b3;--line:#232b36;
      --card:#141922;--room:#182029;--roomln:#2c3644;--ok:#4ade80;--bad:#f87171;
      --acc:#60a5fa;--goal:#fbbf24;--plain:#6b7484;--code:#171e28}
:root[data-theme=light]{--bg:#fafbfc;--fg:#1f2328;--mut:#667085;--line:#dde2ea;
      --card:#fff;--room:#eef1f6;--roomln:#ccd3e0;--ok:#15803d;--bad:#b91c1c;
      --acc:#2563eb;--goal:#c2680a;--plain:#98a2b3;--code:#eff2f7}
*{box-sizing:border-box}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:19px;margin:0 0 2px;font-weight:650;text-wrap:balance}
.sub{color:var(--mut);font-size:12.5px;margin-bottom:14px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
       font-weight:650;letter-spacing:.03em;vertical-align:2px}
.b-ok{background:var(--ok);color:var(--bg)}.b-bad{background:var(--bad);color:var(--bg)}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,1fr);gap:14px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
         color:var(--mut);margin:0 0 8px;font-weight:650}
.mapbox{overflow:auto}
svg{display:block;max-width:100%;height:auto}
.ctl{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
button{font:inherit;font-size:12.5px;padding:3px 11px;border:1px solid var(--line);
       background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)}
input[type=range]{flex:1;min-width:140px;accent-color:var(--acc)}
.step{font-variant-numeric:tabular-nums;color:var(--mut);font-size:12px;min-width:74px}
.now{margin-top:8px;padding:7px 9px;background:var(--code);border-radius:7px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
     word-break:break-word}
.log{max-height:270px;overflow:auto;margin:0;padding:0;list-style:none;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.log li{padding:2px 6px;border-radius:4px;display:flex;gap:6px;cursor:pointer}
.log li:hover{background:var(--code)}
.log li.cur{background:var(--acc);color:var(--bg)}
.log .n{color:var(--mut);min-width:20px;text-align:right;
        font-variant-numeric:tabular-nums}
.log li.cur .n{color:var(--bg)}
.log .rej{color:var(--bad)}.log li.cur .rej{color:var(--bg)}
.chk{list-style:none;margin:0;padding:0;font-size:12px}
.chk li{display:flex;gap:7px;padding:2px 0;align-items:flex-start}
.chk .m{width:12px;flex:none;font-weight:700}
.chk .yes{color:var(--ok)}.chk .no{color:var(--mut)}
.chk code{background:var(--code);padding:0 4px;border-radius:4px;font-size:11px;
          word-break:break-all}
.bar{height:5px;background:var(--code);border-radius:3px;overflow:hidden;margin:6px 0 10px}
.bar>i{display:block;height:100%;background:var(--ok);transition:width .18s}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--mut);
        margin-top:8px}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.note{font-size:11.5px;color:var(--mut);margin-top:14px;line-height:1.55;
      border-top:1px solid var(--line);padding-top:10px}
.goalnl{font-size:13px;margin:0 0 8px}
</style>
<div class="wrap">
  <h1 id="ttl"></h1>
  <div class="sub" id="sub"></div>
  <div class="grid">
    <div>
      <div class="card">
        <h2>Trajectory &amp; scene <span style="text-transform:none;letter-spacing:0">
          — schematic layout, not geometry</span></h2>
        <div class="mapbox"><svg id="map"></svg></div>
        <div class="ctl">
          <button id="prev">◀</button><button id="play">▶ play</button>
          <button id="next">▶</button>
          <input type="range" id="slider" min="0" value="0">
          <span class="step" id="stepl"></span>
        </div>
        <div class="now" id="now"></div>
        <div class="legend">
          <span><i style="background:var(--goal)"></i>goal object</span>
          <span><i style="background:var(--acc)"></i>touched by a tool</span>
          <span><i style="background:var(--plain)"></i>other</span>
          <span>▲ agent</span><span>▦ room</span>
        </div>
      </div>
    </div>
    <div>
      <div class="card" style="margin-bottom:14px">
        <h2>Task completion</h2>
        <p class="goalnl" id="gnl"></p>
        <div class="bar"><i id="barfill"></i></div>
        <div id="progress" style="font-size:12px;color:var(--mut);margin-bottom:8px"></div>
        <h2 style="margin-top:12px">Goal conditions — bddl evaluator</h2>
        <ul class="chk" id="conds"></ul>
        <h2 style="margin-top:12px">Ground atoms — option 0</h2>
        <ul class="chk" id="atoms"></ul>
      </div>
      <div class="card">
        <h2>Primitives called</h2>
        <ul class="log" id="log"></ul>
      </div>
    </div>
  </div>
  <div class="note" id="note"></div>
</div>
<script>
const D = __DATA__;
const F = D.frames;
let i = 0, timer = null;
const $ = s => document.querySelector(s);

$("#ttl").innerHTML = D.activity.replace(/_/g," ") +
  ' <span class="badge ' + (D.success ? "b-ok" : "b-bad") + '">' +
  (D.success ? "SUCCESS" : "FAILED") + "</span>";
$("#sub").textContent = D.driver + "  ·  obs=" + D.obs_mode + "  ·  goal=" +
  D.goal_format + "  ·  " + D.n_steps + " tool calls  ·  BEHAVIOR-TextWorld";
$("#gnl").textContent = D.goal_nl || "";
$("#note").innerHTML =
  "<b>What is real and what is drawn.</b> BEHAVIOR-TextWorld state is a set of BDDL " +
  "ground literals; it has no coordinates, and <code>observe()</code> reports none. " +
  "The xy on this canvas is laid out by <code>demo_render.py</code> from room " +
  "membership and containment, so <b>distances mean nothing</b> — what is faithful is " +
  "the topology and when it changes. The tool sequence, every accept/reject, and both " +
  "checklists come from the environment unmodified: the upper list is " +
  "<code>bddl.activity.evaluate_goal_conditions</code>, the same call that scores the " +
  "reward; the lower is the option-0 ground expansion the prompt renders. A quantified " +
  "goal has several ground options and only one need hold, so the badge can read " +
  "FAILED with every atom ticked. No reasoning or memory panel: the policy has neither.";

const slider = $("#slider");
slider.max = F.length - 1;

function traj(upto){
  const p = [];
  for (let k = 0; k <= upto; k++){
    const a = F[k].agent;
    if (!p.length || p[p.length-1][0] !== a[0] || p[p.length-1][1] !== a[1]) p.push(a);
  }
  return p;
}

const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, text){
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}

function drawMap(){
  const svg = $("#map"), f = F[i];
  svg.setAttribute("viewBox", `0 0 ${D.width} ${D.height}`);
  svg.setAttribute("width", D.width); svg.setAttribute("height", D.height);
  svg.replaceChildren();
  for (const r of D.rooms){
    svg.appendChild(el("rect", {x:r.x, y:r.y, width:r.w, height:r.h, rx:10,
      fill:"var(--room)", stroke:"var(--roomln)"}));
    svg.appendChild(el("text", {x:r.x+10, y:r.y+18, "font-size":12,
      fill:"var(--mut)"}, r.name));
  }
  const p = traj(i);
  if (p.length > 1){
    svg.appendChild(el("path", {d: p.map((q,k)=>(k?"L":"M")+q[0]+","+q[1]).join(" "),
      fill:"none", stroke:"var(--acc)", "stroke-width":2, "stroke-dasharray":"5 4",
      opacity:.7}));
    p.forEach((q,k)=>{
      svg.appendChild(el("circle", {cx:q[0], cy:q[1], r:8, fill:"var(--acc)", opacity:.16}));
      svg.appendChild(el("text", {x:q[0], y:q[1]+3, "font-size":8,
        "text-anchor":"middle", fill:"var(--acc)"}, k));
    });
  }
  for (const o of f.objects){
    const fill = o.kind === "goal" ? "var(--goal)"
               : o.kind === "touched" ? "var(--acc)" : "var(--plain)";
    const r = o.anchor ? 7 : 5;
    svg.appendChild(el("circle", {cx:o.x, cy:o.y, r:r, fill:fill,
      stroke: o.held ? "var(--fg)" : "none", "stroke-width": o.held ? 2 : 0}));
    svg.appendChild(el("text", {x:o.x, y:o.y-r-3, "font-size":9,
      "text-anchor":"middle", fill:"var(--fg)"}, o.label));
  }
  const a = f.agent;
  svg.appendChild(el("polygon",
    {points:`${a[0]},${a[1]-11} ${a[0]-9},${a[1]+7} ${a[0]+9},${a[1]+7}`,
     fill:"var(--fg)"}));
}

function checklist(node, labels, flags){
  node.replaceChildren();
  labels.forEach((lab, k)=>{
    const li = document.createElement("li");
    const m = document.createElement("span");
    m.className = "m " + (flags[k] ? "yes" : "no");
    m.textContent = flags[k] ? "✓" : "○";
    const c = document.createElement("code");
    c.textContent = lab;
    li.append(m, c); node.appendChild(li);
  });
}

function buildLog(){
  const log = $("#log");
  log.replaceChildren();
  F.forEach((f, k)=>{
    const li = document.createElement("li");
    li.dataset.k = k;
    if (f.ok === false) li.classList.add("rej");
    const n = document.createElement("span"); n.className = "n"; n.textContent = k;
    const t = document.createElement("span");
    t.textContent = f.call + (f.ok === false ? "  ✗" : "");
    if (f.ok === false) t.className = "rej";
    li.append(n, t);
    li.onclick = ()=>{ i = k; render(); };
    log.appendChild(li);
  });
}

function render(){
  const f = F[i];
  drawMap();
  slider.value = i;
  $("#stepl").textContent = i + " / " + (F.length - 1);
  const okTxt = f.ok === null ? "" : (f.ok ? "  → ok" : "  → REJECTED");
  $("#now").textContent = f.call + okTxt + (f.reason ? "\n" + f.reason : "") +
    (f.held ? "\nholding: " + f.held : "") + (f.at ? "\nat: " + f.at : "");
  const pct = f.n_goal ? 100 * f.n_sat / f.n_goal : 0;
  $("#barfill").style.width = pct + "%";
  $("#progress").textContent = f.n_sat + " / " + f.n_goal +
    " goal conditions satisfied" + (f.success ? "  — task complete" : "");
  checklist($("#conds"), D.cond_labels, f.conds);
  checklist($("#atoms"), D.atom_labels, f.atoms);
  document.querySelectorAll("#log li").forEach(li=>
    li.classList.toggle("cur", +li.dataset.k === i));
  const cur = document.querySelector("#log li.cur");
  if (cur) cur.scrollIntoView({block:"nearest"});
}

function stop(){ if (timer){ clearInterval(timer); timer = null; $("#play").textContent = "▶ play"; } }
$("#prev").onclick = ()=>{ stop(); i = Math.max(0, i-1); render(); };
$("#next").onclick = ()=>{ stop(); i = Math.min(F.length-1, i+1); render(); };
slider.oninput = ()=>{ stop(); i = +slider.value; render(); };
$("#play").onclick = ()=>{
  if (timer){ stop(); return; }
  if (i >= F.length - 1) i = 0;
  $("#play").textContent = "❚❚ pause";
  timer = setInterval(()=>{ if (i >= F.length-1){ stop(); return; } i++; render(); }, 700);
};
document.addEventListener("keydown", e=>{
  if (e.key === "ArrowLeft") $("#prev").click();
  if (e.key === "ArrowRight") $("#next").click();
  if (e.key === " "){ e.preventDefault(); $("#play").click(); }
});

buildLog(); render();
</script>
"""


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity", default="picking_up_trash")
    ap.add_argument("--driver", default="expert", choices=sorted(DRIVERS))
    ap.add_argument("--model", default="", help="checkpoint path (sft) or model id (api)")
    ap.add_argument("--base-url", default="https://ai-gateway.roboparty.com/v1")
    ap.add_argument("--obs-mode", default="full",
                    choices=["full", "partial", "object"])
    ap.add_argument("--goal-format", default="atoms", choices=["atoms", "nl"])
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="", help="HTML path (default: ./<activity>_demo.html)")
    ap.add_argument("--svg", action="store_true", help="also write a static final-frame SVG")
    ap.add_argument("--json", action="store_true", help="also write the frame data")
    args = ap.parse_args()

    demo = build_demo(args.activity, args.driver, args.obs_mode, args.goal_format,
                      args.max_turns, args.temperature, args.model, args.base_url)

    out = args.out or f"{args.activity}_demo.html"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(render_html(demo))

    print(f"activity : {demo['activity']}  ({demo['driver']})")
    print(f"steps    : {demo['n_steps']}   success: {demo['success']}   "
          f"goal: {demo['frames'][-1]['n_sat']}/{demo['frames'][-1]['n_goal']}")
    print(f"wrote    : {out}")
    if args.svg:
        p = out.rsplit(".", 1)[0] + ".svg"
        with open(p, "w") as f:
            f.write(render_svg(demo))
        print(f"wrote    : {p}")
    if args.json:
        p = out.rsplit(".", 1)[0] + ".json"
        with open(p, "w") as f:
            json.dump(demo, f, ensure_ascii=False, indent=1)
        print(f"wrote    : {p}")


if __name__ == "__main__":
    main()
