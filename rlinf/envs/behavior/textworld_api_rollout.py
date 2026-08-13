"""Zero-shot API baselines on BEHAVIOR-TextWorld.

The research claim is that RL on real environment feedback beats *prompting a
frontier model*. That comparison is only worth anything if the prompted baseline is
the strongest reasonable one, so this deliberately does NOT reuse the SFT policy's
text protocol:

  * **native tool calling** (`tools=[...]`, `tool_choice="auto"`), not the
    `<tool_call>{...}</tool_call>` string format the fine-tuned Qwen emits. Frontier
    models are trained for the former and handicapping them would make our own
    numbers look better for the wrong reason. A text-format fallback parser is kept
    for models that answer that way anyway.
  * **multi-turn tool results fed back as `role="tool"` messages**, the shape these
    APIs expect.

Everything else is held identical to `textworld_rollout.py`: the same 17 tool
schemas, the same system prompt, the same environment, and success read from the
same real BDDL evaluator. Only the policy differs.

TWO INDEPENDENT AXES, and they are easy to conflate:

  * ``--obs-mode full|partial``  -- OBSERVABILITY. How much of the world `observe()`
    reports: every task-relevant object, or only the robot's current room.
  * ``--goal-format atoms|nl``   -- GOAL RENDERING. Ground BDDL atoms
    (`inside(can__of__soda.n.01_1, ashcan.n.01_1)`) or the English sentence rendered
    from the unground BDDL ("Every can of soda is inside the trash can.").

They are orthogonal; the 2x2 is the point. `atoms` lets a model transliterate
predicate names into same-named tools, which is exactly what claim #1 predicts and
what the SFT checkpoint's 82.5% on `atoms`+`full` demonstrated.

CREDENTIALS. The key is read from ``$ROBOPARTY_API_KEY`` and nowhere else -- not a
flag (shell history, and visible in `ps`), not a repo file. The gateway guide's rules
are explicit that the key must not reach a git repository.

    export ROBOPARTY_API_KEY=...        # not stored in this tree
    python -m rlinf.envs.behavior.textworld_api_rollout \\
        --model chat-fast --n 40 --obs-mode full --goal-format nl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.rft_sample import result_payload
from rlinf.envs.behavior.symbolic_world import SymbolicACI, SymbolicWorld
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld

DEFAULT_BASE_URL = "https://ai-gateway.roboparty.com/v1"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Consecutive prose replies tolerated before an episode is abandoned. Nudging is
# standard agent-loop practice and it is what keeps this baseline honest -- see the
# note in `run_episode`. It is capped rather than unbounded so a model that is
# genuinely stuck (refusing, or looping on an apology) still terminates.
MAX_NUDGES = 3
NUDGE = ("Continue working on the task. Respond with a tool call, not prose. "
         "If the goal is already achieved, call end_task.")


def _assistant_msg(msg, **extra) -> dict:
    """Echo an assistant turn back into the transcript, keeping `reasoning_content`.

    DeepSeek's thinking mode REQUIRES it: drop it and the *next* request fails with

        400 ... "The `reasoning_content` in the thinking mode must be passed back
        to the API."

    Measured 2026-08-13: this killed 21 of 164 episodes, and it did so
    asymmetrically -- 10-11 per `object` arm against 1 per `full` arm -- because the
    error only fires on turns where the model actually reasoned, which correlates
    with episode difficulty. An API failure that tracks difficulty is not a datum,
    it is a bias against whichever condition is harder. Baselines we intend to beat
    must not be handicapped by our own loop.
    """
    out = {"role": "assistant", "content": msg.content or "", **extra}
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None and getattr(msg, "model_extra", None):
        reasoning = msg.model_extra.get("reasoning_content")
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


def openai_tools() -> list[dict]:
    """The same 17 schemas, in OpenAI function-calling shape.

    Sourced from `sft_build.tool_schemas()` so the API baseline and the RL policy
    can never drift apart in what actions they are offered.
    """
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in sb.tool_schemas()]


def _parse_text_tool_call(content: str):
    """Fallback for models that emit the Qwen `<tool_call>` string despite `tools=`."""
    if not content:
        return None
    m = _TOOL_CALL_RE.search(content)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
        return call["name"], call.get("arguments", {}) or {}
    except Exception:                                              # noqa: BLE001
        return None


class Usage:
    """Thread-safe token/cost accounting. An API baseline that does not report what
    it cost is not reproducible."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = self.completion = self.calls = self.errors = 0

    def add(self, usage, errored: bool = False) -> None:
        with self.lock:
            self.calls += 1
            self.errors += int(errored)
            if usage is not None:
                self.prompt += getattr(usage, "prompt_tokens", 0) or 0
                self.completion += getattr(usage, "completion_tokens", 0) or 0


def run_episode(client, model, activity, obs_mode, goal_format, max_turns,
                temperature, usage: Usage, aci=None) -> dict:
    """One episode. Returns a record; never raises -- an API failure is a datum.

    @aci lets a caller supply its own ACI -- ``demo_render.RecordingACI`` wraps one
    to snapshot the episode. Passing it in rather than re-running elsewhere is what
    keeps the demo showing the same episode this harness would have scored.
    """
    env = BehaviorTextWorld(activity, obs_mode=obs_mode)
    if aci is None:
        aci = SymbolicACI(SymbolicWorld(activity), obs_mode=obs_mode)
    world = aci.world

    goal_nl = env.goal_nl if goal_format == "nl" else None
    messages = sb.build_prompt_messages(activity, env.goal_lines, obs_mode,
                                        goal_nl=goal_nl)
    tools = openai_tools()

    steps: list[dict] = []
    ended = False
    stop_reason = "max_turns"
    api_error = ""
    nudges = 0
    turns = 0

    while turns < max_turns:
        turns += 1
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                tool_choice="auto", temperature=temperature, max_tokens=512)
            usage.add(getattr(resp, "usage", None))
        except Exception as ex:                                    # noqa: BLE001
            usage.add(None, errored=True)
            # Keep the message, not just the class. A run where 25% of one arm died
            # as `api_error:BadRequestError` was undiagnosable without it, and an
            # asymmetric error rate silently biases the arm it hits.
            stop_reason = f"api_error:{type(ex).__name__}"
            api_error = str(ex)[:400]
            break

        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []

        if calls:
            nudges = 0
            call = calls[0]
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except Exception:                                      # noqa: BLE001
                args = {}
            call_id = call.id
            messages.append(_assistant_msg(
                msg, tool_calls=[{"id": call_id, "type": "function",
                                  "function": {"name": name,
                                               "arguments": call.function.arguments}}]))
        else:
            parsed = _parse_text_tool_call(msg.content or "")
            if parsed is None:
                # NUDGE, do not kill the episode. Measured on the first 2x2 run:
                # 90 of 99 early stops happened after exactly ONE step -- the model
                # called `observe`, then narrated its plan in prose instead of
                # calling the next tool. Ending there scored a planning failure for
                # what is really a turn-taking convention, and it depressed the
                # frontier baseline by a factor of ~2. A baseline we are trying to
                # BEAT must not be handicapped by our own loop.
                nudges += 1
                if nudges > MAX_NUDGES:
                    stop_reason = "no_tool_call"
                    break
                messages.append(_assistant_msg(msg))
                messages.append({"role": "user", "content": NUDGE})
                continue
            nudges = 0
            name, args = parsed
            call_id = None
            messages.append(_assistant_msg(msg))

        fn = getattr(aci, name, None)
        if fn is None or name.startswith("_"):
            payload = {"ok": False, "reason": f"unknown_tool:{name}"}
        else:
            try:
                payload = result_payload(name, fn(**(args or {})))
            except TypeError as ex:
                payload = {"ok": False, "reason": f"bad_arguments: {ex}"}

        body = json.dumps(payload, ensure_ascii=False)
        if call_id is not None:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": body})
        else:
            messages.append({"role": "user", "content": body})

        steps.append({"name": name, "arguments": args, "payload": payload})
        if name == "end_task":
            ended, stop_reason = True, "end_task"
            break

    gs = world.goal_status()
    n_sat = len(gs["satisfied"])
    n_goal = n_sat + len(gs["unsatisfied"])
    return {
        "activity": activity, "obs_mode": obs_mode, "goal_format": goal_format,
        "n_steps": len(steps), "ended": ended, "stop_reason": stop_reason,
        "api_error": api_error,
        "coverage": (n_sat / n_goal) if n_goal else 0.0,
        "success": bool(n_goal and not gs["unsatisfied"] and n_sat > 0),
        "tools": [s["name"] for s in steps],
        "rejections": [str(s["payload"].get("reason", ""))[:70]
                       for s in steps
                       if s["name"] != "end_task" and s["payload"].get("ok") is False],
    }


def summarise(records: list[dict], label: str, usage: Usage, seconds: float) -> dict:
    n = max(len(records), 1)
    out = {
        "condition": label,
        "episodes": len(records),
        "end_task_rate": sum(r["ended"] for r in records) / n,
        "goal_coverage": sum(r["coverage"] for r in records) / n,
        "success_rate": sum(r["success"] for r in records) / n,
        "mean_steps": sum(r["n_steps"] for r in records) / n,
        "prompt_tokens": usage.prompt,
        "completion_tokens": usage.completion,
        "api_calls": usage.calls,
        "api_errors": usage.errors,
        "seconds": round(seconds, 1),
    }
    print(f"\n=== {label} ===")
    print(f"  episodes      : {out['episodes']}")
    print(f"  end_task_rate : {out['end_task_rate']:.3f}")
    print(f"  goal_coverage : {out['goal_coverage']:.3f}")
    print(f"  success_rate  : {out['success_rate']:.3f}")
    print(f"  mean_steps    : {out['mean_steps']:.1f}")
    print(f"  tokens        : {usage.prompt} prompt + {usage.completion} completion "
          f"over {usage.calls} calls ({usage.errors} errors)")
    stops = collections.Counter(r["stop_reason"] for r in records)
    print(f"  stop_reasons  : {dict(stops.most_common())}")
    rej: collections.Counter = collections.Counter()
    for r in records:
        rej.update(r["rejections"])
    if rej:
        print("  top rejections:")
        for reason, count in rej.most_common(6):
            print(f"    {count:4d}  {reason}")
    return out


def load_activities(spec: str, limit: int) -> list[str]:
    if not spec.startswith("@"):
        acts = [a.strip() for a in spec.split(",") if a.strip()]
    else:
        acts = []
        with open(spec[1:]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                acts.append(json.loads(json.loads(line)["solutions"])["activity"]
                            if line.startswith("{") else line)
    return acts[:limit] if limit > 0 else acts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="chat-fast")
    ap.add_argument("--base-url", default=os.environ.get("ROBOPARTY_BASE_URL",
                                                         DEFAULT_BASE_URL))
    ap.add_argument("--activities", default="@/data/behavior-data/tw/val_s2.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--obs-mode", default="both",
                    choices=["full", "partial", "object", "both"])
    ap.add_argument("--goal-format", default="both", choices=["atoms", "nl", "both"])
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel episodes; each has its own world, so they are "
                         "independent")
    ap.add_argument("--out", help="write per-episode records as JSONL")
    args = ap.parse_args()

    key = os.environ.get("ROBOPARTY_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOPARTY_API_KEY is not set. Export it in the shell that runs this "
            "(deliberately not a flag: flags land in shell history and `ps`, and the "
            "gateway guide forbids the key reaching a repo)."
        )

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=args.base_url, timeout=120.0, max_retries=3)
    acts = load_activities(args.activities, args.n)
    obs_modes = ["full", "partial"] if args.obs_mode == "both" else [args.obs_mode]
    goal_formats = (["atoms", "nl"] if args.goal_format == "both"
                    else [args.goal_format])

    print(f"model       : {args.model}  @ {args.base_url}")
    print(f"activities  : {len(acts)} (held-out S2)")
    print(f"conditions  : obs={obs_modes} x goal={goal_formats}")
    print(f"budget      : <= {len(acts) * len(obs_modes) * len(goal_formats) * args.max_turns} "
          f"API calls at temperature {args.temperature}")

    all_records, summaries = [], []
    for obs_mode in obs_modes:
        for goal_format in goal_formats:
            label = f"obs={obs_mode} goal={goal_format}"
            usage = Usage()
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run_episode, client, args.model, a, obs_mode,
                                       goal_format, args.max_turns, args.temperature,
                                       usage)
                           for a in acts]
                records = []
                for i, fut in enumerate(futures, 1):
                    try:
                        records.append(fut.result())
                    except Exception as ex:                        # noqa: BLE001
                        print(f"    episode failed: {type(ex).__name__}: {ex}")
                    if i % 10 == 0:
                        print(f"    [{label}] {i}/{len(acts)}", flush=True)
            summaries.append(summarise(records, label, usage, time.time() - t0))
            all_records.extend(records)

    print("\n--- zero-shot summary "
          "(same env, same 17 tools, same BDDL evaluator as the RL policy) ---")
    print(f"{'condition':28s} {'success':>8s} {'coverage':>9s} {'end_task':>9s} "
          f"{'steps':>6s}")
    for s in summaries:
        print(f"{s['condition']:28s} {s['success_rate']:8.3f} "
              f"{s['goal_coverage']:9.3f} {s['end_task_rate']:9.3f} "
              f"{s['mean_steps']:6.1f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
            json.dump({"model": args.model, "n_activities": len(acts),
                       "summaries": summaries}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
