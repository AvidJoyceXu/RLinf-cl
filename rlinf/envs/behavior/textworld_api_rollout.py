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

Everything else is held identical to `textworld_rollout.py`: the same mode-specific
tool schemas (17 core tools plus 8 viewpoint tools in camera modes), the same system
prompt, the same environment, and success read from the same real BDDL evaluator.
Only the policy differs.

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
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.episode_contract import (
    MAX_NUDGES,
    NUDGE,
    outcome_from_goal_status,
    validate_detect_runtime,
    validate_geometry_instance,
)
from rlinf.envs.behavior.rft_sample import result_payload
from rlinf.envs.behavior.rollout_store import (
    EpisodeStore,
    atomic_write_json,
    episode_key,
)
from rlinf.envs.behavior.symbolic_world import SymbolicACI, SymbolicWorld
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld
from rlinf.envs.behavior.turn_budget import TurnBudget, turn_budget_for

DEFAULT_BASE_URL = "https://ai-gateway.roboparty.com/v1"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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


def openai_tools(
    obs_mode: str = "full",
    minimal: bool = False,
    selection_mode: str = "handle",
) -> list[dict]:
    """The same mode-specific schemas, in OpenAI function-calling shape.

    Sourced from `sft_build.tool_schemas()` so the API baseline and the RL policy
    can never drift apart in what actions they are offered.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in sb.all_schemas(
            obs_mode,
            minimal=minimal,
            selection_mode=selection_mode,
        )
    ]


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
    except Exception:  # noqa: BLE001
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


def run_episode(
    client,
    model,
    activity,
    obs_mode,
    goal_format,
    max_turns,
    temperature,
    usage: Usage,
    aci=None,
    layout_prefer: str = "sampled",
    prompt_mode: str = "guided",
    reason_mode: str = "terse",
    budget_meta: dict | None = None,
    selection_mode: str = "handle",
    episode_spec: dict | None = None,
    context_tokenizer=None,
    context_max_length: int = 6144,
    context_max_new_tokens: int = 512,
    provenance: dict | None = None,
    request_seed: int | None = None,
    top_p: float = 1.0,
) -> dict:
    """One episode. Returns a record; never raises -- an API failure is a datum.

    @aci lets a caller supply its own ACI -- ``demo_render.RecordingACI`` wraps one
    to snapshot the episode. Passing it in rather than re-running elsewhere is what
    keeps the demo showing the same episode this harness would have scored.
    """
    # Every mode that reads the scene json refuses a generated layout by design --
    # poses must come from the sampler, in the scene's frame and satisfying the BDDL
    # initial conditions. The default below is `generated` for the fov study, so these
    # must override it or every episode dies in the constructor.
    if obs_mode in (
        "detect",
        "detect_scope",
        "detect_scope_scan",
        "detect_scope_memory",
        "fov_distract",
    ):
        layout_prefer = "sampled"
    episode_spec = dict(episode_spec or {})
    instance_source = episode_spec.get("instance_source")
    env = BehaviorTextWorld(
        activity,
        obs_mode=obs_mode,
        selection_mode=selection_mode,
        instance_source_selection=instance_source,
    )
    if aci is None:
        # `generated` by default for a REASON, not convenience: only 6 of the 112
        # activities in the fov subset have sampled poses, so leaving the default
        # would give those six metre distances and the rest ordinal bands --
        # two observation conditions inside one arm. Uniform beats real here.
        aci = SymbolicACI(
            SymbolicWorld(
                activity,
                instance_source_selection=instance_source,
            ),
            obs_mode=obs_mode,
            layout_prefer=layout_prefer,
            reason_mode=reason_mode,
            selection_mode=selection_mode,
            layout_source_selection=instance_source,
        )
    elif getattr(aci, "selection_mode", selection_mode) != selection_mode:
        raise ValueError(
            f"supplied ACI selection_mode={getattr(aci, 'selection_mode', None)!r} "
            f"does not match requested {selection_mode!r}"
        )
    world = aci.world
    actual_geometry_key = validate_geometry_instance(
        getattr(aci, "layout", None), episode_spec.get("geometry_instance_key")
    )

    goal_nl = env.goal_nl if goal_format == "nl" else None
    base_messages = sb.build_prompt_messages(
        activity,
        env.goal_lines,
        obs_mode,
        goal_nl=goal_nl,
        prompt_mode=prompt_mode,
        selection_mode=selection_mode,
    )
    tools = openai_tools(
        obs_mode,
        minimal=prompt_mode == "minimal",
        selection_mode=selection_mode,
    )
    native_turns: list[list[dict]] = []
    shadow_turns: list[tuple[list[int], list[int]]] = []
    prompt_ids: list[int] = []
    max_context_len = context_max_length - context_max_new_tokens
    if context_tokenizer is not None:
        prompt_text = context_tokenizer.apply_chat_template(
            base_messages,
            tools=sb.all_schemas(obs_mode, selection_mode=selection_mode),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = context_tokenizer.encode(prompt_text, add_special_tokens=False)
        from spatialcode.sft_format import compact_complete_turns

        compact_complete_turns(prompt_ids, [], max_context_len)

    def current_messages() -> tuple[list[dict], int]:
        if context_tokenizer is None:
            return base_messages + [
                message for turn in native_turns for message in turn
            ], 0
        from spatialcode.sft_format import compact_complete_turns

        _, first_kept = compact_complete_turns(
            prompt_ids, shadow_turns, max_context_len
        )
        return (
            base_messages
            + [message for turn in native_turns[first_kept:] for message in turn],
            first_kept,
        )

    steps: list[dict] = []
    ended = False
    stop_reason = "max_turns"
    api_error = ""
    nudges = 0
    turns = 0
    model_turns = 0
    context_compactions = 0
    total_nudges = 0
    episode_prompt_tokens = 0
    episode_completion_tokens = 0
    episode_api_calls = 0
    episode_api_errors = 0
    started_at = time.monotonic()

    while turns < max_turns:
        try:
            messages, first_kept = current_messages()
        except ValueError as ex:
            stop_reason = "context_overflow"
            api_error = str(ex)[:400]
            break
        context_compactions += int(first_kept > 0)
        model_turns += 1
        try:
            request_kwargs = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": context_max_new_tokens,
            }
            if request_seed is not None:
                request_kwargs["seed"] = request_seed
            resp = client.chat.completions.create(
                **request_kwargs,
            )
            response_usage = getattr(resp, "usage", None)
            usage.add(response_usage)
            episode_api_calls += 1
            if response_usage is not None:
                episode_prompt_tokens += (
                    getattr(response_usage, "prompt_tokens", 0) or 0
                )
                episode_completion_tokens += (
                    getattr(response_usage, "completion_tokens", 0) or 0
                )
        except Exception as ex:  # noqa: BLE001
            usage.add(None, errored=True)
            episode_api_calls += 1
            episode_api_errors += 1
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
            except Exception:  # noqa: BLE001
                args = {}
            call_id = call.id
            assistant_message = _assistant_msg(
                msg,
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            )
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
                total_nudges += 1
                if nudges > MAX_NUDGES:
                    stop_reason = "no_tool_call"
                    break
                assistant_message = _assistant_msg(msg)
                native_turns.append(
                    [assistant_message, {"role": "user", "content": NUDGE}]
                )
                if context_tokenizer is not None:
                    prose = (
                        assistant_message.get("reasoning_content", "")
                        + "\n"
                        + assistant_message.get("content", "")
                    ).strip()
                    shadow_turns.append(
                        (
                            context_tokenizer.encode(prose, add_special_tokens=False),
                            context_tokenizer.encode(NUDGE, add_special_tokens=False),
                        )
                    )
                continue
            nudges = 0
            name, args = parsed
            call_id = None
            assistant_message = _assistant_msg(msg)

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
            result_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": body,
            }
        else:
            result_message = {"role": "user", "content": body}
        native_turns.append([assistant_message, result_message])
        if context_tokenizer is not None:
            from spatialcode.sft_format import format_tool_call, format_tool_response

            assistant_text = (
                assistant_message.get("reasoning_content", "")
                + "\n"
                + assistant_message.get("content", "")
            ).strip()
            shadow_call = (
                format_tool_call(name, args, assistant_text or None)
                if call_id is not None
                else assistant_text
            )
            shadow_turns.append(
                (
                    context_tokenizer.encode(shadow_call, add_special_tokens=False),
                    context_tokenizer.encode(
                        format_tool_response(payload), add_special_tokens=False
                    ),
                )
            )

        steps.append({"name": name, "arguments": args, "payload": payload})
        turns += 1
        if name == "end_task":
            ended, stop_reason = True, "end_task"
            break

    outcome = outcome_from_goal_status(world.goal_status(), ended)
    record = {
        **dict(provenance or {}),
        "activity": activity,
        "obs_mode": obs_mode,
        "goal_format": goal_format,
        "selection_mode": selection_mode,
        # The budget is per-EPISODE once it is oracle-relative, so it has to travel with
        # the episode. A record that does not carry its own budget cannot be compared
        # with any other record, and `max_turns` is the variable that dominates every
        # result measured on this benchmark so far.
        "max_turns": max_turns,
        "n_steps": len(steps),
        "model_turns": model_turns,
        "tool_call_rate": len(steps) / model_turns if model_turns else 0.0,
        "nudge_count": total_nudges,
        "prompt_tokens": episode_prompt_tokens,
        "completion_tokens": episode_completion_tokens,
        "api_calls": episode_api_calls,
        "api_errors": episode_api_errors,
        "seconds": round(time.monotonic() - started_at, 3),
        "context_compactions": context_compactions,
        "instance_source": instance_source,
        "geometry_instance_key": actual_geometry_key,
        "ended": ended,
        "stop_reason": stop_reason,
        "api_error": api_error,
        "coverage": outcome["coverage"],
        "success": outcome["success"],
        "proper_completion": outcome["proper_completion"],
        "tools": [s["name"] for s in steps],
        "tool_trace": steps,
        "rejections": [
            str(s["payload"].get("reason", ""))[:70]
            for s in steps
            if s["name"] != "end_task" and s["payload"].get("ok") is False
        ],
    }
    if budget_meta:
        record.update(budget_meta)
        # The effective value passed to the loop is authoritative. Refuse metadata
        # drift rather than writing a self-contradictory benchmark record.
        if record["max_turns"] != max_turns:
            raise ValueError(
                f"budget metadata says max_turns={record['max_turns']}, "
                f"episode ran with {max_turns}"
            )
    return record


def summarise(records: list[dict], label: str) -> dict:
    n = max(len(records), 1)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in records)
    api_calls = sum(int(r.get("api_calls", 0)) for r in records)
    api_errors = sum(int(r.get("api_errors", 0)) for r in records)
    out = {
        "condition": label,
        "episodes": len(records),
        "end_task_rate": sum(r["ended"] for r in records) / n,
        "goal_coverage": sum(r["coverage"] for r in records) / n,
        "success_rate": sum(r["success"] for r in records) / n,
        "proper_completion_rate": sum(r["proper_completion"] for r in records) / n,
        "mean_steps": sum(r["n_steps"] for r in records) / n,
        "mean_model_turns": sum(r["model_turns"] for r in records) / n,
        "tool_call_rate": sum(r["n_steps"] for r in records)
        / max(sum(r["model_turns"] for r in records), 1),
        "nudges": sum(r["nudge_count"] for r in records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "api_calls": api_calls,
        "api_errors": api_errors,
        "seconds": round(sum(float(r.get("seconds", 0.0)) for r in records), 1),
    }
    print(f"\n=== {label} ===")
    print(f"  episodes      : {out['episodes']}")
    print(f"  end_task_rate : {out['end_task_rate']:.3f}")
    print(f"  goal_coverage : {out['goal_coverage']:.3f}")
    print(f"  success_rate  : {out['success_rate']:.3f}")
    print(f"  proper_comp.  : {out['proper_completion_rate']:.3f}")
    print(f"  mean_steps    : {out['mean_steps']:.1f}")
    print(
        f"  tokens        : {prompt_tokens} prompt + {completion_tokens} completion "
        f"over {api_calls} calls ({api_errors} errors)"
    )
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


def harness_failure_record(
    activity: str,
    obs_mode: str,
    goal_format: str,
    selection_mode: str,
    error: Exception,
    budget: TurnBudget,
    episode_spec: dict,
    provenance: dict | None = None,
) -> dict:
    """Keep construction/provider failures in the declared denominator."""
    record = {
        **dict(provenance or {}),
        "activity": activity,
        "obs_mode": obs_mode,
        "goal_format": goal_format,
        "selection_mode": selection_mode,
        "instance_source": episode_spec.get("instance_source"),
        "geometry_instance_key": episode_spec.get("geometry_instance_key"),
        "max_turns": budget.max_turns,
        "n_steps": 0,
        "model_turns": 0,
        "tool_call_rate": 0.0,
        "nudge_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_calls": 0,
        "api_errors": 0,
        "seconds": 0.0,
        "context_compactions": 0,
        "ended": False,
        "stop_reason": f"harness_error:{type(error).__name__}",
        "api_error": str(error)[:400],
        "coverage": 0.0,
        "success": False,
        "proper_completion": False,
        "tools": [],
        "tool_trace": [],
        "rejections": [],
    }
    record.update(budget.record())
    return record


def load_episode_specs(spec: str, limit: int) -> list[dict]:
    if not spec.startswith("@"):
        if "/" in spec or spec.endswith((".jsonl", ".txt")):
            raise ValueError(
                f"activity-list paths must be prefixed with '@' (received {spec!r})"
            )
        rows = [
            {"activity": value.strip()} for value in spec.split(",") if value.strip()
        ]
    else:
        rows = []
        with open(spec[1:]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    row = json.loads(line)
                    episode = json.loads(row["solutions"])
                    if not isinstance(episode, dict) or not episode.get("activity"):
                        raise ValueError(f"invalid episode spec in {spec}: {episode!r}")
                    rows.append(episode)
                else:
                    rows.append({"activity": line})
    return rows[:limit] if limit > 0 else rows


def load_activities(spec: str, limit: int) -> list[str]:
    """Compatibility wrapper for callers that only need activity names."""
    return [row["activity"] for row in load_episode_specs(spec, limit)]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="chat-fast")
    ap.add_argument("--replicate-id", default="0")
    ap.add_argument("--rlinf-revision", default=os.environ.get("RLINF_REVISION"))
    ap.add_argument("--outer-revision", default=os.environ.get("OUTER_REVISION"))
    ap.add_argument("--container-image", default=os.environ.get("CONTAINER_IMAGE"))
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional provider seed; omit when the endpoint does not support it",
    )
    ap.add_argument(
        "--base-url", default=os.environ.get("ROBOPARTY_BASE_URL", DEFAULT_BASE_URL)
    )
    ap.add_argument(
        "--activities",
        default="@/data/behavior-data/text_detect_v1/rl/val_s2.jsonl",
    )
    ap.add_argument("--n", type=int, default=0, help="0 runs the full frozen split")
    ap.add_argument(
        "--prompt-mode",
        default="guided",
        choices=["guided", "minimal"],
        help="minimal strips the procedural guidance from the "
        "system prompt, the tool descriptions and the "
        "per-obs-mode hint (E1, brainstorm/260815)",
    )
    ap.add_argument(
        "--reason-mode",
        default="terse",
        choices=["verbose", "terse", "silent"],
        help="refusal detail. verbose='not near X; go_to(X) first' "
        "(state+plan), terse='not near X' (state only), "
        "silent='' (ALFWorld's Nothing-happens). E1b",
    )
    ap.add_argument(
        "--obs-mode",
        default="detect_scope",
        choices=[
            "full",
            "partial",
            "object",
            # the 2x2 that splits `detect` into its two axes:
            # rows = distractors, columns = selection mechanism
            "fov",
            "fov_distract",
            "detect_scope",
            "detect_scope_scan",
            "detect_scope_memory",
            "detect",
            "both",
        ],
    )
    ap.add_argument("--goal-format", default="nl", choices=["atoms", "nl", "both"])
    ap.add_argument(
        "--selection-mode",
        default="handle",
        choices=["handle", "bbox"],
        help="detect-family object reference exposed to the policy; "
        "bbox is a text-serialization ablation, not multimodal grounding",
    )
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument(
        "--oracle-steps",
        default=None,
        help="JSON table from `viewpoint_search --out`. Switches the "
        "turn budget from a flat --max-turns to an ORACLE-RELATIVE "
        "one: max_turns = ceil(K * oracle_steps[activity]). A flat "
        "budget is "
        "unfair in both directions -- the reference search needs a "
        "median 56 steps and a p90 in the hundreds -- so a flat 40 "
        "floors the hard activities (measured: 32/32 episodes hit the "
        "ceiling and none terminated) while a flat 400 lets the easy "
        "third idle.",
    )
    ap.add_argument(
        "--budget-cap",
        type=int,
        default=0,
        help="hard ceiling on the oracle-relative budget, 0 = none. "
        "Cost is SUPERLINEAR in turns (context accumulates), and "
        "the oracle's "
        "step counts span 56..1127 here, so K=2 would ask for 2254 "
        "turns on the worst activity. Cap it and REPORT the cap: a "
        "capped episode is scored under a different rule from an "
        "uncapped one, and the record says which via `capped`.",
    )
    ap.add_argument(
        "--budget-k",
        type=float,
        default=2.0,
        help="K in max_turns = K * oracle_steps. REPORT IT: it is an "
        "axis of the result, not a constant. K also absorbs the "
        "fact that the "
        "oracle is privileged, so its step count is a LOWER bound on "
        "what an unprivileged policy needs.",
    )
    ap.add_argument(
        "--allow-missing-oracle",
        action="store_true",
        help="explicitly fall back to --max-turns for activities absent "
        "from --oracle-steps. Off by default because mixing budget "
        "rules inside one condition is not directly comparable; "
        "fallback episodes are labelled budget_source=flat_fallback.",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--context-tokenizer",
        default="/data/hf_home/qwen25-tokenizer",
        help="Qwen tokenizer used only for the shared harness context budget",
    )
    ap.add_argument("--context-max-length", type=int, default=6144)
    ap.add_argument("--context-max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel episodes; each has its own world, so they are independent",
    )
    ap.add_argument("--out", help="write per-episode records as JSONL")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="validate and resume atomically persisted per-episode records",
    )
    args = ap.parse_args()
    requested_obs_modes = (
        ["full", "partial"] if args.obs_mode == "both" else [args.obs_mode]
    )
    for requested_obs_mode in requested_obs_modes:
        validate_detect_runtime(requested_obs_mode)

    key = os.environ.get("ROBOPARTY_API_KEY")
    if not key:
        raise SystemExit(
            "ROBOPARTY_API_KEY is not set. Export it in the shell that runs this "
            "(deliberately not a flag: flags land in shell history and `ps`, and the "
            "gateway guide forbids the key reaching a repo)."
        )

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=args.base_url, timeout=120.0, max_retries=3)
    episode_specs = load_episode_specs(args.activities, args.n)
    acts = [spec["activity"] for spec in episode_specs]
    if len(set(acts)) != len(acts):
        raise SystemExit("activity input contains duplicates")
    obs_modes = ["full", "partial"] if args.obs_mode == "both" else [args.obs_mode]
    goal_formats = ["atoms", "nl"] if args.goal_format == "both" else [args.goal_format]

    oracle_steps = {}
    if args.oracle_steps:
        with open(args.oracle_steps) as f:
            oracle_steps = json.load(f).get("oracle_steps", {})
        if not oracle_steps:
            raise SystemExit(
                f"{args.oracle_steps} carries no oracle_steps; run "
                "`viewpoint_search --all --out <file>` first"
            )
    elif all(spec.get("oracle_turns") is not None for spec in episode_specs):
        oracle_steps = {
            spec["activity"]: int(spec["oracle_turns"]) for spec in episode_specs
        }

    from transformers import AutoTokenizer

    context_tokenizer = AutoTokenizer.from_pretrained(args.context_tokenizer)

    activities_path = (
        os.path.abspath(args.activities[1:])
        if args.activities.startswith("@")
        else None
    )
    run_provenance = {
        "adapter": "openai_chat_completions",
        "model_id": args.model,
        "endpoint": args.base_url,
        "activities_path": activities_path,
        "activities_sha256": _sha256(activities_path) if activities_path else None,
        "replicate_id": str(args.replicate_id),
        "rlinf_revision": args.rlinf_revision or None,
        "outer_revision": args.outer_revision or None,
        "container_image": args.container_image or None,
        "harness_file_sha256": _sha256(__file__),
        "prompt_mode": args.prompt_mode,
        "reason_mode": args.reason_mode,
        "request_seed": args.seed,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "context_max_length": args.context_max_length,
            "max_new_tokens": args.context_max_new_tokens,
        },
    }

    try:
        budgets: dict[str, TurnBudget] = {
            activity: turn_budget_for(
                activity,
                flat_max_turns=args.max_turns,
                oracle_steps=oracle_steps,
                budget_k=args.budget_k,
                budget_cap=args.budget_cap,
                allow_missing_oracle=args.allow_missing_oracle,
            )
            for activity in acts
        }
    except (KeyError, ValueError) as ex:
        raise SystemExit(f"invalid turn budget: {ex}") from ex

    declared_episodes = [
        (obs_mode, goal_format, spec)
        for obs_mode in obs_modes
        for goal_format in goal_formats
        for spec in episode_specs
    ]
    expected_keys = [
        episode_key(
            activity=spec["activity"],
            instance_source=spec.get("instance_source"),
            geometry_instance_key=spec.get("geometry_instance_key"),
            instance_id=spec.get("instance_id", 0),
            obs_mode=obs_mode,
            goal_format=goal_format,
        )
        for obs_mode, goal_format, spec in declared_episodes
    ]
    if args.resume and not args.out:
        raise SystemExit("--resume requires --out")
    store = None
    if args.out:
        store = EpisodeStore(
            args.out,
            contract={
                "adapter": "openai_chat_completions",
                "provenance": run_provenance,
                "episode_specs": episode_specs,
                "obs_modes": obs_modes,
                "goal_formats": goal_formats,
                "budgets": {
                    activity: budget.record() for activity, budget in budgets.items()
                },
                "context_tokenizer": args.context_tokenizer,
            },
            expected_keys=expected_keys,
            resume=args.resume,
        )
        if store.completed_count:
            print(
                f"resuming {store.completed_count}/{len(declared_episodes)} "
                f"persisted episodes from {store.partial_dir}",
                flush=True,
            )

    print(f"model       : {args.model}  @ {args.base_url}")
    print(f"activities  : {len(acts)} (held-out S2)")
    if oracle_steps:
        _b = [budgets[a].max_turns for a in acts]
        _capped = sum(b.capped for b in budgets.values())
        _fallback = sum(b.budget_source == "flat_fallback" for b in budgets.values())
        print(
            f"budget      : ORACLE-RELATIVE K={args.budget_k}"
            f"{f' cap {args.budget_cap}' if args.budget_cap else ''} -> "
            f"median {sorted(_b)[len(_b) // 2]} turns, min {min(_b)}, max {max(_b)}"
            f"{f'  ({_capped} episodes hit the cap)' if _capped else ''}"
            f"{f'  ({_fallback} flat fallbacks)' if _fallback else ''}"
        )
    else:
        print(f"budget      : FLAT {args.max_turns} turns")
    print(f"conditions  : obs={obs_modes} x goal={goal_formats}")
    print(f"selection   : {args.selection_mode}")
    print(f"prompt      : {args.prompt_mode}  refusals: {args.reason_mode}")
    max_calls = (
        sum(b.max_turns for b in budgets.values()) * len(obs_modes) * len(goal_formats)
    )
    print(f"API ceiling : <= {max_calls} calls at temperature {args.temperature}")

    all_records, summaries = [], []
    for obs_mode in obs_modes:
        for goal_format in goal_formats:
            label = f"obs={obs_mode} goal={goal_format}"
            usage = Usage()
            condition_entries = [
                (index, spec, expected_keys[index])
                for index, (declared_obs, declared_goal, spec) in enumerate(
                    declared_episodes
                )
                if declared_obs == obs_mode and declared_goal == goal_format
            ]
            records_by_index = {
                index: store.record(index)
                for index, _, _ in condition_entries
                if store is not None and store.is_complete(index)
            }
            pending_entries = [
                entry for entry in condition_entries if entry[0] not in records_by_index
            ]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        run_episode,
                        client,
                        args.model,
                        spec["activity"],
                        obs_mode,
                        goal_format,
                        budgets[spec["activity"]].max_turns,
                        args.temperature,
                        usage,
                        None,
                        "sampled",
                        args.prompt_mode,
                        args.reason_mode,
                        budgets[spec["activity"]].record(),
                        args.selection_mode,
                        spec,
                        context_tokenizer,
                        args.context_max_length,
                        args.context_max_new_tokens,
                        run_provenance,
                        args.seed,
                        args.top_p,
                    ): (index, spec, key)
                    for index, spec, key in pending_entries
                }
                for fut in as_completed(futures):
                    index, spec, key = futures[fut]
                    activity = spec["activity"]
                    try:
                        record = fut.result()
                    except Exception as ex:  # noqa: BLE001
                        print(f"    episode failed: {type(ex).__name__}: {ex}")
                        record = harness_failure_record(
                            activity,
                            obs_mode,
                            goal_format,
                            args.selection_mode,
                            ex,
                            budgets[activity],
                            spec,
                            run_provenance,
                        )
                    if store is not None:
                        store.commit(index, key, record)
                    records_by_index[index] = record
                    completed = (
                        store.completed_count
                        if store is not None
                        else len(all_records) + len(records_by_index)
                    )
                    print(
                        f"    [{label}] {completed}/{len(declared_episodes)} "
                        f"{activity} persisted={store is not None}",
                        flush=True,
                    )
            records = [records_by_index[index] for index, _, _ in condition_entries]
            summaries.append(summarise(records, label))
            all_records.extend(records)

    print(
        "\n--- zero-shot summary "
        "(same env, same mode-specific tools and evaluator as the RL policy) ---"
    )
    print(
        f"{'condition':28s} {'proper':>8s} {'success':>8s} {'coverage':>9s} "
        f"{'end_task':>9s} "
        f"{'steps':>6s}"
    )
    for s in summaries:
        print(
            f"{s['condition']:28s} {s['proper_completion_rate']:8.3f} "
            f"{s['success_rate']:8.3f} "
            f"{s['goal_coverage']:9.3f} {s['end_task_rate']:9.3f} "
            f"{s['mean_steps']:6.1f}"
        )

    if args.out and store is not None:
        all_records = store.records(require_complete=True)
        atomic_write_json(
            args.out.replace(".jsonl", "_summary.json"),
            {
                **run_provenance,
                "model": args.model,
                "n_activities": len(acts),
                "budget": {
                    "oracle_steps_file": args.oracle_steps,
                    "flat_max_turns": args.max_turns,
                    "budget_k": args.budget_k,
                    "budget_cap": args.budget_cap or None,
                    "allow_missing_oracle": args.allow_missing_oracle,
                },
                "selection_mode": args.selection_mode,
                "context": {
                    "tokenizer": args.context_tokenizer,
                    "max_length": args.context_max_length,
                    "max_new_tokens": args.context_max_new_tokens,
                    "compaction": "canonical prompt + newest complete turns",
                },
                "summaries": summaries,
            },
            indent=1,
        )
        store.finalize()
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
