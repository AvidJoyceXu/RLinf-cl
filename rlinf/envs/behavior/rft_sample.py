"""Iterative RFT (ReST-style) sampler for the BEHAVIOR semantic tool-call agent.

Phase A of RFT. Loads the current policy (a HuggingFace Qwen2.5 dir) and, for
each training activity, runs G rollouts through ``SemanticACI`` with temperature
sampling, scoring each trajectory by the REAL BDDL ``is_success`` (+ optional
staged per-atom coverage). Emits scored-trajectory rows in the SAME schema the
SFT trainer consumes (``EqaToolCallSftDataset``: ``prompt_messages`` +
``tool_steps`` + a ``reward``/``advantage`` sidecar), so Phase B needs no env:

* RAFT / ReST^EM  : filter to ``reward_terminal == 1`` and re-run the existing
                    SFT trainer on the survivors (zero trainer change).
* advantage-weight: normalize ``reward_staged`` within each ``group_id`` and
                    weight the SFT loss (a small trainer extension).

Runs in the ``behavior-smoke`` container (``openvla-oft`` venv): needs
omnigibson + torch + transformers, NOT sglang/megatron. See memory
``behavior-grpo-venv-split``. This deliberately mirrors the proven single-episode
harness (``rollout_sft.py``) so the RL rollout distribution matches SFT exactly
(flat continuation: system+tools rendered once, then per turn the model's
``<tool_call>`` followed by the raw tool-response JSON).

ONE ACTIVITY PER PROCESS: OmniGibson locks its global macros (e.g. ``HEADLESS``)
at the first env boot, so a second ``boot_env`` in the same process fails with
``Cannot set attribute HEADLESS in MacroDict`` (observed on the iter0 smoke). A
driver must launch a fresh process per activity; passing several activities here
only records boot errors for all but the first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# The flat-continuation format helpers live at the top level of spatialcode.
sys.path.insert(0, "/workspace")

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def load_model(policy_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(policy_dir)
    model = AutoModelForCausalLM.from_pretrained(policy_dir, torch_dtype=torch.bfloat16)
    model.eval().cuda()
    return model, tok


def result_payload(tool: str, res):
    """Mirror the SFT tool-response contract (sft_build / rollout_sft):
    observe -> the full obs dict; end_task -> {ok,success}; others -> {ok,reason}.
    observe() returns a plain dict; every other tool returns a ToolResult."""
    if tool == "observe":
        return res                                   # res IS the obs dict
    if tool == "end_task":
        return {"ok": bool(res.ok), "success": bool(res.ok)}
    return {"ok": bool(res.ok), "reason": getattr(res, "reason", "")}


# --------------------------------------------------------------------------- #
# One rollout (temperature sampling)
# --------------------------------------------------------------------------- #
def rollout_once(model, tok, aci, activity, goal_lines, obs_mode,
                 max_turns, temperature, top_p, max_new_tokens=256):
    """Run one flat-continuation episode; return (tool_steps, tool_trace).

    tool_steps -> SFT-row shape [{name, arguments, result_text}]; the FIRST is an
    observe seeded from the reset obs (matches trajectory_to_row's observe-first).
    tool_trace -> [{name, arguments, meta}] for the reward module (meta carries
    the BDDL goal_status snapshot after each tool)."""
    import torch
    from rlinf.envs.behavior import sft_build as sb

    messages = sb.build_prompt_messages(activity, goal_lines, obs_mode)
    prompt = tok.apply_chat_template(messages, tools=sb.tool_schemas(),
                                     tokenize=False, add_generation_prompt=True)
    ctx = prompt
    tool_steps, tool_trace = [], []

    def _meta():
        gs = aci.goal_status()
        n_sat, n_goal = len(gs["satisfied"]), len(gs["satisfied"]) + len(gs["unsatisfied"])
        return {"num_satisfied": n_sat, "num_goal": n_goal,
                "is_success": (len(gs["unsatisfied"]) == 0 and n_sat > 0)}

    for _ in range(max_turns):
        ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=top_p, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = TC.search(gen)
        if not m:
            break                                    # unparseable -> episode ends
        model_out = gen[:m.end()]
        try:
            call = json.loads(m.group(1))
            name, args = call["name"], call.get("arguments", {}) or {}
        except Exception:
            break
        fn = getattr(aci, name, None)
        if fn is None:
            break
        try:
            res = fn(**args)
        except Exception:
            break
        payload = result_payload(name, res)
        tool_steps.append({"name": name, "arguments": args,
                           "result_text": json.dumps(payload, ensure_ascii=False)})
        tool_trace.append({"name": name, "arguments": args, "meta": _meta()})
        ctx = ctx + model_out + "\n" + json.dumps(payload, ensure_ascii=False) + "\n"
        if name == "end_task":
            break
    return tool_steps, tool_trace


# --------------------------------------------------------------------------- #
# Env boot + per-activity sampling
# --------------------------------------------------------------------------- #
def boot_env(activity, scene):
    from omegaconf import OmegaConf
    from omnigibson.envs import VectorEnvironment
    from rlinf.envs.behavior.semantic_tools import SemanticACI
    from rlinf.envs.behavior.utils import setup_omni_cfg

    cfg = OmegaConf.load("/workspace/RLinf/examples/embodiment/config/env/behavior_r1pro.yaml")
    OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.activity_name", activity, force_add=True)
    OmegaConf.update(cfg, "omni_config.scene.scene_model", scene, force_add=True)
    env = VectorEnvironment(1, OmegaConf.to_container(setup_omni_cfg(cfg), resolve=True))
    env.reset()
    e0 = env.envs[0]
    aci = SemanticACI(e0, obs_mode="full")
    return env, e0, aci


def prep_plan(task, activity, inst_dir):
    """Compute the (instance-invariant) expert plan + discover instance files.

    The pose cache is NOT built here — it is instance-specific and must be built
    on the same geometry it is executed on (see build_cache_on_instance)."""
    from rlinf.envs.behavior import expert_planner as ep
    from rlinf.envs.behavior.instance_loader import discover_activity_instance_files

    plan = ep.plan(task)
    files = discover_activity_instance_files(inst_dir, activity, 0, "tro_state")
    files = sorted(files, key=lambda f: f.instance_id)
    return plan, files


def reset_to_instance(aci, e0, inst, reset_scene: bool = True):
    """Load an instance's task-relevant object state and clear held-object bookkeeping.

    ``reset_scene=False`` skips the trailing ``env.scene.reset()``. That call is the
    dominant cost of an episode reset -- it restores the *whole* scene registry, and
    OmniGibson invalidates and rebuilds the pose cache once per kinematic prim, so it
    is quadratic (measured: one reset ran past 90 min on house_double_floor_lower;
    see code-doc/0729 - unified RL container/DEBUG-LOG.md §5). The loader has already
    written the task-relevant poses, stepped physics 25x and called
    ``update_initial_file()``, so on a *fresh* load the scene.reset() largely
    re-applies state that is already in place.

    It is NOT skippable after ``build_pose_cache`` has seated objects into
    containers -- there the reset is what restores a clean start. Callers must check
    the start state (``goal_status`` / ``observe``) when they pass False."""
    from rlinf.envs.behavior.instance_loader import load_activity_instance_tro_state
    load_activity_instance_tro_state(e0, inst.instance_id, inst.path,
                                     reset_scene=reset_scene)
    aci._held = None


def build_cache_on_instance(aci, e0, plan, inst):
    """Build the place pose cache on the CURRENT instance, then reset back to it
    so the episode starts clean with poses matching THIS instance's geometry.

    Per-instance is required: `build_pose_cache` records absolute placement poses,
    which only land inside the target on the geometry they were sampled on. A
    once-per-boot cache (the earlier bug) missed moved containers on every
    instance != the boot instance -> place_inside failed with Inside=False and the
    object stayed held, stalling the episode (observed on picking_up_trash 0/8).
    Mirrors the harvester's per-instance run_expert."""
    from omnigibson.object_states import Open

    if not plan.pairs:
        return
    for tgt in plan.place_targets:
        o = aci._resolve(tgt)
        if o is not None and Open in getattr(o, "states", {}):
            aci.go_to(tgt); aci.open(tgt)
    aci.build_pose_cache(plan.pairs)      # seats objects into containers
    reset_to_instance(aci, e0, inst)      # restore clean start; cache persists on aci


def sample_activity(model, tok, activity, group_size, instances_per_activity,
                    temperature, top_p, max_turns, obs_mode, seed):
    from rlinf.envs.behavior import sft_build as sb
    from rlinf.envs.behavior.harvest_sft import resolve_scene_and_dir
    from rlinf.algorithms.rewards import behavior as bhr

    scene, inst_dir = resolve_scene_and_dir(activity)
    env, e0, aci = boot_env(activity, scene)
    task = e0.task
    goal_lines = sb.goal_atoms_to_lines(task.ground_goal_state_options[0])
    plan, files = prep_plan(task, activity, inst_dir)
    if not files:
        env.close()
        return [], {"activity": activity, "error": "no_instances"}

    rows, n_success, n_contam = [], 0, 0
    n_inst = min(instances_per_activity, len(files))
    for g in range(group_size):
        inst = files[g % n_inst]
        reset_to_instance(aci, e0, inst)
        build_cache_on_instance(aci, e0, plan, inst)   # per-instance pose cache
        if aci.is_success():                        # contamination guard
            n_contam += 1
            continue
        tool_steps, tool_trace = rollout_once(
            model, tok, aci, activity, goal_lines, obs_mode,
            max_turns, temperature, top_p)
        terminal = bhr.compute_score(tool_trace)
        staged = bhr.compute_staged_rewards(tool_trace)
        success = bhr.episode_is_success(tool_trace)
        n_success += int(success)
        rows.append({
            "activity": activity,
            "task": sb.activity_to_task(activity),
            "success": bool(success),
            "num_turns": len(tool_steps),
            "prompt_messages": sb.build_prompt_messages(activity, goal_lines, obs_mode),
            "tool_steps": tool_steps,
            "answer": "",
            # RFT sidecar (Phase B reads these; the SFT dataset ignores them):
            "group_id": f"{activity}#inst{inst.instance_id}",
            "instance_id": inst.instance_id,
            "reward_terminal": float(terminal),
            "reward_staged": float(sum(staged)),
            "properly_ended": bhr.episode_properly_ended(tool_trace),
        })
    env.close()
    stat = {"activity": activity, "scene": scene, "n_rollouts": len(rows),
            "n_success": n_success, "n_contaminated": n_contam,
            "success_rate": round(n_success / max(len(rows), 1), 3)}
    return rows, stat


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="HF policy dir (e.g. SFT export)")
    ap.add_argument("--activities", required=True,
                    help="comma-separated activity names, or @path to a newline file")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--instances-per-activity", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--obs-mode", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output scored-trajectory jsonl")
    args = ap.parse_args()

    if args.activities.startswith("@"):
        with open(args.activities[1:]) as f:
            activities = [ln.strip() for ln in f if ln.strip()]
    else:
        activities = [a.strip() for a in args.activities.split(",") if a.strip()]

    import torch
    torch.manual_seed(args.seed)
    model, tok = load_model(args.policy)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stat_path = args.out.replace(".jsonl", "") + ".stats.json"
    all_stats, n_rows = [], 0
    t0 = time.time()
    with open(args.out, "w") as fout:
        for i, activity in enumerate(activities):
            try:
                rows, stat = sample_activity(
                    model, tok, activity, args.group_size,
                    args.instances_per_activity, args.temperature, args.top_p,
                    args.max_turns, args.obs_mode, args.seed)
            except Exception as ex:
                import traceback; traceback.print_exc()
                stat = {"activity": activity, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
                rows = []
            for r in rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
            n_rows += len(rows)
            all_stats.append(stat)
            print(f"[{i+1}/{len(activities)}] {activity}: {json.dumps(stat)}", flush=True)
    with open(stat_path, "w") as f:
        json.dump({"policy": args.policy, "n_activities": len(activities),
                   "n_rows": n_rows, "elapsed_s": round(time.time() - t0, 1),
                   "per_activity": all_stats}, f, indent=2)
    print("RFT_SAMPLE_DONE " + json.dumps({"out": args.out, "n_rows": n_rows,
          "stats": stat_path}), flush=True)


if __name__ == "__main__":
    main()
