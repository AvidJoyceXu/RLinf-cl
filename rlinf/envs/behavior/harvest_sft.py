"""Harvest BEHAVIOR expert trajectories into a tool-call SFT dataset (M3, step 2).

Because ``env.update_task()`` cannot switch activities (it only re-samples the
same one), each activity needs its OWN fresh IsaacSim process. So this script has
two subcommands:

  ``one``   -- boot ONE activity, run the symbolic expert, and write a per-activity
               shard (0/1 JSONL row) + a status JSON. Meant to be called once per
               activity by an external driver loop (one boot each, ~5-7 min).
  ``merge`` -- combine all shards in a dir into the final train JSONL, write the
               tool-schema sidecar, and print a per-activity manifest.

Honesty (0710 §4, and the standing no-faking rule): a row is kept ONLY if the
trajectory reached the REAL BDDL goal (``traj.success``). Activities the planner
cannot solve (unsupported predicates, placement flakiness) are recorded in the
status with their reason and excluded from the training set -- never fabricated.

Usage (inside the behavior-smoke container, venv openvla-oft):
    export OMNIGIBSON_DATA_PATH=/data/behavior-data OMNI_KIT_ACCEPT_EULA=YES
    export TMPDIR=/data/behavior-data/tmp HF_HOME=/data/hf_home
    python -m rlinf.envs.behavior.harvest_sft one \
        --activity carrying_in_groceries --out-dir /data/behavior-data/sft
    # ... repeat per activity (driver loop) ...
    python -m rlinf.envs.behavior.harvest_sft merge \
        --out-dir /data/behavior-data/sft --dataset behavior_sft_train.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import traceback

CFG_REL = "examples/embodiment/config/env/behavior_r1pro.yaml"


def _rlinf_root() -> str:
    # this file: <root>/rlinf/envs/behavior/harvest_sft.py
    return str(pathlib.Path(__file__).resolve().parents[3])


def harvest_one(activity: str, out_dir: str, obs_mode: str) -> dict:
    """Boot @activity, run the expert, write shard + status. Returns the status."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from omegaconf import OmegaConf
    from omnigibson.envs import VectorEnvironment

    from rlinf.envs.behavior import expert_planner as ep
    from rlinf.envs.behavior import sft_build as sb
    from rlinf.envs.behavior.semantic_tools import SemanticACI
    from rlinf.envs.behavior.utils import setup_omni_cfg

    os.makedirs(out_dir, exist_ok=True)
    status = {"activity": activity, "obs_mode": obs_mode, "kept": False}

    cfg = OmegaConf.load(os.path.join(_rlinf_root(), CFG_REL))
    OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.activity_name", activity, force_add=True)
    omni_cfg = setup_omni_cfg(cfg)
    env = VectorEnvironment(1, OmegaConf.to_container(omni_cfg, resolve=True))
    env.reset()
    e0 = env.envs[0]
    task = e0.task

    aci = SemanticACI(e0, obs_mode=obs_mode)
    # Grounding snapshot BEFORE the planner mutates the scene.
    initial_obs = aci.observe()
    goal_atoms = task.ground_goal_state_options[0]
    goal_lines = sb.goal_atoms_to_lines(goal_atoms)

    traj = ep.run_expert(aci, task, task_description=sb.activity_to_task(activity))
    row = sb.trajectory_to_row(traj, initial_obs, goal_lines, obs_mode=obs_mode)

    gs = aci.goal_status()
    status.update({
        "success": bool(traj.success),
        "num_steps": traj.num_steps,
        "num_turns": row["num_turns"],
        "unsupported": [list(u) if isinstance(u, (list, tuple)) else u
                        for u in traj.unsupported],
        "fail_steps": [f"{s.tool}({s.args}): {s.reason}"
                       for s in traj.steps if not s.ok],
        "goal_unsatisfied": [str(x) for x in gs["unsatisfied"]],
        "kept": bool(traj.success),
    })

    shard = os.path.join(out_dir, f"shard_{activity}.jsonl")
    with open(shard, "w") as f:
        if traj.success:                       # keep ONLY clean demonstrations
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"status_{activity}.json"), "w") as f:
        json.dump(status, f, indent=2)

    try:
        env.close()
    except Exception:
        pass
    return status


def merge(out_dir: str, dataset: str) -> dict:
    from rlinf.envs.behavior import sft_build as sb

    rows, manifest = [], []
    for st_path in sorted(glob.glob(os.path.join(out_dir, "status_*.json"))):
        st = json.load(open(st_path))
        manifest.append({k: st.get(k) for k in
                         ("activity", "success", "kept", "num_turns",
                          "unsupported", "goal_unsatisfied", "error")})
    for shard in sorted(glob.glob(os.path.join(out_dir, "shard_*.jsonl"))):
        rows += [json.loads(l) for l in open(shard) if l.strip()]

    data_path = os.path.join(out_dir, dataset)
    with open(data_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sb.dump_tool_schemas(data_path + ".tools.json")
    with open(os.path.join(out_dir, "harvest_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    kept = [m["activity"] for m in manifest if m.get("kept")]
    print(f"dataset: {data_path}  rows={len(rows)}  kept_activities={len(kept)}")
    print(f"sidecar: {data_path}.tools.json ({len(sb.tool_schemas())} tools)")
    print("kept:", ", ".join(kept))
    print("dropped:", ", ".join(m["activity"] for m in manifest if not m.get("kept")))
    return {"rows": len(rows), "kept": kept}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--activity", required=True)
    one.add_argument("--out-dir", required=True)
    one.add_argument("--obs-mode", default="full", choices=["full", "partial"])
    mrg = sub.add_parser("merge")
    mrg.add_argument("--out-dir", required=True)
    mrg.add_argument("--dataset", default="behavior_sft_train.jsonl")
    args = ap.parse_args()

    if args.cmd == "one":
        try:
            st = harvest_one(args.activity, args.out_dir, args.obs_mode)
        except Exception as ex:  # boot/planner crash -> record, keep nothing
            st = {"activity": args.activity, "kept": False,
                  "error": f"{type(ex).__name__}: {str(ex)[:400]}"}
            os.makedirs(args.out_dir, exist_ok=True)
            open(os.path.join(args.out_dir, f"shard_{args.activity}.jsonl"), "w").close()
            with open(os.path.join(args.out_dir, f"status_{args.activity}.json"), "w") as f:
                json.dump(st, f, indent=2)
            traceback.print_exc()
        print("HARVEST_ONE_DONE " + json.dumps(st))
    else:
        merge(args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
