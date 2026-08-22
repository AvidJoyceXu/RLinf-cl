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
import random
import traceback

CFG_REL = "examples/embodiment/config/env/behavior_r1pro.yaml"
# Scenes that ship pre-sampled task instances (they partition the RLinf-50 with
# no overlap: 22 / 23 / 5). See 0714 scenes+multi-instance design doc.
INSTANCE_SCENES = (
    "house_double_floor_lower", "house_single_floor", "house_double_floor_upper",
)


def _rlinf_root() -> str:
    # this file: <root>/rlinf/envs/behavior/harvest_sft.py
    return str(pathlib.Path(__file__).resolve().parents[3])


def resolve_scene_and_dir(
    activity: str,
    scene: str | None = None,
    source: str | None = None,
) -> tuple[str, str]:
    """Return ``(scene, instance_dir)`` from one explicit versioned source.

    ``BEHAVIOR_INSTANCE_SOURCE`` selects the source when ``source`` is omitted. The
    default remains ``2025-official`` for the installed v3.7 simulator, but the
    resolver no longer searches 2025, 2026 and local samples as one anonymous pool.
    """
    from rlinf.envs.behavior.instance_sources import resolve_activity_location

    data_root = os.environ.get("OMNIGIBSON_DATA_PATH", "/data/behavior-data")
    location = resolve_activity_location(
        activity,
        source_name=source,
        scene=scene,
        data_root=data_root,
    )
    return location.scene, location.directory


def harvest_one(activity: str, out_dir: str, obs_mode: str, scene: str | None = None,
                instances: int = 1, instance_sample: str = "first",
                seed: int = 0) -> dict:
    """Boot @activity ONCE, then harvest up to @instances pre-sampled instances
    in-place (Lever B) and write a shard (0..K rows) + status.

    Multi-instance is cheap because `load_activity_instance_tro_state` applies an
    instance in-place (25 physics steps, no reload) — the 8-min boot is amortized.
    HONESTY GUARD: each instance must start UNSATISFIED; if the goal already holds
    at t=0 the instance is contaminated by objects a prior trajectory spawned
    (sliced halves / diced particles that in-place reset does not remove) — we skip
    it rather than record a spuriously-satisfied trajectory. So pose-only
    rearrangement tasks yield up to K rows; product-creating tasks safely yield ~1.
    """
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import time

    from omegaconf import OmegaConf
    from omnigibson.envs import VectorEnvironment

    from rlinf.envs.behavior import expert_planner as ep
    from rlinf.envs.behavior import sft_build as sb
    from rlinf.envs.behavior.instance_loader import (
        discover_activity_instance_files,
        load_activity_instance_tro_state,
    )
    from rlinf.envs.behavior.semantic_tools import SemanticACI
    from rlinf.envs.behavior.utils import setup_omni_cfg

    os.makedirs(out_dir, exist_ok=True)
    scene, inst_dir = resolve_scene_and_dir(activity, scene)
    status = {"activity": activity, "scene": scene, "obs_mode": obs_mode,
              "instances_requested": instances, "instances_tried": 0,
              "instances_kept": 0, "instances_contaminated": 0, "kept": False}

    cfg = OmegaConf.load(os.path.join(_rlinf_root(), CFG_REL))
    OmegaConf.update(cfg, "omni_config.env.env_wrapper", None, force_add=True)
    OmegaConf.update(cfg, "omni_config.task.activity_name", activity, force_add=True)
    OmegaConf.update(cfg, "omni_config.scene.scene_model", scene, force_add=True)
    defid = int(OmegaConf.select(cfg, "omni_config.task.activity_definition_id") or 0)
    omni_cfg = setup_omni_cfg(cfg)
    env = VectorEnvironment(1, OmegaConf.to_container(omni_cfg, resolve=True))
    env.reset()
    e0 = env.envs[0]
    task = e0.task

    files = discover_activity_instance_files(inst_dir, activity, defid, "tro_state")
    picks = [(f.instance_id, f.path) for f in files]
    if instance_sample == "random":
        random.Random(seed).shuffle(picks)
    picks = picks[:max(1, instances)]

    goal_lines = sb.goal_atoms_to_lines(task.ground_goal_state_options[0])
    rows, notes, per_inst_sec = [], [], []
    for iid, path in picks:
        t0 = time.time()
        load_activity_instance_tro_state(e0, iid, path, reset_scene=True)
        aci = SemanticACI(e0, obs_mode=obs_mode)
        status["instances_tried"] += 1
        if aci.is_success():  # contamination guard (see docstring) -- never fake
            status["instances_contaminated"] += 1
            if len(notes) < 8:
                notes.append(f"inst {iid}: pre-satisfied at t=0 -> skipped (contamination)")
            per_inst_sec.append(round(time.time() - t0, 1))
            continue
        initial_obs = aci.observe()
        traj = ep.run_expert(aci, task, task_description=sb.activity_to_task(activity))
        if "unsupported" not in status:  # planner coverage is instance-invariant
            status["unsupported"] = [list(u) if isinstance(u, (list, tuple)) else u
                                     for u in traj.unsupported]
        if traj.success:
            rows.append(sb.trajectory_to_row(traj, initial_obs, goal_lines, obs_mode))
            status["instances_kept"] += 1
        elif len(notes) < 8:
            fs = "; ".join(f"{s.tool}:{s.reason}" for s in traj.steps if not s.ok)
            notes.append(f"inst {iid}: unsat -> {fs[:160]}")
        per_inst_sec.append(round(time.time() - t0, 1))

    status["kept"] = status["instances_kept"] > 0
    status["notes"] = notes
    status["per_instance_sec"] = per_inst_sec
    status["mean_instance_sec"] = round(sum(per_inst_sec) / len(per_inst_sec), 1) \
        if per_inst_sec else None

    shard = os.path.join(out_dir, f"shard_{activity}.jsonl")
    with open(shard, "w") as f:
        for row in rows:
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
                         ("activity", "scene", "kept", "instances_tried",
                          "instances_kept", "instances_contaminated",
                          "unsupported", "error")})
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
    one.add_argument("--scene", default=None,
                     help="scene_model; auto-discovers the activity's home scene if omitted")
    one.add_argument("--instances", type=int, default=1,
                     help="max pre-sampled instances to harvest in this boot (Lever B)")
    one.add_argument("--instance-sample", default="first", choices=["first", "random"])
    one.add_argument("--seed", type=int, default=0)
    mrg = sub.add_parser("merge")
    mrg.add_argument("--out-dir", required=True)
    mrg.add_argument("--dataset", default="behavior_sft_train.jsonl")
    args = ap.parse_args()

    if args.cmd == "one":
        try:
            st = harvest_one(args.activity, args.out_dir, args.obs_mode,
                             scene=args.scene, instances=args.instances,
                             instance_sample=args.instance_sample, seed=args.seed)
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
