"""Build the online-RL prompt dataset for the BEHAVIOR agent.

One row = one task instance the agent loop will roll out. Deliberately **sim-free**
(it only reads instance filenames), so the dataset can be regenerated in the
trainer venv without booting IsaacSim.

Row shape matches what ``create_rl_dataset`` feeds the agent loop:

    {"prompt": "<activity, human readable>",          # placeholder; the real prompt is
     "solutions": "{\\"activity\\":..., \\"instance_id\\":...}"}   # built in pre_process_query

``solutions`` (``data.answer_key``) carries the episode spec rather than an answer —
there is no answer to leak. The ground goal atoms are instance-resolved and arrive
from the env server at ``session_start``, which is what keeps this builder sim-free.

**Splits are the point.** ``--split`` writes disjoint activity/instance sets so
contamination is structurally impossible rather than a discipline:

    S0  seen activity, seen instance      sanity only — SFT memorizes this
    S1  seen activity, unseen instance    spatial robustness
    S2  UNSEEN activity, seen scene       the primary metric (currently 0%)
    S3  unseen scene                      transfer

Freeze S2/S3 *before* scaling the harvest, or every later result is contaminated.

    with-env python -m rlinf.envs.behavior.build_rl_dataset \\
        --out /data/behavior-data/rl --instances-per-activity 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Held-out activities. Chosen for predicate diversity rather than convenience:
# a transform task, a multi-object rearrangement, a container-ordering task and a
# device task, so S2 measures generalization across *kinds* of plan, not just
# across names. Edit this list once, then never again — that is the whole point.
DEFAULT_HELDOUT_ACTIVITIES = [
    "cook_bacon",
    "collecting_childrens_toys",
    "clearing_food_from_table_into_fridge",
    "turning_on_radio",
]
DEFAULT_HELDOUT_SCENE = "house_double_floor_upper"


def discover(activities: list[str], per_activity: int, seed: int) -> list[dict]:
    from rlinf.envs.behavior.harvest_sft import resolve_scene_and_dir
    from rlinf.envs.behavior.instance_loader import discover_activity_instance_files

    rng = random.Random(seed)
    rows = []
    for activity in activities:
        try:
            scene, inst_dir = resolve_scene_and_dir(activity)
        except Exception as ex:                       # activity has no shipped instances
            print(f"  skip {activity}: {type(ex).__name__}: {ex}")
            continue
        files = sorted(discover_activity_instance_files(inst_dir, activity, 0, "tro_state"),
                       key=lambda f: f.instance_id)
        if not files:
            print(f"  skip {activity}: no pre-sampled instances")
            continue
        picked = files if per_activity <= 0 else rng.sample(
            files, min(per_activity, len(files)))
        for f in sorted(picked, key=lambda f: f.instance_id):
            rows.append({"activity": activity, "scene": scene, "instance_id": f.instance_id})
    return rows


def discover_textworld(verified_path: str) -> list[str]:
    """Activities for the BEHAVIOR-TextWorld benchmark.

    Read from the file ``symbolic_expert --all`` writes rather than recomputed here:
    membership means "the symbolic expert actually reached the BDDL goal", which
    costs a full solve over ~800 activities. Provenance beats convenience -- a
    benchmark whose membership rule is a ten-minute side effect of the dataset
    builder is one nobody can reproduce.
    """
    if not os.path.exists(verified_path):
        raise SystemExit(
            f"{verified_path} not found. Produce it first:\n"
            f"  python -m rlinf.envs.behavior.symbolic_expert --all "
            f"--out {verified_path}"
        )
    with open(verified_path) as f:
        return sorted(json.load(f)["solved"])


def build_textworld(out_dir: str, verified_path: str, s2_frac: float,
                    seed: int) -> None:
    """Write train / val_s2 for the symbolic benchmark.

    ONLY those two splits exist here, and that is a property of the backend rather
    than a shortcut. S1 (unseen instance) and S3 (unseen scene) are meaningless
    without geometry: BDDL ships exactly one definition per activity, so there is
    one start state each and no scene at all. S2 -- held-out ACTIVITIES -- is the
    primary metric anyway, and this backend is the first one that can measure it at
    scale: hundreds of held-out activities instead of four.
    """
    activities = discover_textworld(verified_path)
    rng = random.Random(seed)
    shuffled = list(activities)
    rng.shuffle(shuffled)
    n_s2 = max(1, int(round(len(shuffled) * s2_frac)))
    s2_acts = sorted(shuffled[:n_s2])
    train_acts = sorted(shuffled[n_s2:])

    os.makedirs(out_dir, exist_ok=True)
    for name, acts in (("train", train_acts), ("val_s2", s2_acts)):
        specs = [{"activity": a, "instance_id": 0} for a in acts]
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in to_dataset_rows(specs):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:8s} {len(specs):5d} rows  {len(acts):4d} activities  -> {path}")

    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"backend": "textworld",
                   "train_activities": train_acts,
                   "s2_heldout_activities": s2_acts,
                   "verified_source": verified_path,
                   "seed": seed}, f, indent=2)
    print(f"\nS2 = {len(s2_acts)} held-out activities, never trained on.")
    print("Freeze this split now — it is the primary metric and cannot be re-drawn later.")


def to_dataset_rows(specs: list[dict]) -> list[dict]:
    return [
        {"prompt": spec["activity"].replace("_", " "),
         "solutions": json.dumps(spec, ensure_ascii=False)}
        for spec in specs
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--activities", default=None,
                    help="comma-separated, or @file; default: read from --manifest")
    ap.add_argument("--manifest",
                    default="/data/behavior-data/sft_mi/harvest_manifest.json",
                    help="harvest manifest; activities with kept=true are the ones "
                         "the expert planner can actually solve (25 of 49 attempted)")
    ap.add_argument("--instances-per-activity", type=int, default=8)
    ap.add_argument("--val-instances-per-activity", type=int, default=2)
    ap.add_argument("--heldout-activities", default=",".join(DEFAULT_HELDOUT_ACTIVITIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="omnigibson",
                    choices=["omnigibson", "textworld"],
                    help="textworld: activity-level splits over the verified-solvable "
                         "symbolic benchmark; no instances, no scenes")
    ap.add_argument("--verified", default="/data/behavior-data/tw_verified.json",
                    help="textworld only: output of `symbolic_expert --all`")
    ap.add_argument("--s2-frac", type=float, default=0.2,
                    help="textworld only: fraction of activities held out as S2")
    args = ap.parse_args()

    if args.backend == "textworld":
        build_textworld(args.out, args.verified, args.s2_frac, args.seed)
        return

    if args.activities and args.activities.startswith("@"):
        activities = [l.strip() for l in open(args.activities[1:]) if l.strip()]
    elif args.activities:
        activities = [a.strip() for a in args.activities.split(",") if a.strip()]
    else:
        if not os.path.exists(args.manifest):
            ap.error(f"no --activities given and manifest not found: {args.manifest}\n"
                     "Pass --activities explicitly, or point --manifest at the "
                     "harvest_manifest.json produced by harvest_sft.py.")
        manifest = json.load(open(args.manifest))
        activities = [r["activity"] for r in manifest if r.get("kept")]
        print(f"activities from manifest: {len(activities)} solvable "
              f"of {len(manifest)} attempted")

    heldout = {a for a in args.heldout_activities.split(",") if a}
    train_acts = [a for a in activities if a not in heldout]
    s2_acts = [a for a in activities if a in heldout]

    os.makedirs(args.out, exist_ok=True)
    rng_train = discover(train_acts, args.instances_per_activity, args.seed)
    # S1 draws from the SAME activities with a different seed, so instances differ
    # from train's; a shared seed would silently make S1 a copy of train.
    rng_s1 = discover(train_acts, args.val_instances_per_activity, args.seed + 1000)
    train_keys = {(r["activity"], r["instance_id"]) for r in rng_train}
    rng_s1 = [r for r in rng_s1 if (r["activity"], r["instance_id"]) not in train_keys]
    rng_s2 = discover(s2_acts, args.val_instances_per_activity + 3, args.seed + 2000)

    for name, specs in (("train", rng_train), ("val_s1", rng_s1), ("val_s2", rng_s2)):
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in to_dataset_rows(specs):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        acts = len({s["activity"] for s in specs})
        print(f"{name:8s} {len(specs):5d} rows  {acts:3d} activities  -> {path}")

    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"train_activities": train_acts,
                   "s2_heldout_activities": s2_acts,
                   "s3_heldout_scene": DEFAULT_HELDOUT_SCENE,
                   "seed": args.seed}, f, indent=2)
    print(f"\nS2 held-out activities (never trained on): {s2_acts}")
    print("Freeze this split now — it is the primary metric and cannot be re-drawn later.")


if __name__ == "__main__":
    main()
