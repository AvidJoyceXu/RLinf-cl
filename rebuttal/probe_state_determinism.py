"""Is probe-state collection reproducible?

analyze_correction_vector_field.py seeds the LIBERO env with a fixed seed (42)
and rolls the base policy out with do_sample=False, so two collections ought to
return byte-identical states -- and therefore an identical DC. Measured DC on a
fixed checkpoint pair instead moves over a range as wide as the whole published
DC table, so one of those two assumptions does not hold in practice.

This collects the same task twice in one process and reports where, if anywhere,
the two trajectories diverge.

    python rebuttal/probe_state_determinism.py --task 1 --episodes 2 --obs_mode rgb
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ANALYSIS_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples/embodiment/eval/correction_field_analysis"
)
sys.path.insert(0, str(ANALYSIS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_lora_config_object.yaml")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--obs_mode", default="rgb", choices=["privileged", "rgb"])
    args = ap.parse_args()

    os.chdir(ANALYSIS_DIR)
    from analyze_correction_vector_field import (
        collect_state_samples_from_base_rollout,
        create_simple_libero_env,
        load_config,
    )
    from rlinf.models.embodiment.openvla_oft import get_model as get_base_model

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = cfg.actor.base_model.copy()
    base_cfg.model_path = cfg.residual_policy.base_model_path
    base_model = get_base_model(base_cfg)
    base_model.eval().to(device)

    visual_encoder = None
    if args.obs_mode == "rgb":
        from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
            DEFAULT_VISUAL_ENCODER,
            FrozenVisualEncoder,
        )

        visual_encoder = FrozenVisualEncoder(model_path=DEFAULT_VISUAL_ENCODER)
        visual_encoder.eval().to(device)

    suite = (
        cfg.env.train.task_suite_name
        if hasattr(cfg.env, "train")
        else cfg.env.task_suite_name
    )

    runs = []
    for rep in (1, 2):
        print(f"\n########## collection {rep} ##########")
        env, _ = create_simple_libero_env(suite, args.task, visual_encoder=visual_encoder)
        states = collect_state_samples_from_base_rollout(
            base_model, suite, args.task, env, cfg, device,
            num_episodes=args.episodes, max_steps=240,
        )
        env.close()
        runs.append(states)

    a, b = runs
    print("\n=== comparison ===")
    print(f"shapes: {a.shape} vs {b.shape}")
    if a.shape != b.shape:
        print("VERDICT: not reproducible -- different number of states collected")
        return

    same = np.array_equal(a, b)
    print(f"byte-identical: {same}")
    if same:
        print("VERDICT: probe-state collection IS deterministic; "
              "cross-repeat DC spread must come from somewhere else")
        return

    diff = np.abs(a - b)
    per_step = diff.max(axis=1)
    first = int(np.argmax(per_step > 0))
    print(f"first differing state index: {first} of {len(a)} "
          f"(= step {first % 240} of episode {first // 240})")
    print(f"max abs diff overall: {diff.max():.6g}")
    print(f"max abs diff at first divergence: {per_step[first]:.6g}")
    # How fast does the divergence grow -- chaotic amplification or a constant offset?
    for k in (0, 1, 2, 5, 10, 20, 50):
        idx = first + k
        if idx < len(per_step):
            print(f"  +{k:>3} steps: max|Δ| = {per_step[idx]:.6g}")
    print("VERDICT: probe-state collection is NOT reproducible despite the fixed seed")


if __name__ == "__main__":
    main()
