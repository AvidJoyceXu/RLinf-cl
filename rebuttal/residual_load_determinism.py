"""Does loading a residual checkpoint give the same weights every time?

Probe-state collection was verified byte-identical and residual eval takes the
distribution mean (no sampling), so a fixed pair of checkpoints ought to give a
fixed DC. It does not. This checks the remaining step: whether
load_residual_policy actually restores every parameter, or leaves some randomly
initialised (get_model builds the module first, and the file branch loads with
strict=False, which silently tolerates missing keys).

    python rebuttal/residual_load_determinism.py --checkpoint <hf_model_dir> --obs_dim 846
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--obs_dim", type=int, default=846)
    args = ap.parse_args()

    os.chdir(ANALYSIS_DIR)
    from analyze_correction_vector_field import get_eval_action, load_config, load_residual_policy

    cfg = load_config(args.config)
    cfg.actor.model.obs_dim = args.obs_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m1 = load_residual_policy(args.checkpoint, cfg, device)
    m2 = load_residual_policy(args.checkpoint, cfg, device)

    sd1, sd2 = m1.state_dict(), m2.state_dict()
    print("\n=== weight comparison across two loads of the SAME checkpoint ===")
    print(f"n params: {len(sd1)}")
    differing = []
    for k in sd1:
        if not torch.equal(sd1[k].float().cpu(), sd2[k].float().cpu()):
            d = (sd1[k].float() - sd2[k].float()).abs().max().item()
            differing.append((k, d, tuple(sd1[k].shape)))
    if differing:
        print(f"DIFFERING params: {len(differing)} of {len(sd1)}")
        for k, d, shape in differing[:20]:
            print(f"  {k:<50} shape={shape} max|Δ|={d:.6g}")
    else:
        print("all parameters identical")

    # What the metric actually consumes: the eval action on a fixed state batch.
    rng = np.random.default_rng(0)
    states = rng.standard_normal((256, args.obs_dim)).astype(np.float32)
    a1 = get_eval_action(m1, states, device)
    a2 = get_eval_action(m2, states, device)
    print("\n=== eval action on a fixed state batch ===")
    print(f"max|Δ action| = {np.abs(a1 - a2).max():.6g}")

    cos = np.sum(a1 * a2, axis=1) / (
        np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1) + 1e-8
    )
    print(f"mean cosine(load1, load2) = {cos.mean():.6f}  (1.0 == identical field)")
    if cos.mean() < 0.999:
        print("VERDICT: the same checkpoint yields a DIFFERENT residual field per load "
              "-- DC/RFC cannot be reproducible")
    else:
        print("VERDICT: loading is deterministic; look elsewhere for the DC spread")


if __name__ == "__main__":
    main()
