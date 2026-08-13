#!/usr/bin/env python3
"""Derive the RGB-observation configs from the privileged-state ones.

The RGB ablation must be a controlled comparison: every hyper-parameter of the
residual RL run (res_scale, learning rates, gamma, buffer size, env count,
reset ids, ...) has to stay byte-identical to the privileged run, and only the
observation pathway may change. Generating the configs instead of hand-writing
them makes that property auditable -- the diff below is the complete list of
what differs.

Usage:
    python rebuttal/make_rgb_configs.py                 # libero_object 1,6,7,8,9
    python rebuttal/make_rgb_configs.py --suite libero_spatial --tasks 0 2 3 6 7
"""

import argparse
import os
import re

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "embodiment",
    "config",
)

VISUAL_ENCODER_BLOCK = """
# Frozen visual encoder (RGB observation mode)
# Single third-person camera frame -> frozen ViT -> 384-d feature, mirroring the
# real-robot residual input. Two consecutive frames are stacked by the env and
# encoded in the rollout worker; the encoder is never trained.
visual_encoder:
  enabled: True
  model_path: "facebook/dinov2-small"
  image_size: 224
  pooling: "cls"
"""

OBS_DIM_NEW = (
    "obs_dim: 846 # RGB obs: 2 x 384 frozen-ViT features + 2 x 39 robot0_proprio-state"
)

# The two suites do not share an experiment_name convention (libero_object uses
# "<suite>_task<id>_single_trial_lora_openvlaoft", libero_spatial uses
# "task<id>_single_trial_lora_residual_sac_openvlaoft"), and obs_dim differs per
# suite, so both anchors are matched by pattern rather than by literal.
EXPERIMENT_NAME_RE = re.compile(r'(experiment_name:\s*")([^"]+)(")')
OBS_DIM_RE = re.compile(r"obs_dim:\s*\d+[^\n]*")


def sub_once(pattern, repl, text, suite, task_id, what):
    text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise ValueError(f"Anchor not found in {suite} task {task_id}: {what}")
    return text


def convert(text: str, suite: str, task_id: int) -> str:
    replacements = [
        (
            "- model/lora_residual_policy@actor.model",
            "- model/lora_residual_policy_rgb@actor.model",
        ),
        ("\n# Network Configuration", f"{VISUAL_ENCODER_BLOCK}\n# Network Configuration"),
    ]
    for old, new in replacements:
        if old not in text:
            raise ValueError(f"Anchor not found in {suite} task {task_id}: {old!r}")
        text = text.replace(old, new, 1)

    text = sub_once(EXPERIMENT_NAME_RE, r"\1\2_rgb\3", text, suite, task_id,
                    "experiment_name")
    text = sub_once(OBS_DIM_RE, OBS_DIM_NEW, text, suite, task_id, "obs_dim")

    # `obs_mode: rgb` makes the env emit stacked frames and stop emitting the
    # privileged object-to-eef relations entirely.
    anchor = f"    specific_reset_id: {task_id}"
    if text.count(anchor) != 2:
        raise ValueError(
            f"Expected 2 `specific_reset_id: {task_id}` lines (train + eval), "
            f"found {text.count(anchor)}"
        )
    text = text.replace(anchor, f'{anchor}\n    obs_mode: "rgb"')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 6, 7, 8, 9])
    args = parser.parse_args()

    for task_id in args.tasks:
        src = os.path.join(
            CONFIG_DIR, f"{args.suite}_task{task_id}_lora_residual_sac_openvlaoft.yaml"
        )
        dst = os.path.join(
            CONFIG_DIR,
            f"{args.suite}_task{task_id}_lora_residual_sac_openvlaoft_rgb.yaml",
        )
        with open(src) as f:
            text = f.read()
        with open(dst, "w") as f:
            f.write(convert(text, args.suite, task_id))
        print(f"wrote {os.path.relpath(dst)}")


if __name__ == "__main__":
    main()
