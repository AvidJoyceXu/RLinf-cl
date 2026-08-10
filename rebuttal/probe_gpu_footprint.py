#!/usr/bin/env python3
"""Measure the GPU footprint of the OpenVLA-OFT base policy at rollout batch sizes.

Answers "can this fit in the memory currently free on a shared GPU" with a
measurement instead of an estimate. Loads the frozen base model plus the
frozen visual encoder and runs one deterministic action prediction at a few
batch sizes, reporting peak reserved memory.

Run inside the container on a chosen GPU:
    CUDA_VISIBLE_DEVICES=0 /opt/venv/openvla-oft/bin/python rebuttal/probe_gpu_footprint.py
"""

import os

import numpy as np
import torch

CONFIG_DIR = "/workspace/RLinf/examples/embodiment/config"
CONFIG_NAME = "libero_object_task1_lora_residual_sac_openvlaoft_rgb"
BATCH_SIZES = (8, 16, 32)


def gib(num_bytes):
    return num_bytes / (1024**3)


def main():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from rlinf.models import get_model
    from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
        FrozenVisualEncoder,
    )

    os.environ.setdefault("EMBODIED_PATH", "/workspace/RLinf/examples/embodiment")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name=CONFIG_NAME)

    device = torch.device("cuda")
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU total {gib(total):.1f} GiB, free before load {gib(free_before):.1f} GiB")

    base_model = get_model(cfg.actor.base_model)
    base_model.eval().to(device)
    for param in base_model.parameters():
        param.requires_grad = False
    after_base = torch.cuda.memory_reserved()
    print(f"base model (OpenVLA-OFT, {cfg.actor.model.precision}) reserved: {gib(after_base):.1f} GiB")

    encoder = FrozenVisualEncoder().to(device)
    residual = get_model(cfg.actor.model).to(device)
    after_all = torch.cuda.memory_reserved()
    print(
        f"+ frozen ViT + residual policy reserved: {gib(after_all):.1f} GiB "
        f"(residual adds {gib(after_all - after_base):.2f} GiB)"
    )

    task_description = "pick up the cream cheese and place it in the basket"
    for batch_size in BATCH_SIZES:
        torch.cuda.reset_peak_memory_stats()
        env_obs = {
            "main_images": torch.randint(
                0, 256, (batch_size, 256, 256, 3), dtype=torch.uint8, device=device
            ),
            "wrist_images": torch.randint(
                0, 256, (batch_size, 256, 256, 3), dtype=torch.uint8, device=device
            ),
            "states": torch.zeros(batch_size, 8, dtype=torch.float32, device=device),
            "task_descriptions": [task_description] * batch_size,
        }
        try:
            with torch.no_grad():
                # Encode first: the base model's preprocessing rewrites
                # `env_obs["main_images"]` in place, which is also the order the
                # rollout worker uses.
                encoder.encode_frames(env_obs["main_images"], env_obs["main_images"])
                base_model.predict_action_batch(
                    env_obs=env_obs,
                    mode="eval",
                    calulate_logprobs=False,
                    calulate_values=False,
                    return_obs=False,
                    do_sample=False,
                )
            peak = torch.cuda.max_memory_reserved()
            free_now, _ = torch.cuda.mem_get_info()
            print(
                f"batch {batch_size:>3}: peak reserved {gib(peak):.1f} GiB, "
                f"free left on device {gib(free_now):.1f} GiB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"batch {batch_size:>3}: OOM ({str(exc).splitlines()[0]})")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
