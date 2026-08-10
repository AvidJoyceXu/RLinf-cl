#!/usr/bin/env python3
"""Smoke tests for the RGB residual observation pathway.

Covers the two things that are easy to get silently wrong:

1. Frame stacking across reset boundaries. `_wrap_obs` is called twice during
   an auto-reset (once for the final observation, once after the reset), so a
   naive "update prev inside _wrap_obs" collapses the stack onto a single
   frame for the envs that did *not* reset.
2. The frozen encoder's output width and determinism, and that the assembled
   observation has the width the RGB configs declare.

Run inside the container:
    /opt/venv/openvla-oft/bin/python rebuttal/smoke_test_rgb_obs.py
"""

import sys

import numpy as np
import torch

PROPRIO_DIM = 39
IMG_SIZE = 32  # small stand-in for the 256x256 agentview frame
EXPECTED_OBS_DIM = 846


def make_raw_obs(num_envs, marker):
    """Synthetic LIBERO raw observations whose content encodes the timestep."""
    obs_list = []
    for env_id in range(num_envs):
        value = marker * 10 + env_id
        obs_list.append(
            {
                "agentview_image": np.full((IMG_SIZE, IMG_SIZE, 3), value, np.uint8),
                "robot0_eye_in_hand_image": np.full(
                    (IMG_SIZE, IMG_SIZE, 3), value, np.uint8
                ),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
                "robot0_proprio-state": np.full(PROPRIO_DIM, float(value)),
            }
        )
    return obs_list


def make_env(num_envs):
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = object.__new__(LiberoEnv)
    env.num_envs = num_envs
    env.obs_mode = "rgb"
    env.num_stack_frames = 2
    env._prev_frame = None
    env._cur_frame = None
    env.task_descriptions = ["dummy task"] * num_envs
    return env


def proprio_halves(obs):
    """Split the stacked proprio into (previous frame, current frame) markers."""
    stacked = obs["rl_proprio_stacked"]
    return stacked[:, 0], stacked[:, PROPRIO_DIM]


def test_frame_stacking():
    num_envs = 2
    env = make_env(num_envs)

    # --- initial reset: previous frame is undefined, so it duplicates ---------
    obs0 = env._wrap_obs(make_raw_obs(num_envs, 0), reset_env_idx=[0, 1])
    prev, cur = proprio_halves(obs0)
    assert torch.equal(prev, cur), f"reset must duplicate the frame, got {prev} vs {cur}"
    assert cur.tolist() == [0.0, 1.0], cur

    # --- two ordinary steps: the stack advances by exactly one step -----------
    for marker in (1, 2):
        env._prev_frame = env._cur_frame  # what LiberoEnv.step does
        obs = env._wrap_obs(make_raw_obs(num_envs, marker))
        prev, cur = proprio_halves(obs)
        expected_prev = [(marker - 1) * 10.0, (marker - 1) * 10.0 + 1]
        expected_cur = [marker * 10.0, marker * 10.0 + 1]
        assert prev.tolist() == expected_prev, (marker, prev.tolist())
        assert cur.tolist() == expected_cur, (marker, cur.tolist())
        assert torch.equal(
            obs["prev_main_images"][:, 0, 0, 0].double().cpu(), prev.double().cpu()
        ), "prev image must match the prev proprio frame"

    # --- auto-reset: env 0 resets, env 1 keeps stepping -----------------------
    env._prev_frame = env._cur_frame
    final_obs = env._wrap_obs(make_raw_obs(num_envs, 3))
    final_prev, final_cur = proprio_halves(final_obs)
    assert final_prev.tolist() == [20.0, 21.0]
    assert final_cur.tolist() == [30.0, 31.0]
    final_image_snapshot = final_obs["main_images"].clone()

    # LiberoEnv.reset re-wraps *all* envs after replacing only the reset ones.
    raw_after_reset = make_raw_obs(num_envs, 3)
    raw_after_reset[0] = make_raw_obs(1, 9)[0]  # env 0 restarted
    obs_after_reset = env._wrap_obs(raw_after_reset, reset_env_idx=[0])
    prev, cur = proprio_halves(obs_after_reset)
    assert prev[0] == cur[0] == 90.0, f"reset env must duplicate: {prev[0]}, {cur[0]}"
    assert prev[1] == 21.0 and cur[1] == 31.0, (
        f"non-reset env must keep its stack, got prev={prev[1]} cur={cur[1]}"
    )
    assert torch.equal(final_obs["main_images"], final_image_snapshot), (
        "the reset must not mutate an observation that was already handed out"
    )

    # --- the next step advances from the post-reset frame ---------------------
    env._prev_frame = env._cur_frame
    obs_next = env._wrap_obs(make_raw_obs(num_envs, 4))
    prev, cur = proprio_halves(obs_next)
    assert prev.tolist() == [90.0, 31.0], prev.tolist()
    assert cur.tolist() == [40.0, 41.0], cur.tolist()

    # --- privileged mode must be untouched ------------------------------------
    priv_env = make_env(num_envs)
    priv_env.obs_mode = "privileged"
    priv_obs = priv_env._wrap_obs(make_raw_obs(num_envs, 0), reset_env_idx=[0, 1])
    assert "object_to_robot_relations" in priv_obs
    assert "prev_main_images" not in priv_obs
    assert "rl_proprio_stacked" not in priv_obs

    print("[ok] frame stacking across reset boundaries")


def test_visual_encoder():
    from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
        FrozenVisualEncoder,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenVisualEncoder().to(device)
    assert encoder.feature_dim == 384, encoder.feature_dim
    assert not any(p.requires_grad for p in encoder.parameters()), "encoder must be frozen"

    images = torch.randint(0, 256, (3, 256, 256, 3), dtype=torch.uint8)
    features = encoder(images)
    assert features.shape == (3, 384), features.shape
    assert features.dtype == torch.float32, features.dtype
    assert torch.isfinite(features).all()

    # Frozen and deterministic: the same frame must encode identically.
    assert torch.allclose(features, encoder(images), atol=1e-5)

    # Distinct frames must produce distinct features, otherwise the stacked
    # observation would carry no temporal information.
    assert not torch.allclose(features[0], features[1], atol=1e-3)

    stacked = encoder.encode_frames(images, images)
    assert stacked.shape == (3, 768), stacked.shape
    assert torch.allclose(stacked[:, :384], stacked[:, 384:], atol=1e-5)

    # The full residual observation must match what the RGB configs declare.
    obs_dim = stacked.shape[1] + 2 * PROPRIO_DIM
    assert obs_dim == EXPECTED_OBS_DIM, obs_dim

    # `train()` must not take the encoder out of eval mode.
    encoder.train()
    assert not encoder.vit.training, "frozen encoder must stay in eval mode"

    print(f"[ok] frozen visual encoder (feature_dim=384, obs_dim={obs_dim}, device={device})")


def test_libero_env_end_to_end():
    """Drive a real 2-env LIBERO instance in RGB mode through reset/step/auto-reset."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from rlinf.data.io_struct import EnvOutput
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
        FrozenVisualEncoder,
    )

    config_dir = "/workspace/RLinf/examples/embodiment/config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="libero_object_task1_lora_residual_sac_openvlaoft_rgb")

    env_cfg = cfg.env.eval
    num_envs = 2
    env_cfg.total_num_envs = num_envs
    env_cfg.max_episode_steps = 3  # force a truncation-driven auto-reset quickly

    env = LiberoEnv(env_cfg, num_envs, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        raw_obs, _ = env.reset()
        obs = EnvOutput(obs=raw_obs).to_dict()["obs"]
        assert obs["prev_main_images"] is not None, "RGB mode must ship the previous frame"
        assert obs["rl_flatten_obs"].shape == (num_envs, 2 * PROPRIO_DIM), (
            f"env must emit stacked proprio only, got {tuple(obs['rl_flatten_obs'].shape)}"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = FrozenVisualEncoder().to(device)
        features = encoder.encode_frames(obs["prev_main_images"], obs["main_images"])
        full_obs = torch.cat([features.cpu(), obs["rl_flatten_obs"]], dim=-1)
        assert full_obs.shape == (num_envs, EXPECTED_OBS_DIM), tuple(full_obs.shape)
        assert full_obs.shape[1] == cfg.actor.model.obs_dim, (
            f"assembled obs {full_obs.shape[1]} != configured obs_dim {cfg.actor.model.obs_dim}"
        )

        # On reset the two visual halves come from the same duplicated frame.
        assert torch.allclose(features[:, :384], features[:, 384:], atol=1e-5), (
            "reset frame must be duplicated in the visual stack"
        )

        saw_auto_reset = False
        for _ in range(4):
            actions = np.zeros((num_envs, 7), dtype=np.float32)
            raw_obs, _, _, truncations, _ = env.step(actions)
            obs = EnvOutput(obs=raw_obs).to_dict()["obs"]
            assert obs["rl_flatten_obs"].shape == (num_envs, 2 * PROPRIO_DIM)
            assert obs["prev_main_images"].shape == obs["main_images"].shape
            if bool(truncations.any()):
                saw_auto_reset = True

        assert saw_auto_reset, "expected a truncation-driven auto-reset within 4 steps"
        # Nothing privileged may survive in RGB mode.
        assert "object_to_robot_relations" not in raw_obs

        print(f"[ok] LIBERO env end-to-end in rgb mode (obs_dim={full_obs.shape[1]}, auto-reset exercised)")
    finally:
        env.env.close()


def test_rollout_worker_encoding():
    """`_encode_visual_obs` must assemble [f(prev), f(cur), p_prev, p_cur] once."""
    from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
        FrozenVisualEncoder,
    )
    from rlinf.workers.rollout.hf.residual_rollout_worker import ResidualRolloutWorker

    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker = object.__new__(ResidualRolloutWorker)
    worker.visual_encoder = FrozenVisualEncoder().to(device)

    num_envs = 2
    prev_images = torch.randint(0, 256, (num_envs, 64, 64, 3), dtype=torch.uint8)
    cur_images = torch.randint(0, 256, (num_envs, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.arange(num_envs * 2 * PROPRIO_DIM, dtype=torch.float32).reshape(
        num_envs, 2 * PROPRIO_DIM
    )
    obs = {
        "prev_main_images": prev_images,
        "main_images": cur_images,
        "rl_flatten_obs": proprio.clone(),
    }

    encoded = worker._encode_visual_obs(obs)
    assert encoded["rl_flatten_obs"].shape == (num_envs, EXPECTED_OBS_DIM), tuple(
        encoded["rl_flatten_obs"].shape
    )

    # The proprio half must survive unchanged at the tail, in [prev, cur] order.
    tail = encoded["rl_flatten_obs"][:, -2 * PROPRIO_DIM :].cpu()
    assert torch.equal(tail, proprio), "proprio half must be appended verbatim"

    # The visual half must be the two frames encoded in [prev, cur] order.
    expected_visual = worker.visual_encoder.encode_frames(prev_images, cur_images)
    assert torch.allclose(
        encoded["rl_flatten_obs"][:, : 2 * 384], expected_visual, atol=1e-5
    )

    # Re-encoding the same dict must be a no-op rather than prepending twice.
    again = worker._encode_visual_obs(encoded)
    assert again["rl_flatten_obs"].shape == (num_envs, EXPECTED_OBS_DIM)

    # Privileged mode short-circuits.
    worker.visual_encoder = None
    passthrough = {"rl_flatten_obs": torch.zeros(num_envs, 88)}
    assert worker._encode_visual_obs(passthrough)["rl_flatten_obs"].shape == (num_envs, 88)

    print("[ok] rollout worker visual encoding (order, idempotence, passthrough)")


def test_privileged_mode_unregressed():
    """The privileged pathway must be byte-for-byte what it was before RGB mode."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from rlinf.data.io_struct import EnvOutput
    from rlinf.envs.libero.libero_env import LiberoEnv

    config_dir = "/workspace/RLinf/examples/embodiment/config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="libero_object_task1_lora_residual_sac_openvlaoft")

    env_cfg = cfg.env.eval
    num_envs = 2
    env_cfg.total_num_envs = num_envs
    env_cfg.max_episode_steps = 3

    env = LiberoEnv(env_cfg, num_envs, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        raw_obs, _ = env.reset()
        obs = EnvOutput(obs=raw_obs).to_dict()["obs"]
        assert obs["prev_main_images"] is None, "privileged mode must not ship stacked frames"
        assert obs["rl_flatten_obs"].shape == (num_envs, cfg.actor.model.obs_dim), (
            f"expected {cfg.actor.model.obs_dim}-d privileged obs, "
            f"got {tuple(obs['rl_flatten_obs'].shape)}"
        )
        for _ in range(4):
            raw_obs, _, _, _, _ = env.step(np.zeros((num_envs, 7), dtype=np.float32))
            obs = EnvOutput(obs=raw_obs).to_dict()["obs"]
            assert obs["rl_flatten_obs"].shape == (num_envs, cfg.actor.model.obs_dim)
        print(f"[ok] privileged mode unregressed (obs_dim={cfg.actor.model.obs_dim})")
    finally:
        env.env.close()


if __name__ == "__main__":
    failures = []
    for test in (
        test_frame_stacking,
        test_visual_encoder,
        test_rollout_worker_encoding,
        test_libero_env_end_to_end,
        test_privileged_mode_unregressed,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - smoke test reports and continues
            import traceback

            traceback.print_exc()
            failures.append(f"{test.__name__}: {exc}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print("\nall smoke tests passed")
