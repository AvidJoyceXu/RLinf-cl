import torch

from rlinf.algorithms.loss_scales import group_scale
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.workers.actor.ma_megatron_actor_worker import (
    _dynamic_group_balance_factor,
    _shift_logprobs_right,
)


def test_shift_logprobs_right_preserves_values_and_gradients():
    logprobs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    shifted = _shift_logprobs_right(logprobs)

    torch.testing.assert_close(
        shifted, torch.tensor([[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]])
    )
    shifted.sum().backward()
    torch.testing.assert_close(
        logprobs.grad, torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    )


def test_zero_mask_policy_loss_preserves_distributed_metric_schema():
    values = torch.zeros((1, 3), dtype=torch.float32)
    kwargs = {
        "logprobs": values,
        "old_logprobs": values,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "advantages": torch.ones_like(values),
        "fast_path_zero_loss_mask": True,
    }

    _, zero_metrics = compute_ppo_actor_loss(
        **kwargs, loss_mask=torch.zeros_like(values, dtype=torch.bool)
    )
    _, normal_metrics = compute_ppo_actor_loss(
        **kwargs, loss_mask=torch.ones_like(values, dtype=torch.bool)
    )

    assert zero_metrics.keys() == normal_metrics.keys()


def test_ppo_clipping_matches_hand_calculation_for_both_advantage_signs():
    logprobs = torch.log(torch.tensor([[1.5, 0.5]], dtype=torch.float32))
    old_logprobs = torch.zeros_like(logprobs)
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float32)

    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=torch.ones_like(logprobs, dtype=torch.bool),
    )

    # Positive advantage clips ratio 1.5 to 1.2: -1.2. Negative advantage clips
    # ratio 0.5 to 0.8: +0.8. Their token mean is -0.2.
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    torch.testing.assert_close(metrics["actor/clip_fraction"], torch.tensor(1.0))


def test_group_scale_is_shared_by_policy_and_kl_without_mutating_advantages():
    batch = {
        "idx_to_traj": [0, 0, 1],
        "advantages": torch.tensor([[1.0], [2.0], [3.0]]),
        "loss_scales": torch.ones((3, 1)),
    }
    context = {
        "folding_scale": [],
        "data_parallel_world_size": 2,
        "actor_global_batch_size": 4,
    }

    scaled = group_scale(context, batch)

    torch.testing.assert_close(
        scaled["advantages"], torch.tensor([[1.0], [2.0], [3.0]])
    )
    torch.testing.assert_close(scaled["loss_scales"], torch.full((3, 1), 1.5))


def test_dynamic_group_balance_removes_source_rank_packed_count():
    # Rank-local packed counts 4 and 6 both become 5 after redistribution.
    # Their old post-pack group factors 4/8 and 6/8 must both become 5/8.
    left = (4 / 8) * _dynamic_group_balance_factor(
        local_training_sequences=4, balanced_training_sequences=5
    )
    right = (6 / 8) * _dynamic_group_balance_factor(
        local_training_sequences=6, balanced_training_sequences=5
    )

    assert left == right == 5 / 8


def test_trajectory_packing_preserves_sequence_mean_loss_objective():
    response_mask = torch.tensor(
        [
            [False, True, True, False, False, False],
            [False, False, False, True, True, False],
            [False, True, True, False, False, False],
        ]
    )
    loss_values = torch.tensor(
        [
            [0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 4.0, 0.0],
            [0.0, 5.0, 7.0, 0.0, 0.0, 0.0],
        ]
    )
    loss_scales = torch.tensor(
        [
            [0.0, 0.4, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.6, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 4, 5, 0],
            [6, 7, 8, 0, 0, 0],
        ]
    )
    batch = {
        "idx_to_traj": [0, 0, 1],
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0),
        "response_mask": response_mask,
        "position_ids": torch.arange(6).repeat(3, 1),
        "is_end": torch.tensor([False, True, True]),
        "prompt_lengths": torch.tensor([1, 3, 1]),
        "response_lengths": torch.tensor([2, 2, 2]),
        "prev_logprobs": loss_values,
        "ref_logprobs": torch.zeros_like(loss_values),
        "rewards": torch.zeros(3),
        "advantages": response_mask.float(),
        "loss_scales": loss_scales,
        "extra:idx_to_sub_traj": torch.zeros(3, dtype=torch.long),
    }

    def objective(values: dict[str, torch.Tensor]) -> torch.Tensor:
        token_loss = values["prev_logprobs"] * values["loss_scales"]
        sequence_loss = token_loss.sum(dim=-1) / values["response_mask"].sum(dim=-1)
        return sequence_loss.mean()

    before = objective(batch)
    packed = DynamicRolloutResult.pack_traj_batch(
        {"folding_scale": ["group_level"]}, batch
    )

    assert packed["input_ids"].shape[0] == 2
    torch.testing.assert_close(objective(packed), before)
