import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.workers.actor.ma_megatron_actor_worker import _shift_logprobs_right


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
