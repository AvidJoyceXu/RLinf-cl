from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlinf.utils.data_iter_utils import get_seqlen_balanced_partitions
from rlinf.utils.distributed import RolloutDataBalance


def _gloo_ragged_balance_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        response_counts = [1, 2] if rank == 0 else [3, 4, 5]
        batch_size = len(response_counts)
        seq_len = 8
        response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for row, count in enumerate(response_counts):
            response_mask[row, 1 : 1 + count] = True
        # Simulate the post-pack group factor P_local / B for B=8, then remove
        # P_local before cross-rank redistribution as the actor does.
        local_packed_sequences = batch_size
        source_group_scale = local_packed_sequences / 8
        loss_scales = response_mask.float() * source_group_scale
        loss_scales.div_(local_packed_sequences)
        batch = {
            "input_ids": torch.arange(batch_size * seq_len).reshape(
                batch_size, seq_len
            ),
            "prompt_lengths": torch.ones(batch_size, dtype=torch.int32),
            "response_lengths": torch.tensor(response_counts, dtype=torch.int32),
            "response_mask": response_mask,
            "advantages": response_mask.float(),
            "loss_scales": loss_scales,
        }
        pad = {
            key: torch.zeros((1, *value.shape[1:]), dtype=value.dtype)
            for key, value in batch.items()
        }

        balanced = RolloutDataBalance.from_rollout_batches_dynamic(
            rollout_batches=batch,
            dp_world_size=world_size,
            dp_rank=rank,
            dp_group=dist.group.WORLD,
            rollout_batch_pad=pad,
            split_fix_chunk=1,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        balanced["loss_scales"].mul_(balanced["input_ids"].shape[0])

        assert balanced["input_ids"].shape[0] == 3
        valid_tokens = balanced["response_mask"].sum().to(torch.long)
        padding_rows = (balanced["response_lengths"] == 0).sum().to(torch.long)
        dist.all_reduce(valid_tokens)
        dist.all_reduce(padding_rows)
        assert valid_tokens.item() == sum([1, 2, 3, 4, 5])
        assert padding_rows.item() == 1
        torch.testing.assert_close(
            balanced["loss_scales"][balanced["response_mask"]],
            torch.full(
                (int(balanced["response_mask"].sum().item()),),
                3 / 8,
                dtype=balanced["loss_scales"].dtype,
            ),
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_preserves_ragged_tokens_and_zero_mask_padding():
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "gloo-init")
        mp.spawn(
            _gloo_ragged_balance_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )
