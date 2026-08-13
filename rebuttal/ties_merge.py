"""TIES-Merging of RLinf LoRA residual experts.

TIES (Yadav et al., 2023) merges task vectors in three steps -- trim to the
top-K% of entries by magnitude, elect a sign per coordinate, then average only
the entries that agree with the elected sign. The reference implementation is
`ties-merging/src/utils/merge_utils.py`; the three functions are reproduced here
on tensors rather than reused, because that repo's entry point is wired to its
own T5/IA3 data and eval stack.

Two adaptations are needed to apply it to our residual experts, both forced by
the structure of the checkpoints rather than chosen:

1. TIES is element-wise and therefore assumes all task vectors live in a shared
   coordinate system. A LoRA layer W = B @ A is invariant to
   W -> (B R^-1)(R A), and our experts are trained independently, so their
   factors are not comparable -- measured cosine between A factors of different
   experts is ~0. TIES is therefore applied to the *effective* update
   W_i = B_i @ A_i and the result is refactorised by truncated SVD at the
   original rank.
2. A task vector is normally theta_i - theta_pre. Residual experts have no
   shared pretrained initialisation, but B is initialised at ~0
   (`normal_(std=0.01)`), so the effective update at initialisation is ~0 and
   W_i itself is the task vector.

Parameters that are not part of a LoRA pair (actor_mean, actor_logstd, the
action scale/bias buffers) are averaged, matching what the other merge operators
do with them, so that the comparison isolates the LoRA merging rule.

    python rebuttal/ties_merge.py --checkpoints <dir>... --output <dir> [--k 0.2] [--lam 1.0]
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LORA_PAIRS = [("fc1_A.weight", "fc1_B.weight"),
              ("fc2_A.weight", "fc2_B.weight"),
              ("fc3_A.weight", "fc3_B.weight")]


def trim(flat: torch.Tensor, k: float) -> torch.Tensor:
    """Keep the top-k fraction of entries by magnitude, per task vector."""
    n_keep = max(1, int(flat.shape[1] * k))
    thresh = flat.abs().kthvalue(flat.shape[1] - n_keep + 1, dim=1, keepdim=True).values
    return flat * (flat.abs() >= thresh)


def elect_sign(flat: torch.Tensor) -> torch.Tensor:
    """Elect one sign per coordinate by total mass (TIES 'mass' rule)."""
    sign = torch.sign(flat.sum(dim=0))
    # coordinates with zero total mass fall back to the majority sign
    majority = torch.sign(torch.sign(flat).sum(dim=0))
    sign[sign == 0] = majority[sign == 0]
    return sign


def disjoint_mean(flat: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Average only the entries agreeing with the elected sign."""
    keep = torch.where(sign.unsqueeze(0) > 0, flat > 0, flat < 0)
    selected = flat * keep
    counts = keep.sum(dim=0).clamp(min=1)
    return selected.sum(dim=0) / counts


def ties(vectors: torch.Tensor, k: float, lam: float) -> torch.Tensor:
    """vectors: [n_tasks, d] -> merged [d]"""
    trimmed = trim(vectors, k)
    return lam * disjoint_mean(trimmed, elect_sign(trimmed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=float, default=0.2, help="fraction of entries kept when trimming")
    ap.add_argument("--lam", type=float, default=1.0, help="scaling applied to the merged task vector")
    args = ap.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("need at least two checkpoints to merge")

    states = []
    for c in args.checkpoints:
        sd = {}
        for shard in sorted(c.glob("*.safetensors")):
            sd.update(load_file(str(shard)))
        if not sd:
            raise SystemExit(f"no .safetensors in {c}")
        states.append(sd)

    keys = set(states[0])
    for sd in states[1:]:
        if set(sd) != keys:
            raise SystemExit("checkpoints do not share the same tensor set")

    merged = {}
    handled = set()

    for a_key, b_key in LORA_PAIRS:
        if a_key not in keys or b_key not in keys:
            continue
        rank = states[0][a_key].shape[0]
        effective = torch.stack([
            (sd[b_key].float() @ sd[a_key].float()).flatten() for sd in states
        ])
        out_shape = (states[0][b_key].shape[0], states[0][a_key].shape[1])
        w = ties(effective, args.k, args.lam).reshape(out_shape)

        # refactorise at the original rank so the merged expert stays a LoRA policy
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        root = torch.diag(torch.sqrt(s[:rank]))
        b_new = (u[:, :rank] @ root).to(states[0][b_key].dtype)
        a_new = (root @ vh[:rank, :]).to(states[0][a_key].dtype)

        recon_err = (b_new.float() @ a_new.float() - w).norm() / (w.norm() + 1e-12)
        print(f"  {a_key.split('.')[0]}: TIES on W{tuple(out_shape)}, "
              f"rank-{rank} refactor rel-err {recon_err:.4f}")

        merged[a_key], merged[b_key] = a_new, b_new
        handled.update({a_key, b_key})

    for k in sorted(keys - handled):
        ts = [sd[k] for sd in states]
        if ts[0].is_floating_point():
            merged[k] = torch.stack([t.float() for t in ts]).mean(0).to(ts[0].dtype)
        else:
            merged[k] = ts[0]

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(args.output / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in merged.values())},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in merged},
    }
    (args.output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"TIES merged {len(args.checkpoints)} experts (K={args.k}, lambda={args.lam}) -> {args.output}")


if __name__ == "__main__":
    main()
