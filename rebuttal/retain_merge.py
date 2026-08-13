"""RETAIN weight-space interpolation of two OpenVLA-OFT checkpoints.

Implements the merge of the RETAIN paper (arXiv 2512.08333), Eq. 2:

    theta_tilde = (1 - alpha) * theta_a + alpha * theta_b

and, applied repeatedly with `theta_a` set to the previous result, the continual
variant of Eq. 4:

    theta_tilde_n = (1 - alpha) * theta_tilde_{n-1} + alpha * theta_ft,n

Both checkpoints must share the tensor set and shard layout (they do here: base
and every LIBERO SFT checkpoint carry the same 982 tensors across 4 shards).
Shards are processed one at a time so peak RAM stays at roughly one shard pair.

Note on scope: RLinf's OpenVLA-OFT wrapper loads only the safetensors
(vision_backbone / language_model / projector). The `action_head--*.pt` and
`proprio_projector--*.pt` files that the openvla-oft training script writes
alongside are not read by RLinf, so they are copied through untouched rather
than interpolated -- interpolating them would have no effect on evaluation.

    python rebuttal/retain_merge.py --a <ckpt_a> --b <ckpt_b> --alpha 0.5 --out <dir>
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def shard_files(root: Path):
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return sorted(set(wm.values())), wm
    shards = sorted(p.name for p in root.glob("*.safetensors"))
    return shards, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="theta_a (kept with weight 1-alpha)")
    ap.add_argument("--b", type=Path, required=True, help="theta_b (kept with weight alpha)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a_shards, _ = shard_files(args.a)
    b_shards, _ = shard_files(args.b)
    if a_shards != b_shards:
        raise SystemExit(
            f"shard layouts differ:\n  a: {a_shards}\n  b: {b_shards}\n"
            "interpolation needs a matching layout"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    alpha = args.alpha

    total = 0
    for shard in a_shards:
        merged = {}
        with safe_open(str(args.a / shard), framework="pt") as fa, \
             safe_open(str(args.b / shard), framework="pt") as fb:
            ka, kb = set(fa.keys()), set(fb.keys())
            if ka != kb:
                raise SystemExit(
                    f"{shard}: tensor sets differ "
                    f"(only in a: {sorted(ka - kb)[:5]}, only in b: {sorted(kb - ka)[:5]})"
                )
            for k in sorted(ka):
                ta, tb = fa.get_tensor(k), fb.get_tensor(k)
                if ta.shape != tb.shape:
                    raise SystemExit(f"{shard}:{k} shape {ta.shape} vs {tb.shape}")
                if ta.is_floating_point():
                    dt = ta.dtype
                    merged[k] = ((1.0 - alpha) * ta.float() + alpha * tb.float()).to(dt)
                else:
                    # buffers such as position ids are not interpolated
                    if not torch.equal(ta, tb):
                        raise SystemExit(f"{shard}:{k} non-float tensors differ; refusing to guess")
                    merged[k] = ta
                total += 1
        save_file(merged, str(args.out / shard), metadata={"format": "pt"})
        print(f"  wrote {shard} ({len(merged)} tensors)")
        del merged

    # The OFT continuous action head and proprio projector live outside the
    # safetensors as standalone .pt files, and they are part of theta -- the L1
    # regression head is what run_libero_eval.py actually decodes actions with.
    # They are interpolated too. This is well-defined here because the five
    # checkpoints' action heads share a lineage (pairwise cosine ~0.94), unlike
    # e.g. independently initialised LoRA factors.
    for p in sorted(args.a.glob("*.pt")):
        q = args.b / p.name
        if not q.exists():
            raise SystemExit(f"{p.name} missing from {args.b}; cannot interpolate")
        da = torch.load(p, map_location="cpu")
        db = torch.load(q, map_location="cpu")
        if da.keys() != db.keys():
            raise SystemExit(f"{p.name}: key sets differ")
        out = {}
        for k in da:
            ta, tb = da[k], db[k]
            if torch.is_tensor(ta) and ta.is_floating_point():
                out[k] = ((1.0 - alpha) * ta.float() + alpha * tb.float()).to(ta.dtype)
            else:
                out[k] = ta
        torch.save(out, args.out / p.name)
        print(f"  interpolated {p.name} ({len(out)} tensors)")

    # dataset_statistics.json holds the action normalization quantiles used to
    # un-normalize predicted actions. It is not a model parameter, so no
    # weight-space rule covers it -- but it is checkpoint-specific (between two of
    # our LIBERO-Object SFT checkpoints q01 differs by 38%, q99 by 29%), and
    # copying one side's statistics onto interpolated weights puts the decoded
    # action scale out of step with the policy. It is interpolated with the same
    # alpha, which is the only choice consistent with the weight merge.
    stats_name = "dataset_statistics.json"
    sa, sb = args.a / stats_name, args.b / stats_name
    if sa.exists() and sb.exists():
        da, db = json.loads(sa.read_text()), json.loads(sb.read_text())

        def blend(x, y):
            if isinstance(x, dict):
                return {k: blend(x[k], y[k]) for k in x if k in y}
            if isinstance(x, list):
                return [blend(u, v) for u, v in zip(x, y)]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return (1.0 - alpha) * x + alpha * y
            return x

        if set(da) != set(db):
            raise SystemExit(f"{stats_name}: top-level keys differ ({set(da)} vs {set(db)})")
        (args.out / stats_name).write_text(json.dumps(blend(da, db), indent=2))
        print(f"  interpolated {stats_name}")

    # Everything else the loader needs: config, tokenizer, index, the prismatic
    # modeling code.
    for p in sorted(args.a.iterdir()):
        if (
            p.is_file()
            and not p.name.endswith(".safetensors")
            and not p.name.endswith(".pt")
            and p.name != stats_name
        ):
            shutil.copy2(p, args.out / p.name)
    for extra in ("lora_adapter",):
        src = args.a / extra
        if src.is_dir() and not (args.out / extra).exists():
            shutil.copytree(src, args.out / extra)

    print(f"interpolated {total} tensors with alpha={alpha}")
    print(f"  (1-alpha)*{args.a.name} + alpha*{args.b.name} -> {args.out}")


if __name__ == "__main__":
    main()
