"""Verify a matched detect SFT Hugging Face export against its FSDP full weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(export_dir: Path, *, required: bool) -> dict:
    manifest = export_dir / "EXPORT_MANIFEST.sha256"
    if not manifest.is_file():
        if required:
            raise FileNotFoundError(f"missing export manifest: {manifest}")
        return {"manifest": None, "files": 0}
    checked = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*").removeprefix("./")
        path = export_dir / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"export hash mismatch for {relative}: {actual} != {expected}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"empty export manifest: {manifest}")
    return {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "files": checked,
    }


def verify_export(weights_path: Path, export_dir: Path, *, require_manifest: bool):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    manifest = verify_manifest(export_dir, required=require_manifest)
    config = AutoConfig.from_pretrained(export_dir)
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    model = AutoModelForCausalLM.from_pretrained(
        export_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    expected = torch.load(
        weights_path, map_location="cpu", weights_only=True, mmap=True
    )
    actual = model.state_dict()
    if expected.keys() != actual.keys():
        raise ValueError(
            "state-dict keys differ: "
            f"missing={sorted(expected.keys() - actual.keys())[:10]}, "
            f"extra={sorted(actual.keys() - expected.keys())[:10]}"
        )
    parameters = 0
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if expected_tensor.shape != actual_tensor.shape:
            raise ValueError(
                f"shape mismatch for {name}: "
                f"{tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
            )
        if expected_tensor.dtype != actual_tensor.dtype:
            raise ValueError(
                f"dtype mismatch for {name}: "
                f"{actual_tensor.dtype} != {expected_tensor.dtype}"
            )
        if not torch.equal(expected_tensor, actual_tensor.cpu()):
            raise ValueError(f"tensor content mismatch for {name}")
        parameters += expected_tensor.numel()
    return {
        **manifest,
        "weights_path": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "export_dir": str(export_dir),
        "architecture": config.architectures,
        "tokenizer_class": type(tokenizer).__name__,
        "tensors": len(expected),
        "parameters": parameters,
        "dtype": str(next(iter(expected.values())).dtype),
        "bitwise_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    weights = Path(args.actor_dir) / "model_state_dict/full_weights.pt"
    result = verify_export(
        weights, Path(args.export_dir), require_manifest=args.require_manifest
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(payload + "\n")


if __name__ == "__main__":
    main()
