"""Slice GLM-5.3 checkpoint shard 00001 into a 2-layer bf16 mini checkpoint.

Implements spec §3: parse index.json, collect embed_tokens / lm_head / layer 0-1
keys, dequantize FP8 block-wise (128x128, weight_scale_inv) to bf16, re-init
model.norm.weight to ones (the final norm lives in shard 00141 which we do not
have), and reconcile the output key set against the original 2-layer model's
state_dict keys.

Usage:
    python scripts/slice_checkpoint.py \
        --src model --dst checkpoints/glm53-mini-2l
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_BLOCK = 128

TARGET_PREFIXES = ("model.layers.0.", "model.layers.1.")
TARGET_EXACT = ("model.embed_tokens.weight", "lm_head.weight")
EXPECTED_KEYS_PER_LAYER = 27


def dequant_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 block-wise dequant: weight [out,in], scale_inv [ceil(out/128), ceil(in/128)]."""
    w = weight.to(torch.float32)
    s = scale_inv.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return (w * s[: w.shape[0], : w.shape[1]]).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("model"))
    ap.add_argument("--dst", type=Path, default=Path("checkpoints/glm53-mini-2l"))
    args = ap.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((args.src / "config.json").read_text())

    target_keys = sorted(
        k for k in weight_map if k in TARGET_EXACT or k.startswith(TARGET_PREFIXES)
    )
    for layer in (0, 1):
        n = sum(k.startswith(f"model.layers.{layer}.") for k in target_keys)
        assert n == EXPECTED_KEYS_PER_LAYER, f"layer {layer}: expected {EXPECTED_KEYS_PER_LAYER} keys, got {n}"

    shards = {weight_map[k] for k in target_keys}
    assert shards == {"model-00001-of-00141.safetensors"}, f"target keys span unexpected shards: {shards}"
    assert weight_map["model.norm.weight"] == "model-00141-of-00141.safetensors"

    tensors: dict[str, torch.Tensor] = {}
    n_dequant = 0
    with safe_open(args.src / "model-00001-of-00141.safetensors", framework="pt") as f:
        available = set(f.keys())
        missing = [k for k in target_keys if k not in available]
        assert not missing, f"keys missing from shard 00001: {missing}"
        for k in target_keys:
            if k.endswith(".weight_scale_inv"):
                continue
            t = f.get_tensor(k)
            scale_key = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
            # A tensor is FP8 iff its weight_scale_inv companion key exists; we do
            # not trust config's modules_to_not_convert (it lists stale names such
            # as `self_attn.indexers_proj` that match no checkpoint key).
            if scale_key in available:
                tensors[k] = dequant_fp8_blockwise(t, f.get_tensor(scale_key))
                n_dequant += 1
            else:
                tensors[k] = t.to(torch.bfloat16)

    hidden = config["hidden_size"]
    norm_key = "model.norm.weight"
    tensors[norm_key] = torch.ones(hidden, dtype=torch.bfloat16)
    print(f"WARNING: {norm_key} not in shard 00001 (lives in shard 00141); re-initialized to ones (RMSNorm default)")

    # Mini config per spec §3 step 4
    mini_config = dict(config)
    mini_config["num_hidden_layers"] = 2
    mini_config["mlp_layer_types"] = ["dense", "dense"]
    mini_config["indexer_types"] = ["full", "full"]
    mini_config.pop("quantization_config", None)
    mini_config["dtype"] = "bfloat16"
    # architectures stays GlmMoeDsaForCausalLM: slice is validated with the original class.

    # Reconciliation (spec §3 step 6): output keys must equal the original 2-layer
    # model's state_dict keys minus nothing (norm re-inited above).
    from transformers.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    mcfg = GlmMoeDsaConfig(**mini_config)
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(mcfg)
    expected = set(model.state_dict().keys())
    got = set(tensors.keys())
    assert got == expected, f"key mismatch: missing={sorted(expected - got)}, unexpected={sorted(got - expected)}"

    args.dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.dst / "model.safetensors", metadata={"format": "pt"})
    (args.dst / "config.json").write_text(json.dumps(mini_config, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(args.src / name, args.dst / name)

    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    report = {
        "source_shard": "model-00001-of-00141.safetensors",
        "keys_written": len(tensors),
        "keys_dequantized": n_dequant,
        "keys_reinitialized": [norm_key],
        "reconciliation": "passed: output keys == original 2-layer state_dict keys",
        "total_gib": round(total_bytes / 2**30, 3),
    }
    (args.dst / "slice_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
