"""Benchmarks for the hybrid DSA/SWA modification (spec section 6).

Three measurements, all CPU-friendly:
  1. KV-cache memory: synthetic tensors fed through DynamicCache.update/update_indexer
     (never a full forward — the DSA indexer score matrix [B, S, H, T] in fp32 would be
     ~137GB at 32k), with the SWA truncation applied exactly as the modeling code does.
  2. Analytical FLOPs per token per layer for DSA vs SWA.
  3. End-to-end prefill/decode latency of the real 2-layer mini model, eager, CPU.

Usage:
    python scripts/bench.py --out results/bench.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import glm53_hybrid  # noqa: F401
from glm53_hybrid import GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM, load_hybrid

MINI = Path("checkpoints/glm53-mini-2l")

LENGTHS = [1024, 4096, 8192, 16384, 32768]
WINDOW = 4096


def cache_bytes(pattern, seq_len, truncate, window=WINDOW):
    """Feed synthetic K/V (+indexer k) through a real DynamicCache and measure bytes."""
    cfg_dict = json.loads((MINI / "config.json").read_text())
    cfg = GlmMoeDsaHybridConfig(
        **cfg_dict, hybrid_pattern=pattern, sliding_window=window, swa_truncate_cache=truncate
    )
    cache = DynamicCache(config=cfg)
    h, d_k, d_v, d_idx = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.index_head_dim
    for i, mode in enumerate(pattern):
        k = torch.zeros(1, h, seq_len, d_k, dtype=torch.bfloat16)
        v = torch.zeros(1, h, seq_len, d_v, dtype=torch.bfloat16)
        cache.update(k, v, i)
        layer = cache.layers[i]
        if mode == "dsa":
            cache.update_indexer(torch.zeros(1, seq_len, d_idx, dtype=torch.bfloat16), i)
        elif truncate:
            # Mirror the modeling code's SWA truncation (M5).
            layer.keys = layer.keys[:, :, -window:, :]
            layer.values = layer.values[:, :, -window:, :]
    total = 0
    per_layer = []
    for layer in cache.layers:
        n = layer.keys.numel() * 2 + layer.values.numel() * 2
        idx = layer.indexer_keys.numel() * 2 if getattr(layer, "indexer_keys", None) is not None else 0
        per_layer.append({"main_kv_bytes": n, "indexer_bytes": idx})
        total += n + idx
    return {"total_bytes": total, "total_gib": round(total / 2**30, 4), "per_layer": per_layer}


def flops_per_token(cfg_dict, seq_len, window=WINDOW):
    """Analytical FLOPs per token per layer (attention-related terms only)."""
    H = cfg_dict["num_attention_heads"]  # 64
    d = cfg_dict["qk_nope_head_dim"] + cfg_dict["qk_rope_head_dim"] + cfg_dict["v_head_dim"]  # 512
    topk = min(cfg_dict["index_topk"], seq_len)
    hi, di = cfg_dict["index_n_heads"], cfg_dict["index_head_dim"]  # 32, 128
    hidden, q_rank = cfg_dict["hidden_size"], cfg_dict["q_lora_rank"]

    dsa_main = 2 * H * d * topk
    indexer_score = 2 * hi * di * seq_len + 2 * hi * seq_len
    indexer_proj = 2 * q_rank * (hi * di) + 2 * hidden * di + 2 * hidden * hi
    swa = 2 * H * d * min(window, seq_len)
    return {
        "dsa_main_attn": dsa_main,
        "dsa_indexer_score": indexer_score,
        "dsa_indexer_proj": indexer_proj,
        "dsa_total": dsa_main + indexer_score + indexer_proj,
        "swa_total": swa,
        "swa_window_effective": min(window, seq_len),
        "dsa_topk_effective": topk,
    }


@torch.no_grad()
def latency(pattern, window, truncate, prefill=256, decode=16, repeats=3):
    cfg_dict = json.loads((MINI / "config.json").read_text())
    cfg = GlmMoeDsaHybridConfig(
        **cfg_dict, hybrid_pattern=pattern, sliding_window=window, swa_truncate_cache=truncate
    )
    cfg._attn_implementation = "eager"
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = GlmMoeDsaHybridForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    load_hybrid(model, MINI)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, prefill))
    out = model(ids, use_cache=True)  # warmup: allocator / thread-pool / dispatch
    past = out.past_key_values
    for _ in range(2):
        out = model(torch.randint(0, cfg.vocab_size, (1, 1)), past_key_values=past, use_cache=True)
        past = out.past_key_values

    prefill_ts, decode_ts = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        prefill_ts.append(time.perf_counter() - t0)
        past = out.past_key_values
        t0 = time.perf_counter()
        for _ in range(decode):
            out = model(torch.randint(0, cfg.vocab_size, (1, 1)), past_key_values=past, use_cache=True)
            past = out.past_key_values
        decode_ts.append((time.perf_counter() - t0) / decode)
    del model
    return {
        "prefill_s_min": round(min(prefill_ts), 3),
        "decode_ms_per_tok_min": round(min(decode_ts) * 1e3, 1),
        "prefill_s_all": [round(t, 3) for t in prefill_ts],
        "decode_ms_all": [round(t * 1e3, 1) for t in decode_ts],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/bench.json"))
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    cfg_dict = json.loads((MINI / "config.json").read_text())
    results = {"cache_memory": {}, "flops": {}, "latency": {}}

    for pattern_name, pattern in [("dsa,dsa", ["dsa", "dsa"]), ("dsa,swa", ["dsa", "swa"])]:
        for truncate in ([False, True] if "swa" in pattern else [False]):
            key = f"{pattern_name},truncate={truncate}"
            results["cache_memory"][key] = {
                str(L): cache_bytes(pattern, L, truncate) for L in LENGTHS
            }
            print(f"cache {key}: "
                  + " ".join(f"{L // 1024}k={results['cache_memory'][key][str(L)]['total_gib']}GiB" for L in LENGTHS))

    for L in LENGTHS:
        results["flops"][str(L)] = flops_per_token(cfg_dict, L)
        f = results["flops"][str(L)]
        print(f"FLOPs/token/layer @ {L}: dsa={f['dsa_total'] / 1e6:.1f}M swa={f['swa_total'] / 1e6:.1f}M "
              f"(dsa topk={f['dsa_topk_effective']}, swa window={f['swa_window_effective']})")

    if not args.skip_latency:
        # window 128 < prefill 256 so SWA truncation is actually exercised
        results["latency"]["dsa,dsa"] = latency(["dsa", "dsa"], WINDOW, False)
        results["latency"]["dsa,swa,w=128"] = latency(["dsa", "swa"], 128, True)
        print("latency:", results["latency"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
