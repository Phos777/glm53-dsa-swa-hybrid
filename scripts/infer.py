"""Inference smoke test: generate from the hybrid mini model and the original mini model.

Weights are real (FP8->bf16 dequantized) but model.norm.weight is re-initialized to ones,
so the text is not expected to be coherent — the goal is an end-to-end runnable pipeline.

Usage:
    python scripts/infer.py --prompt "Hello" --max-new-tokens 16
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import glm53_hybrid  # noqa: F401
from glm53_hybrid import GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM, load_hybrid

MINI = Path("checkpoints/glm53-mini-2l")


def load_hybrid_model(pattern, window, truncate):
    cfg_dict = json.loads((MINI / "config.json").read_text())
    cfg = GlmMoeDsaHybridConfig(
        **cfg_dict, hybrid_pattern=pattern, sliding_window=window, swa_truncate_cache=truncate
    )
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = GlmMoeDsaHybridForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    load_hybrid(model, MINI)
    return model.eval()


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.perf_counter() - t0
    return tok.decode(out[0], skip_special_tokens=True), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--window", type=int, default=4096)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MINI)

    print("=== original mini (GlmMoeDsaForCausalLM, 全 DSA) ===")
    t0 = time.perf_counter()
    ref = AutoModelForCausalLM.from_pretrained(MINI, dtype=torch.bfloat16).eval()
    print(f"load: {time.perf_counter() - t0:.1f}s")
    text, dt = generate(ref, tok, args.prompt, args.max_new_tokens)
    print(f"[{dt:.1f}s] {text!r}")
    del ref

    print("=== hybrid mini (pattern dsa,swa) ===")
    hyb = load_hybrid_model(["dsa", "swa"], args.window, truncate=True)
    text, dt = generate(hyb, tok, args.prompt, args.max_new_tokens)
    print(f"[{dt:.1f}s] {text!r}")


if __name__ == "__main__":
    main()
