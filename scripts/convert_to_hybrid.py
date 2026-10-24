"""Convert the sliced mini checkpoint into a hybrid DSA/SWA model (spec section 4).

Generates the hybrid config, materializes the hybrid checkpoint directory, then loads
weights per-module via `load_hybrid` and writes results/weight_report.json.

Usage:
    python scripts/convert_to_hybrid.py --pattern dsa,swa --window 4096
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

import glm53_hybrid  # noqa: F401  (registers the hybrid model_type)
from glm53_hybrid import GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM, load_hybrid

MINI = Path("checkpoints/glm53-mini-2l")
HYBRID = Path("checkpoints/glm53-mini-2l-hybrid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="dsa,swa", help="comma-separated per-layer modes")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--no-truncate", action="store_true", help="mask only, do not truncate SWA caches")
    ap.add_argument("--src", type=Path, default=MINI)
    ap.add_argument("--dst", type=Path, default=HYBRID)
    ap.add_argument("--report", type=Path, default=Path("results/weight_report.json"))
    args = ap.parse_args()

    pattern = args.pattern.split(",")
    cfg_dict = json.loads((args.src / "config.json").read_text())
    assert len(pattern) == cfg_dict["num_hidden_layers"]
    cfg_dict.update(
        model_type="glm_moe_dsa_hybrid",
        architectures=["GlmMoeDsaHybridForCausalLM"],
        hybrid_pattern=pattern,
        sliding_window=args.window,
        swa_truncate_cache=not args.no_truncate,
    )

    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(args.src / name, args.dst / name)
    # Weights are identical to the sliced checkpoint; link rather than duplicate 5.4GB.
    link = args.dst / "model.safetensors"
    if not link.exists():
        link.symlink_to((args.src / "model.safetensors").resolve())

    # Per-module weight load + compatibility report.
    cfg = GlmMoeDsaHybridConfig(**cfg_dict)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = GlmMoeDsaHybridForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = load_hybrid(model, args.src, args.report)

    print(f"hybrid pattern: {pattern}, window: {args.window}, truncate: {not args.no_truncate}")
    print(f"loaded: {report['loaded']} | dropped_by_design: {report['dropped_by_design']}")
    print(f"dropped keys: {report['dropped_keys']}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
