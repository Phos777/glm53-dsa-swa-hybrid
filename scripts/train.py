"""Continued-training smoke run for the hybrid model (spec section 5, amended).

Pure-subtraction modification => no new parameters; continued training = small-lr
finetuning. This box has no GPU, so per the spec's CPU clause we default to freezing
embed/lm_head/attention and training MLP only (~0.45B trainable).

Amendment from design review: window AND index_topk are set below block_size so both
sparse paths are actually exercised during training (with the defaults of window=4096 /
index_topk=2048 and block=1024 both degrade to full attention and the loss curves of
"before" and "after" would trivially coincide).

Data: synthetic but learnable — each sequence follows token[t+1] = (token[t] + 1) % 1024
with a random start, so a visible loss decrease is expected even in a short run.

Usage:
    python scripts/train.py --pattern dsa,dsa --tag before
    python scripts/train.py --pattern dsa,swa --tag after
"""

import argparse
import json
import time
from pathlib import Path

import torch

import glm53_hybrid  # noqa: F401
from glm53_hybrid import GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM, load_hybrid

MINI = Path("checkpoints/glm53-mini-2l")

FREEZE_GROUPS = {
    "embed": ["model.embed_tokens"],
    "lm_head": ["lm_head"],
    "attention": ["self_attn"],
    "mlp": ["mlp"],
}


def apply_freeze(model, groups):
    prefixes = [p for g in groups for p in FREEZE_GROUPS[g]]
    n_frozen = n_train = 0
    for name, p in model.named_parameters():
        p.requires_grad = not any(pfx in name for pfx in prefixes)
        if p.requires_grad:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return n_train, n_frozen


def make_batch(batch_size, block_size, vocab=1024):
    start = torch.randint(0, vocab, (batch_size, 1))
    offs = torch.arange(block_size).unsqueeze(0)
    return (start + offs) % vocab  # token[t+1] = (token[t] + 1) % vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="dsa,swa")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--index-topk", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze", default="embed,lm_head,attention")
    args = ap.parse_args()

    pattern = args.pattern.split(",")
    tag = args.tag or args.pattern.replace(",", "")
    assert args.window < args.block_size and args.index_topk < args.block_size, (
        "sparse paths must be active: window and index_topk must be < block_size"
    )

    cfg_dict = json.loads((MINI / "config.json").read_text())
    cfg_dict["index_topk"] = args.index_topk
    cfg = GlmMoeDsaHybridConfig(
        **cfg_dict, hybrid_pattern=pattern, sliding_window=args.window, swa_truncate_cache=True
    )
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = GlmMoeDsaHybridForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    load_hybrid(model, MINI)

    n_train, n_frozen = apply_freeze(model, args.freeze.split(","))
    print(f"trainable: {n_train / 1e6:.1f}M | frozen: {n_frozen / 1e6:.1f}M | pattern={pattern} "
          f"window={args.window} index_topk={args.index_topk} block={args.block_size} lr={args.lr}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    torch.manual_seed(42)  # identical data stream across patterns

    model.train()
    log = {"config": vars(args), "trainable_params": n_train, "steps": []}
    for step in range(args.steps):
        ids = make_batch(args.batch_size, args.block_size)
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        log["steps"].append({"step": step, "loss": round(out.loss.item(), 6), "sec": round(dt, 2)})
        print(f"step {step:3d} loss {out.loss.item():.4f} ({dt:.1f}s)")

    out_path = Path(f"results/train_log_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
