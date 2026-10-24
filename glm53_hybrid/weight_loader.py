"""Per-module weight loading for the hybrid DSA/SWA model (spec section 4).

Loads a sliced (dequantized bf16) checkpoint into a hybrid model, classifying
every checkpoint key as `loaded` / `dropped_by_design` (indexer weights of SWA
layers) / `unexpected`, and every model key as loaded / `missing`. Writes a
JSON report grouped by module.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _module_group(key: str) -> str:
    m = _LAYER_RE.match(key)
    if m:
        return f"layer_{m.group(1)}"
    return key.rsplit(".", 1)[0]


def load_hybrid(model, ckpt_dir: str | Path, report_path: str | Path | None = None) -> dict:
    """Load checkpoint weights into `model` (a hybrid model) with a compatibility report.

    Asserts that nothing is unexpectedly missing from the checkpoint and that every
    non-loaded checkpoint key is a by-design dropped SWA indexer weight.
    """
    ckpt_dir = Path(ckpt_dir)
    sd = load_file(ckpt_dir / "model.safetensors")
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    pattern = model.config.hybrid_pattern

    loaded, dropped_by_design, unexpected = [], [], []
    for key in sd:
        if key in model_keys:
            loaded.append(key)
            continue
        m = _LAYER_RE.match(key)
        if m and ".self_attn.indexer." in key and pattern[int(m.group(1))] == "swa":
            dropped_by_design.append(key)
        else:
            unexpected.append(key)
    missing = sorted(model_keys - set(sd.keys()))

    model.load_state_dict({k: sd[k] for k in loaded}, strict=False)

    # `load_state_dict` bypasses `_keep_in_fp32_modules`; align with from_pretrained
    # behavior so long-context indexer scores keep fp32 precision.
    for name, p in model.named_parameters():
        if "indexer.weights_proj" in name:
            p.data = p.data.float()

    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    assert not missing, f"missing checkpoint keys: {missing}"

    def stats(keys, source_sd):
        n = sum(source_sd[k].numel() for k in keys)
        b = sum(source_sd[k].numel() * source_sd[k].element_size() for k in keys)
        return {"keys": len(keys), "params": n, "bytes": b}

    report = {
        "checkpoint": str(ckpt_dir),
        "hybrid_pattern": pattern,
        "loaded": stats(loaded, sd),
        "dropped_by_design": stats(dropped_by_design, sd),
        "dropped_keys": dropped_by_design,
        "missing": missing,
        "unexpected": unexpected,
        "by_module": {},
    }
    for group in sorted({_module_group(k) for k in sd}):
        keys = [k for k in loaded if _module_group(k) == group]
        report["by_module"][group] = stats(keys, sd)

    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return report
