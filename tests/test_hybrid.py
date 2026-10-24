"""Test suite for the GLM-5.3 DSA/SWA hybrid modification (spec section 7).

Functional tests use a tiny random-init config and run on CPU; weight tests need
the sliced mini checkpoint (checkpoints/glm53-mini-2l) and are marked `slow`.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import glm53_hybrid  # noqa: F401  (registers the hybrid model_type)
from glm53_hybrid import GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM, load_hybrid

MINI_CKPT = Path("checkpoints/glm53-mini-2l")

SMALL = dict(
    hidden_size=512,
    num_attention_heads=8,
    # Must be set: with the default 64, num_key_value_groups = 8 // 64 = 0 and the eager
    # path crashes on a 0-head K/V (repeat_kv expands to 0 heads). SDPA happens to skip
    # the repeat and stays correct, but do not rely on that.
    num_key_value_heads=8,
    q_lora_rank=128,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    # 128 keeps a non-zero pass component next to the 32 rope dims, isomorphic to the real
    # model (128 = 64 rope + 64 pass).
    index_head_dim=128,
    index_n_heads=4,
    index_topk=8,
    vocab_size=1024,
    num_hidden_layers=2,
    mlp_layer_types=["dense", "dense"],
    indexer_types=["full", "full"],
    intermediate_size=1024,
    sliding_window=32,
)

slow = pytest.mark.slow
needs_ckpt = pytest.mark.skipif(
    not (MINI_CKPT / "model.safetensors").exists(), reason="mini checkpoint not built"
)


def make_model(pattern, window=32, truncate=False, impl="eager", seed=0, index_topk=8):
    overrides = dict(
        hybrid_pattern=pattern,
        sliding_window=window,
        swa_truncate_cache=truncate,
        index_topk=index_topk,
    )
    cfg = GlmMoeDsaHybridConfig(**{**SMALL, **overrides})
    cfg._attn_implementation = impl
    torch.manual_seed(seed)
    return GlmMoeDsaHybridForCausalLM(cfg).eval()


def copy_shared_weights(dst, src):
    """Copy src's state_dict into dst, allowing dst to lack the SWA indexer keys."""
    dst.load_state_dict(src.state_dict(), strict=False)


# ---------------------------------------------------------------- functional tests


def test_pattern_validation():
    # rule 1: pattern length must match num_hidden_layers
    with pytest.raises(ValueError, match="hybrid_pattern"):
        GlmMoeDsaHybridConfig(**{**SMALL, "hybrid_pattern": ["dsa"]})
    # invalid entry
    with pytest.raises(ValueError, match="hybrid_pattern"):
        GlmMoeDsaHybridConfig(**{**SMALL, "hybrid_pattern": ["dsa", "bogus"]})
    # rule 2: SWA layer cannot be followed by a shared-indexer layer (chain breaks)
    with pytest.raises(ValueError, match="sharing chain"):
        GlmMoeDsaHybridConfig(
            **{**SMALL, "hybrid_pattern": ["swa", "dsa"], "indexer_types": ["full", "shared"],
               "swa_truncate_cache": False}
        )
    # rule 4: truncated SWA at layer 0 breaks cache length bookkeeping
    with pytest.raises(ValueError, match="hybrid_pattern\\[0\\]"):
        GlmMoeDsaHybridConfig(
            **{**SMALL, "hybrid_pattern": ["swa", "dsa"], "swa_truncate_cache": True}
        )


def test_truncate_rejects_chunked_prefill():
    """Rule 5 (runtime): truncate mode must raise on chunked prefill, not silently mis-compute."""
    model = make_model(["dsa", "swa"], window=32, truncate=True)
    with torch.no_grad():
        out = model(torch.randint(0, 1024, (1, 16)), use_cache=True)
        with pytest.raises(NotImplementedError, match="chunked prefill"):
            model(torch.randint(0, 1024, (1, 8)), past_key_values=out.past_key_values, use_cache=True)


def test_swa_equals_full_when_window_covers_seq():
    """With window >= L and L <= index_topk (DSA degenerates to full attention), a DSA layer-1
    and a SWA layer-1 must produce identical outputs on identical weights."""
    seq_len = 8  # == index_topk: the indexer selects every position
    m_dsa = make_model(["dsa", "dsa"], window=32, truncate=False, index_topk=8)
    m_swa = make_model(["dsa", "swa"], window=32, truncate=False, index_topk=8)
    copy_shared_weights(m_swa, m_dsa)
    ids = torch.randint(0, 1024, (1, seq_len))
    with torch.no_grad():
        a = m_dsa(ids).logits
        b = m_swa(ids).logits
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_swa_locality():
    """window=8, L=32: SWA attention weights outside [i-w+1, i] must be exactly zero."""
    w, seq_len = 8, 32
    model = make_model(["dsa", "swa"], window=w, truncate=False)
    ids = torch.randint(0, 1024, (1, seq_len))
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attn = out.attentions[1]  # layer 1 (SWA): [B, H, S, L]
    assert attn is not None and attn.shape[-1] == seq_len
    q = torch.arange(seq_len).unsqueeze(1)
    k = torch.arange(seq_len).unsqueeze(0)
    outside = (k < q - w + 1) | (k > q)
    assert attn[0, :, outside].abs().max().item() == 0.0
    inside = ~(outside | torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1))
    assert attn[0, :, inside].amin().item() > 0.0


def test_cache_truncation():
    """Truncate mode: SWA cache stays <= window through prefill(2x window)+decode; DSA grows."""
    w = 32
    model = make_model(["dsa", "swa"], window=w, truncate=True)
    with torch.no_grad():
        out = model(torch.randint(0, 1024, (1, 2 * w)), use_cache=True)
        past = out.past_key_values
        assert past.layers[1].keys.shape[2] == w
        for i in range(10):
            out = model(torch.randint(0, 1024, (1, 1)), past_key_values=past, use_cache=True)
            past = out.past_key_values
            assert past.layers[1].keys.shape[2] <= w, f"SWA cache exceeded window at decode step {i}"
            assert past.layers[0].keys.shape[2] == 2 * w + i + 1, "DSA cache must grow linearly"


def test_short_seq_no_crash():
    """Corner case: total sequence shorter than the window (window=32, 10 prompt + 5 decode)."""
    model = make_model(["dsa", "swa"], window=32, truncate=True)
    with torch.no_grad():
        out = model(torch.randint(0, 1024, (1, 10)), use_cache=True)
        past = out.past_key_values
        for _ in range(5):
            out = model(torch.randint(0, 1024, (1, 1)), past_key_values=past, use_cache=True)
            past = out.past_key_values
    assert out.logits.shape == (1, 1, SMALL["vocab_size"])
    assert past.layers[1].keys.shape[2] == 15


def test_swa_truncate_decode_equivalence():
    """Key invariant: truncate=True/False must give numerically equal outputs for seq > window.

    NOTE: numerical, not bitwise equivalence. Truncated decode reduces over w columns while the
    untruncated path reduces over the full length with exactly-zero masked terms; the reduction
    tree shape differs, so fp32 ULP-level differences are expected. Hence fp32 + atol=1e-5.
    """
    w, prefill_len, decode_steps = 16, 40, 6
    m_t = make_model(["dsa", "swa"], window=w, truncate=True)
    m_f = make_model(["dsa", "swa"], window=w, truncate=False)
    copy_shared_weights(m_f, m_t)
    ids = torch.randint(0, 1024, (1, prefill_len))
    with torch.no_grad():
        out_t = m_t(ids, use_cache=True)
        out_f = m_f(ids, use_cache=True)
        torch.testing.assert_close(out_t.logits, out_f.logits, rtol=1e-5, atol=1e-5)
        past_t, past_f = out_t.past_key_values, out_f.past_key_values
        for _ in range(decode_steps):
            nxt = torch.randint(0, 1024, (1, 1))
            out_t = m_t(nxt, past_key_values=past_t, use_cache=True)
            out_f = m_f(nxt, past_key_values=past_f, use_cache=True)
            past_t, past_f = out_t.past_key_values, out_f.past_key_values
            torch.testing.assert_close(out_t.logits, out_f.logits, rtol=1e-5, atol=1e-5)
        assert past_t.layers[1].keys.shape[2] == w
        assert past_f.layers[1].keys.shape[2] == prefill_len + decode_steps


# ---------------------------------------------------------------- weight tests (slow)


def build_hybrid_from_mini(pattern, dtype=torch.bfloat16):
    mini_cfg = json.loads((MINI_CKPT / "config.json").read_text())
    cfg = GlmMoeDsaHybridConfig(**mini_cfg, hybrid_pattern=pattern)
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = GlmMoeDsaHybridForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    return model


@slow
@needs_ckpt
def test_slice_loads():
    """The mini checkpoint loads into the original GlmMoeDsaForCausalLM with zero key mismatch."""
    from transformers.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

    cfg = GlmMoeDsaConfig(**json.loads((MINI_CKPT / "config.json").read_text()))
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(cfg)
    sd = load_file(MINI_CKPT / "model.safetensors")
    model.load_state_dict(sd, strict=True, assign=True)  # raises on any missing/unexpected key


@slow
@needs_ckpt
def test_hybrid_weight_report(tmp_path):
    """pattern=[dsa,swa]: loaded = all original keys minus layer-1's 5 indexer params."""
    model = build_hybrid_from_mini(["dsa", "swa"])
    report_path = tmp_path / "weight_report.json"
    report = load_hybrid(model, MINI_CKPT, report_path)

    expected_dropped = {
        f"model.layers.1.self_attn.indexer.{suffix}"
        for suffix in ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias")
    }
    assert set(report["dropped_keys"]) == expected_dropped
    assert report["dropped_by_design"]["keys"] == 5
    assert report["loaded"]["keys"] == 37 - 5
    assert report["missing"] == [] and report["unexpected"] == []
    assert json.loads(report_path.read_text())["loaded"]["keys"] == 32
    # _keep_in_fp32_modules alignment: the remaining (layer 0) indexer weights_proj must be fp32
    assert model.model.layers[0].self_attn.indexer.weights_proj.weight.dtype == torch.float32


@slow
@needs_ckpt
def test_dsa_passthrough():
    """pattern=[dsa,dsa] hybrid model matches the original model on the same weights."""
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(MINI_CKPT, dtype=torch.bfloat16).eval()
    hyb = build_hybrid_from_mini(["dsa", "dsa"]).eval()
    load_hybrid(hyb, MINI_CKPT)
    ids = torch.randint(0, 1024, (1, 16))
    with torch.no_grad():
        a = ref(ids).logits
        b = hyb(ids).logits
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
