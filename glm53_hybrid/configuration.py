# Forked from transformers 5.16.1 transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py
# with hybrid DSA/SWA attention additions (spec section 2.1, change C1).
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="zai-org/GLM-5")
@strict
class GlmMoeDsaHybridConfig(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full
        layer's top-k). Defaults to the pattern derived from `index_topk_freq` /
        `index_skip_topk_offset` (or `index_topk_pattern`).
    sliding_window (`int`, *optional*, defaults to 4096):
        Sliding window size used by layers whose `hybrid_pattern` entry is `"swa"`.
    hybrid_pattern (`list[str]`, *optional*):
        Per-layer attention mode: `"dsa"` keeps the original indexer path, `"swa"` drops the
        indexer and uses a static sliding window. `None` (default) means all `"dsa"`, which is
        bit-identical to the original model.
    swa_truncate_cache (`bool`, *optional*, defaults to True):
        Whether SWA layers truncate their KV cache to `sliding_window` tokens. `False` keeps the
        full cache and only applies the window mask (used for equivalence testing). Truncation
        mode does not support chunked prefill and requires `hybrid_pattern[0] == "dsa"`.

    ```python
    >>> from glm53_hybrid import GlmMoeDsaHybridConfig

    >>> # Initializing a GLM-MoE-DSA configuration
    >>> configuration = GlmMoeDsaHybridConfig()

    >>> # Initializing a model from the configuration
    >>> model = GlmMoeDsaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_moe_dsa_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 202752
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    mlp_bias: bool = False
    head_dim: int = 64
    first_k_dense_replace: int = 3
    layer_types: list[str] | None = None
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    # Hybrid DSA/SWA attention (change C1). `hybrid_pattern[i]` selects the attention mode of
    # layer i: "dsa" keeps the original indexer path, "swa" drops the indexer and attends to a
    # static sliding window through the same scatter->mask path. None means all-"dsa", which is
    # bit-identical to the original model.
    sliding_window: int = 4096
    hybrid_pattern: list[str] | None = None
    # True: SWA layers truncate their KV cache to the window (memory savings).
    # False: mask only, untruncated cache (used by equivalence tests).
    swa_truncate_cache: bool = True

    def __post_init__(self, **kwargs):
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the inherited (Llama) rotary
        # embedding reads `config.head_dim` and then computes the right frequencies with no override needed.
        self.head_dim = self.qk_rope_head_dim
        # MLP layer types: the first `first_k_dense_replace` layers are dense, the rest are MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        # Every layer is DSA — drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        # BC: re-route `num_experts` to `n_routed_experts`
        if (num_experts := kwargs.get("num_experts")) is not None:
            self.n_routed_experts = num_experts
        # Default to MoE from the second layer and on
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Hybrid pattern validation (spec section 1.1). Runs after `indexer_types` is resolved above.
        if self.hybrid_pattern is None:
            self.hybrid_pattern = ["dsa"] * self.num_hidden_layers
        if len(self.hybrid_pattern) != self.num_hidden_layers:
            raise ValueError(
                f"len(hybrid_pattern)={len(self.hybrid_pattern)} != num_hidden_layers={self.num_hidden_layers}"
            )
        if any(p not in ("dsa", "swa") for p in self.hybrid_pattern):
            raise ValueError(f"hybrid_pattern entries must be 'dsa' or 'swa', got {self.hybrid_pattern}")
        for i, p in enumerate(self.hybrid_pattern[:-1]):
            if p == "swa" and self.indexer_types[i + 1] == "shared":
                raise ValueError(
                    f"layer {i} is SWA, so layer {i + 1} cannot be a shared-indexer layer: SWA layers "
                    "do not propagate top-k indices and the sharing chain would break"
                )
        # Cache length bookkeeping (Cache.get_seq_length, position_ids fallback, generate) relies on
        # layer 0; a truncated layer 0 would rewind the perceived sequence length and corrupt RoPE.
        if self.swa_truncate_cache and self.hybrid_pattern[0] != "dsa":
            raise ValueError("swa_truncate_cache=True requires hybrid_pattern[0] == 'dsa'")
        super().__post_init__(**kwargs)


__all__ = ["GlmMoeDsaHybridConfig"]
