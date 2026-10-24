"""Hybrid DSA/SWA GLM-5.3 package. Import this package before using Auto classes:

    import glm53_hybrid  # registers model_type "glm_moe_dsa_hybrid"
    model = AutoModelForCausalLM.from_pretrained("checkpoints/glm53-mini-2l-hybrid")
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import GlmMoeDsaHybridConfig
from .modeling import GlmMoeDsaForCausalLM as GlmMoeDsaHybridForCausalLM
from .modeling import GlmMoeDsaModel as GlmMoeDsaHybridModel
from .weight_loader import load_hybrid

AutoConfig.register("glm_moe_dsa_hybrid", GlmMoeDsaHybridConfig)
AutoModelForCausalLM.register(GlmMoeDsaHybridConfig, GlmMoeDsaHybridForCausalLM)

__all__ = [
    "GlmMoeDsaHybridConfig",
    "GlmMoeDsaHybridForCausalLM",
    "GlmMoeDsaHybridModel",
    "load_hybrid",
]
