"""
Base-model loader for the EGLR pipeline.

Reserved for train_base / evaluate_base (base SFT baseline) which will be
added later. Not used by the EGLR inference pipeline itself — model_eglr.py
loads its frozen base model directly.
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from config import CACHE_DIR  # noqa: F401  (forces HF cache setup before any HF import)
from utils  import get_logger

logger = get_logger(__name__)


def load_base_model(config):
    kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype),
        "device_map":  config.model.device_map,
        "cache_dir":   config.parameters.cache_dir,
    }
    if "gpt2" not in config.model.name.lower():
        kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    return AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)