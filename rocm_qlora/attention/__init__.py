"""
Flash Attention 2 integration for rocm-qlora.
Handles ROCm backend detection (CK vs Triton) and training-safe fallback.
"""
from rocm_qlora.attention.attention_patch import (
    detect_flash_attention,
    patch_model_attention,
    enable_sdpa_optimization,
    install_instructions,
)

__all__ = [
    "detect_flash_attention",
    "patch_model_attention",
    "enable_sdpa_optimization",
    "install_instructions",
]
