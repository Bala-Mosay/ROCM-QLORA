"""
FP8 training support for rocm-qlora.

Requires ROCm >= 6.2 and MI300X/MI300A (gfx942) for actual FP8 compute.
Falls back to BF16 silently on all other hardware.
"""
from rocm_qlora.fp8.fp8_config import (
    detect_fp8_support, FP8Config, get_fp8_dtype, get_fp8_recipe,
)
from rocm_qlora.fp8.fp8_linear import (
    FP8LinearWrapper, wrap_model_for_fp8, enable_fp8_autocast,
)

__all__ = [
    "detect_fp8_support", "FP8Config", "get_fp8_dtype", "get_fp8_recipe",
    "FP8LinearWrapper", "wrap_model_for_fp8", "enable_fp8_autocast",
]
