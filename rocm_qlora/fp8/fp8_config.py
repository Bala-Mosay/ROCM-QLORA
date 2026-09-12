"""
FP8 training configuration and hardware detection for ROCm.
"""
import torch
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from rocm_qlora.quantization.double_quant import _has_fp8, _get_fp8_dtype
from rocm_qlora.kernels import get_device_info

def detect_fp8_support() -> Dict[str, Any]:
    """
    Detects hardware and software support for FP8 training.
    Gated behind ROCm 6.2+ and CDNA3 (MI300X/A) hardware.
    """
    info = {
        "fp8_supported": False,
        "rocm_version": "not ROCm",
        "arch": "unknown",
        "fp8_dtypes_available": _has_fp8(),
        "transformer_engine_available": False,
        "reason": ""
    }
    
    # 1. Check for ROCm and version
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        info["rocm_version"] = torch.version.hip
        try:
            parts = [int(p) for p in info["rocm_version"].split('.')[:2]]
            rocm_ver = tuple(parts) if len(parts) >= 2 else (0, 0)
        except:
            rocm_ver = (0, 0)
    else:
        rocm_ver = (0, 0)

    # 2. Check GPU Architecture
    device_info = get_device_info()
    info["arch"] = device_info.get("arch", "unknown")
    
    # 3. Check TransformerEngine
    try:
        import transformer_engine.pytorch as te
        info["transformer_engine_available"] = True
    except ImportError:
        pass

    # 4. Determine Support and Reason
    if not info["fp8_dtypes_available"]:
        info["reason"] = "FP8 dtypes unavailable in this PyTorch build"
    elif rocm_ver < (6, 2):
        info["reason"] = f"ROCm < 6.2 (found {info['rocm_version']}) — hipBLASLt FP8 requires 6.2+"
    elif info["arch"] not in ("cdna3", "cdna2"):
        info["reason"] = f"Unsupported GPU architecture ({info['arch']}) — MI300X/A required"
    else:
        info["fp8_supported"] = True
        info["reason"] = "Supported"

    return info

@dataclass
class FP8Config:
    """Configuration for FP8 training."""
    enabled: bool = True
    forward_dtype: str = "e4m3"          # E4M3 for weights/activations
    backward_dtype: str = "e5m2"         # E5M2 for gradients
    use_transformer_engine: bool = True  # prefer TE if available
    fallback_dtype: torch.dtype = torch.bfloat16
    amax_history_len: int = 16           # loss scale tracking window
    amax_compute_algo: str = "max"       # "max" or "most_recent"

def get_fp8_dtype(format_str: str) -> torch.dtype:
    """Maps string formats to torch.float8 dtypes.
    Uses AMD-native float8_e4m3fnuz on ROCm, float8_e4m3fn on CUDA.
    """
    if format_str == "e4m3":
        if hasattr(torch, 'float8_e4m3fnuz'):
            return torch.float8_e4m3fnuz  # AMD/ROCm native
        if hasattr(torch, 'float8_e4m3fn'):
            return torch.float8_e4m3fn    # NVIDIA/CUDA native
        raise RuntimeError("No FP8 E4M3 dtype available. Check PyTorch version.")
    elif format_str == "e5m2":
        if hasattr(torch, 'float8_e5m2fnuz'):
            return torch.float8_e5m2fnuz  # AMD/ROCm native
        if hasattr(torch, 'float8_e5m2'):
            return torch.float8_e5m2      # NVIDIA/CUDA native
        raise RuntimeError("No FP8 E5M2 dtype available. Check PyTorch version.")
    else:
        raise ValueError(f"Unknown FP8 format: {format_str}")

def get_fp8_recipe(config: FP8Config):
    """Returns a TransformerEngine recipe if available."""
    try:
        import transformer_engine.pytorch as te
        return te.recipe.DelayedScaling(
            amax_history_len=config.amax_history_len,
            amax_compute_algo=config.amax_compute_algo
        )
    except ImportError:
        return None
