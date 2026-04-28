"""
Flash Attention 2 integration for ROCm with graceful fallback chain.

ROCm Flash Attention has two backends and a critical training constraint:

  Backend 1 — Composable Kernel (CK):
    Default. Supports MI200x, MI250x, MI300x, MI355x, RDNA3, RDNA4.
    RDNA3 CK: forward only — NO BACKWARD PASS. Using for training
    silently produces WRONG gradients. This is a hard safety issue.

  Backend 2 — Triton:
    Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE.
    Supports all AMD CDNA and RDNA GPUs.
    Has backward pass on all supported hardware.
    Recommended for training on RDNA3.

  Backend 3 — torch SDPA:
    Always available. No install required. 20-40% slower than FA2.
    Fully correct for both forward and backward on all hardware.
    Safe fallback of last resort.

Fallback chain (training-safe):
  FA2 Triton (if env var set) → FA2 CK (CDNA only) → torch SDPA

This module NEVER enables FA2 CK for RDNA3 training — gradient
correctness is non-negotiable.
"""
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, Optional
import warnings

from rocm_qlora.kernels.kernel_config import get_device_info, is_triton_available

logger = logging.getLogger("rocm_qlora.attention")

def detect_flash_attention() -> Dict[str, Any]:
    """
    Detects available Flash Attention backends and checks for training safety.
    """
    try:
        import flash_attn
        available = True
        version = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        return {
            "available": False, 
            "version": "none", 
            "backend": "none", 
            "supports_backward": False,
            "arch": "unknown",
            "triton_env_set": False
        }

    triton_env_set = os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE").upper() == "TRUE"
    
    # Get architecture and Triton status
    dev_info = get_device_info()
    arch = dev_info["arch"]
    triton_ready = dev_info["triton_available"]

    backend = "ck" # default
    supports_backward = False

    if triton_env_set:
        backend = "triton"
        supports_backward = True # Triton FA2 has backward on all AMD
    elif arch in ("cdna3", "cdna2"):
        backend = "ck"
        supports_backward = True # CK backward works on CDNA
    elif arch in ("rdna3", "rdna4"):
        backend = "ck"
        supports_backward = False # CK has NO backward on RDNA3/4
    else:
        backend = "ck"
        supports_backward = False # unknown — assume unsafe

    return {
        "available": available,
        "version": version,
        "backend": backend,
        "supports_backward": supports_backward,
        "arch": arch,
        "triton_env_set": triton_env_set
    }
    # NOTE: RDNA3 CK backward is silently wrong — not a crash, a silent correctness bug. 
    # Always check supports_backward before training.

def patch_model_attention(model: nn.Module, verbose: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Applies the training-safe Flash Attention fallback chain.
    Returns: (model, patch_info)
    """
    fa_info = detect_flash_attention()
    
    warning = None
    install_hint = None
    layers_patched = 0
    
    if fa_info['available'] and fa_info['supports_backward']:
        # Safe to use FA2 for training
        model, layers_patched = _apply_flash_attention(model, fa_info['backend'])
        chosen = f"flash_attention_2 ({fa_info['backend']})"
    elif fa_info['available'] and not fa_info['supports_backward']:
        # FA2 available but RDNA3 CK — unsafe for training
        warning = (
            f"FA2 CK on {fa_info['arch']} has no backward pass. "
            f"Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE for FA2, "
            f"or falling back to torch SDPA."
        )
        model = enable_sdpa_optimization(model)
        chosen = "torch_sdpa (FA2 CK backward unsafe)"
        install_hint = "Set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
    else:
        # FA2 not installed
        model = enable_sdpa_optimization(model)
        chosen = "torch_sdpa (FA2 not installed)"
        install_hint = "Install flash-attention from source"

    patch_info = {
        "backend_chosen": chosen,
        "layers_patched": layers_patched,
        "warning": warning,
        "install_hint": install_hint
    }
    
    return model, patch_info

def _apply_flash_attention(model: nn.Module, backend: str) -> Tuple[nn.Module, int]:
    """
    Tries to set model.config._attn_implementation = "flash_attention_2".
    """
    layers_patched = 0
    if hasattr(model, "config"):
        # HuggingFace models respect this key
        model.config._attn_implementation = "flash_attention_2"
        # Count layers (heuristic)
        layers_patched = getattr(model.config, "num_hidden_layers", 0)
    else:
        logger.warning("Model has no config. Manual patching might be needed for non-HuggingFace models.")
        
    return model, layers_patched

def enable_sdpa_optimization(model: nn.Module) -> nn.Module:
    """
    Enables torch SDPA (Scaled Dot Product Attention) optimization flags.
    These are global flags affecting all torch.nn.functional.scaled_dot_product_attention calls.
    """
    # NOTE: these are global flags, not per-model
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True) # Ensure fallback to math if others fail
    
    return model

def install_instructions(arch: Optional[str] = None) -> str:
    """
    Returns exact install commands for the detected or specified architecture.
    """
    if arch is None:
        dev_info = get_device_info()
        arch = dev_info["arch"]
        
    instructions = [
        "To install Flash Attention 2 for ROCm:",
        "",
        "1. Install dependencies:",
        "pip install ninja",
        "",
        "2. Build from source (required for ROCm support):",
        "git clone https://github.com/Dao-AILab/flash-attention.git",
        "cd flash-attention"
    ]
    
    if arch in ("cdna3", "cdna2"):
        instructions.append("python setup.py install")
        instructions.append("\n# CK backend is default for CDNA.")
    elif arch in ("rdna3", "rdna4"):
        instructions.append("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python setup.py install")
        instructions.append("\n# For RDNA3/4, you MUST set this env var for backward pass support:")
        instructions.append("export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE")
    else:
        instructions.append("python setup.py install")
        instructions.append("\n# Architecture unknown. CK is default. Check compatibility.")
        
    return "\n".join(instructions)
