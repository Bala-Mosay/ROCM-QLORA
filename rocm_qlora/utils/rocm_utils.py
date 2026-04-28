"""
ROCm utility functions for hardware detection and memory management.
"""

import torch
import torch.nn as nn
from typing import Dict

def check_rocm() -> Dict:
    """
    Detects ROCm environment and returns structured information.
    
    Returns:
        dict: A dictionary containing 'available', 'version', and 'device_name'.
    """
    # NOTE: PyTorch ROCm builds use the 'cuda' namespace.
    # We check torch.version.hip to distinguish from NVIDIA CUDA.
    is_rocm = torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None
    
    return {
        "available": is_rocm,
        "version": getattr(torch.version, "hip", "N/A"),
        "device_name": torch.cuda.get_device_name(0) if is_rocm else "N/A"
    }

def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """
    Returns memory allocation and reservation statistics in GB.
    
    Args:
        device (str): Device string, always "cuda" for ROCm/CUDA in PyTorch.
        
    Returns:
        dict: Dictionary with 'allocated_gb', 'reserved_gb', and 'free_gb'.
    """
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    
    # Estimate free memory based on total device memory
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free = total_mem - reserved
    
    return {
        "allocated_gb": round(allocated, 4),
        "reserved_gb": round(reserved, 4),
        "free_gb": round(free, 4)
    }

def estimate_model_memory(model: nn.Module, bits: int) -> Dict[str, float]:
    """
    Estimates the memory usage of a model in both FP16 and quantized states.
    
    Args:
        model: The PyTorch model to estimate.
        bits: The target quantization bit-depth (e.g., 4 or 8).
        
    Returns:
        dict: Memory estimates and reduction percentage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Standard FP16/BF16 is 2 bytes per param
    fp16_gb = (total_params * 2) / (1024**3)
    
    # Quantized size: (params * bits / 8) + scales
    # Scales are typically per block (e.g., 64). 
    # For INT8: 1 byte per param + 4 bytes per 64 params (FP32 scale)
    # For INT4: 0.5 bytes per param + 4 bytes per 64 params
    bytes_per_param = bits / 8.0
    scale_overhead = 4 / 64 # Rough estimate: 4-byte float scale for every 64 params
    
    quantized_gb = (total_params * (bytes_per_param + scale_overhead)) / (1024**3)
    reduction_pct = (1 - (quantized_gb / fp16_gb)) * 100
    
    return {
        "fp16_gb": round(fp16_gb, 4),
        "quantized_gb": round(quantized_gb, 4),
        "reduction_pct": round(reduction_pct, 2)
    }

def print_model_summary(model: nn.Module) -> None:
    """
    Prints a formatted table summary of the model layers and parameters.
    
    Args:
        model: The PyTorch model to summarize.
    """
    print(f"{'Layer Name':<40} | {'Type':<20} | {'Params':<12} | {'Bits':<6} | {'Trainable':<10}")
    print("-" * 100)
    
    for name, module in model.named_modules():
        # Only print leaf modules with parameters
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            params = sum(p.numel() for p in module.parameters())
            trainable = any(p.requires_grad for p in module.parameters())
            
            # Determine bits based on dtype (rough estimate)
            first_p = next(module.parameters())
            bit_depth = 32 if first_p.dtype == torch.float32 else 16
            
            print(f"{name[:40]:<40} | {type(module).__name__:<20} | {params:<12,} | {bit_depth:<6} | {str(trainable):<10}")
