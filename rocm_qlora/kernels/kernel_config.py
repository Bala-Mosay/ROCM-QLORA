"""
AMD GPU architecture detection and Triton kernel configuration.
Different AMD GPU families require different tile sizes for peak performance.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import torch

@dataclass
class TritonKernelConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_warps: int
    num_stages: int
    wavefront_size: int  # 32 for RDNA, 64 for CDNA

# gfx942 = MI300X (CDNA3), gfx940/gfx941 = MI300A
# Wavefront=64, native MFMA instructions
CDNA3_CONFIG = TritonKernelConfig(
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    num_warps=4, num_stages=2, wavefront_size=64
)

# CDNA2 (MI250X, MI210) - gfx90a
CDNA2_CONFIG = TritonKernelConfig(
    BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
    num_warps=8, num_stages=2, wavefront_size=64
)

# gfx1100 = RX 7900 (RDNA3)
# Wavefront=32, WMMA instructions
RDNA3_CONFIG = TritonKernelConfig(
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    num_warps=4, num_stages=2, wavefront_size=32
)

# gfx1030 = RX 6000 (RDNA2)
RDNA2_CONFIG = TritonKernelConfig(
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    num_warps=4, num_stages=2, wavefront_size=32
)

# Safe fallback for unknown AMD hardware
GENERIC_AMD_CONFIG = TritonKernelConfig(
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    num_warps=4, num_stages=2, wavefront_size=32
)

def detect_gpu_arch() -> str:
    """
    Detects the AMD GPU architecture by checking the device name.
    Returns: "cdna3", "cdna2", "rdna3", "rdna2", or "unknown"
    """
    if not torch.cuda.is_available():
        return "unknown"
    
    device_name = torch.cuda.get_device_properties(0).name.lower()
    
    # Check for CDNA3 (MI300 series)
    if any(x in device_name for x in ["gfx942", "gfx941", "gfx940", "mi300"]):
        return "cdna3"
    
    # Check for CDNA2 (MI200 series)
    if any(x in device_name for x in ["gfx90a", "mi250", "mi210"]):
        return "cdna2"
    
    # Check for RDNA3 (RX 7000 series)
    if any(x in device_name for x in ["gfx1100", "gfx1101", "gfx1102", "7900", "7800"]):
        return "rdna3"
    
    # Check for RDNA2 (RX 6000 series)
    if any(x in device_name for x in ["gfx1030", "gfx1031", "6900", "6800"]):
        return "rdna2"
    
    return "unknown"

def get_kernel_config() -> TritonKernelConfig:
    """
    Returns the optimal Triton kernel configuration for the detected GPU architecture.
    """
    arch = detect_gpu_arch()
    if arch == "cdna3":
        return CDNA3_CONFIG
    elif arch == "cdna2":
        return CDNA2_CONFIG
    elif arch == "rdna3":
        return RDNA3_CONFIG
    elif arch == "rdna2":
        return RDNA2_CONFIG
    return GENERIC_AMD_CONFIG

def is_triton_available() -> bool:
    """
    Checks if the triton library is installed and available.
    """
    try:
        import triton
        import triton.language as tl
        return True
    except ImportError:
        return False

def get_device_info() -> Dict[str, Any]:
    """
    Returns a dictionary with GPU device information and architectural details.
    """
    if not torch.cuda.is_available():
        return {
            "arch": "unknown",
            "compute_units": 0,
            "vram_gb": 0.0,
            "wavefront_size": 32,
            "triton_available": is_triton_available()
        }
    
    props = torch.cuda.get_device_properties(0)
    arch = detect_gpu_arch()
    config = get_kernel_config()
    
    return {
        "arch": arch,
        "compute_units": getattr(props, "multi_processor_count", 0),
        "vram_gb": props.total_memory / (1024**3),
        "wavefront_size": config.wavefront_size,
        "triton_available": is_triton_available()
    }
