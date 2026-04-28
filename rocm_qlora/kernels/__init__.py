"""
Triton GPU kernels for rocm-qlora.
All kernels have pure-PyTorch fallbacks for CPU testing.
"""
from rocm_qlora.kernels.kernel_config import (
    get_kernel_config, 
    get_device_info, 
    is_triton_available
)
from rocm_qlora.kernels.dequant_matmul import (
    fused_dequant_int8_matmul, 
    fused_dequant_nf4_matmul
)
from rocm_qlora.kernels.nf4_dequant import dequantize_nf4_triton

__all__ = [
    "get_kernel_config", 
    "get_device_info", 
    "is_triton_available",
    "fused_dequant_int8_matmul", 
    "fused_dequant_nf4_matmul",
    "dequantize_nf4_triton",
]
