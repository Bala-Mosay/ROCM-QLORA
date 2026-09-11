"""
V5 HIP kernels for rocm-qlora.

Hand-written HIP C++ kernels using MFMA instructions for peak INT4/INT8
matmul performance on AMD MI300X (gfx942, CDNA3).

All kernels have three-level fallback:
  1. HIP compiled kernel (MI300X only)
  2. V2 Triton fused dequant+matmul
  3. V1 pure PyTorch dequantize + F.linear

Public API:
    from rocm_qlora.hip_kernels import (
        is_hip_kernel_available,
        enable_hip_kernels,
        hip_dequant_int4_matmul,
        hip_dequant_int8_matmul,
        benchmark_hip_vs_triton,
    )
"""

from rocm_qlora.hip_kernels.int4_matmul import (
    is_hip_kernel_available,
    enable_hip_kernels,
    hip_dequant_int4_matmul,
    hip_dequant_int8_matmul,
    benchmark_hip_vs_triton,
)

__all__ = [
    "is_hip_kernel_available",
    "enable_hip_kernels",
    "hip_dequant_int4_matmul",
    "hip_dequant_int8_matmul",
    "benchmark_hip_vs_triton",
]