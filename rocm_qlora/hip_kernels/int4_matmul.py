"""
Python interface for the HIP INT4/INT8 matmul kernels.

Three-level fallback ensures correctness on any hardware:
  1. HIP compiled kernel (MI300X with hipcc)
  2. V2 Triton fused dequant+matmul
  3. V1 pure PyTorch dequantize + F.linear

Public API:
    is_hip_kernel_available() -> bool
    hip_dequant_int4_matmul(x, w_packed, scales, bias, block_size) -> Tensor
    hip_dequant_int8_matmul(x, w_int8, scales, bias, block_size) -> Tensor
    enable_hip_kernels(model) -> int
    benchmark_hip_vs_triton(size, n_runs) -> dict
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Dict
import time
import logging

from rocm_qlora.hip_kernels.build_utils import _ensure_kernel_lib, is_hipcc_available

logger = logging.getLogger(__name__)

# -- Module-level kernel library handle (lazy-loaded) --
_kernel_lib: Optional[object] = None  # None = not tried; False-ish = tried & failed; CDLL = loaded


def _get_kernel_lib() -> Optional[object]:
    """
    Internal: returns the loaded HIP kernel library or None.
    Lazy-loads on first call, caches result for lifetime of process.
    """
    global _kernel_lib
    if _kernel_lib is None:
        _kernel_lib = _ensure_kernel_lib()
    # _ensure_kernel_lib returns None on failure, which becomes False-ish cache
    if _kernel_lib:
        return _kernel_lib
    return None


def is_hip_kernel_available() -> bool:
    """
    Check whether the HIP kernel (compiled .so) is available and loaded.

    Returns True only if:
      - hipcc is available
      - kernel compiled successfully
      - .so loaded via ctypes without errors

    # NOTE: Returns False on non-MI300X hardware — caller should use Triton.
    """
    lib = _get_kernel_lib()
    return lib is not None and is_hipcc_available()


# =========================================================================
# V1 pure PyTorch fallback (always works, no GPU needed)
# =========================================================================

def _v1_int4_fallback(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    V1 fallback: dequantize INT4 packed weights → F.linear.
    Pure PyTorch, works on CPU and GPU.
    """
    from rocm_qlora.quantization.quant_ops import dequantize_int4

    # w_packed can be 1D (from quantize_int4) or 2D (N, K//2).
    # Infer K from x.shape[-1] (the input feature dimension).
    K = x.shape[-1]
    # int4 packing: 2 values per uint8 byte → w_packed.numel() * 2 total elements
    # → N (output dimension) = total_elements / K
    N = w_packed.numel() * 2 // K
    w_fp16 = dequantize_int4(w_packed.flatten(), scales, block_size).reshape(N, K)
    w_fp16 = w_fp16.to(x.dtype)
    if bias is not None:
        bias = bias.to(x.dtype)
    return F.linear(x, w_fp16, bias)


def _v1_int8_fallback(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    V1 fallback: dequantize INT8 weights → F.linear.
    Pure PyTorch, works on CPU and GPU.
    """
    from rocm_qlora.quantization.quant_ops import dequantize_int8

    N, K = w_int8.shape
    w_fp16 = dequantize_int8(w_int8, scales, block_size)
    w_fp16 = w_fp16.to(x.dtype)
    if bias is not None:
        bias = bias.to(x.dtype)
    return F.linear(x, w_fp16, bias)


# =========================================================================
# V2 Triton kernel wrapper (used as fallback level 2)
# =========================================================================

def _triton_int4_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    V2 Triton fused NF4 dequant+matmul.
    """
    from rocm_qlora.kernels import fused_dequant_nf4_matmul, is_triton_available
    if is_triton_available() and x.is_cuda:
        return fused_dequant_nf4_matmul(x, w_packed, scales, bias, block_size)
    # Triton unavailable: drop to V1
    return _v1_int4_fallback(x, w_packed, scales, bias, block_size)


def _triton_int8_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    V2 Triton fused INT8 dequant+matmul.
    """
    from rocm_qlora.kernels import fused_dequant_int8_matmul, is_triton_available
    if is_triton_available() and x.is_cuda:
        return fused_dequant_int8_matmul(x, w_int8, scales, bias, block_size)
    # Triton unavailable: drop to V1
    return _v1_int8_fallback(x, w_int8, scales, bias, block_size)


# =========================================================================
# Level 1: HIP kernel (ctypes dispatch to compiled .so)
# =========================================================================

def _hip_int4_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    HIP INT4 NF4 dequant+matmul via compiled .so.
    Falls back to Triton if HIP kernel fails or is unavailable.
    """
    lib = _get_kernel_lib()
    if lib is None or not x.is_cuda:
        # No HIP kernel loaded or input not on GPU → Triton/V1
        return _triton_int4_matmul(x, w_packed, scales, bias, block_size)

    try:
        # Input validation
        K = x.shape[-1]
        x_2d = x.view(-1, K)
        M, _ = x_2d.shape
        N = w_packed.shape[0]  # N = out_features

        # Ensure all tensors are on same device, fp16 dtype, contiguous
        device = x.device
        x_fp16 = x_2d.contiguous().to(torch.float16)
        w_packed_gpu = w_packed.contiguous().cpu() if not w_packed.is_cuda else w_packed.contiguous()
        scales_gpu = scales.contiguous().cpu() if not scales.is_cuda else scales.contiguous()

        # # NOTE: w_packed must be on GPU for HIP kernel — move if needed
        if not w_packed_gpu.is_cuda:
            w_packed_gpu = w_packed_gpu.to(device)
        if not scales_gpu.is_cuda:
            scales_gpu = scales_gpu.to(device)

        # Allocate output tensor
        output = torch.empty((M, N), device=device, dtype=torch.float16)

        # Get HIP stream from CUDA current stream
        # Use default stream (0 = null stream) for simplicity
        stream = torch.cuda.current_stream().cuda_stream if hasattr(
            torch.cuda.current_stream(), "cuda_stream"
        ) else 0

        # If stream is a cudaStream_t wrapper, extract the raw value
        if hasattr(stream, "value"):
            stream = stream.value
        if stream is None or not isinstance(stream, int):
            stream = 0

        # Call HIP kernel via ctypes
        lib.launch_int4_dequant_matmul(
            stream,
            x_fp16.data_ptr(),          # x
            w_packed_gpu.data_ptr(),     # w_packed
            scales_gpu.data_ptr(),       # scales
            output.data_ptr(),           # c (output)
            M,
            N,
            K,
            block_size,
        )

        # # NOTE: bias is NOT fused in the HIP kernel (yet).
        # Apply bias in PyTorch after HIP kernel returns.
        if bias is not None:
            bias_fp16 = bias.to(torch.float16)
            output = output + bias_fp16

        # Cast to input dtype for mixed precision compatibility
        output = output.to(x.dtype)
        return output.view(*x.shape[:-1], N)

    except Exception as e:
        logger.warning(
            f"HIP INT4 kernel failed: {e}. Falling back to Triton."
        )
        return _triton_int4_matmul(x, w_packed, scales, bias, block_size)


def _hip_int8_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """
    HIP INT8 dequant+matmul via compiled .so.
    Falls back to Triton if HIP kernel fails or is unavailable.
    """
    lib = _get_kernel_lib()
    if lib is None or not x.is_cuda:
        return _triton_int8_matmul(x, w_int8, scales, bias, block_size)

    try:
        K = x.shape[-1]
        x_2d = x.view(-1, K)
        M, _ = x_2d.shape
        N = w_int8.shape[0]  # should match N of weight

        device = x.device
        x_fp16 = x_2d.contiguous().to(torch.float16)
        w_int8_gpu = w_int8.contiguous().cpu() if not w_int8.is_cuda else w_int8.contiguous()
        scales_gpu = scales.contiguous().cpu() if not scales.is_cuda else scales.contiguous()

        if not w_int8_gpu.is_cuda:
            w_int8_gpu = w_int8_gpu.to(device)
        if not scales_gpu.is_cuda:
            scales_gpu = scales_gpu.to(device)

        output = torch.empty((M, N), device=device, dtype=torch.float16)

        stream = 0
        if hasattr(torch.cuda, "current_stream"):
            s = torch.cuda.current_stream()
            if hasattr(s, "cuda_stream"):
                stream = s.cuda_stream
                if hasattr(stream, "value"):
                    stream = stream.value
        if not isinstance(stream, int):
            stream = 0

        lib.launch_int8_dequant_matmul(
            stream,
            x_fp16.data_ptr(),
            w_int8_gpu.data_ptr(),
            scales_gpu.data_ptr(),
            output.data_ptr(),
            M,
            N,
            K,
            block_size,
        )

        if bias is not None:
            output = output + bias.to(torch.float16)

        output = output.to(x.dtype)
        return output.view(*x.shape[:-1], N)

    except Exception as e:
        logger.warning(
            f"HIP INT8 kernel failed: {e}. Falling back to Triton."
        )
        return _triton_int8_matmul(x, w_int8, scales, bias, block_size)


# =========================================================================
# Public API
# =========================================================================

def hip_dequant_int4_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Fused INT4 NF4 dequantization + matrix multiplication.

    Three-level fallback:
      1. HIP compiled kernel (MI300X, CDNA3)
      2. V2 Triton fused_dequant_nf4_matmul
      3. V1 dequantize_int4 + F.linear

    Args:
        x: Input activations [..., K] fp16/bf16.
        w_packed: Packed INT4 weights [N, K//2] uint8 (flattened row-major).
        scales: Quantization scales [num_blocks] fp16.
        bias: Optional bias [N] fp16/bf16.
        block_size: Quantization block size (default 64).

    Returns:
        Output tensor with same shape as input batch + [N].
    """
    # Level 1: Try HIP kernel
    lib = _get_kernel_lib()
    if lib is not None and is_hipcc_available() and x.is_cuda:
        return _hip_int4_matmul(x, w_packed, scales, bias, block_size)

    # Level 2: Triton (handles Level 3 internally)
    return _triton_int4_matmul(x, w_packed, scales, bias, block_size)


def hip_dequant_int8_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Fused INT8 dequantization + matrix multiplication.

    Three-level fallback:
      1. HIP compiled kernel (MI300X, CDNA3)
      2. V2 Triton fused_dequant_int8_matmul
      3. V1 dequantize_int8 + F.linear

    Args:
        x: Input activations [..., K] fp16/bf16.
        w_int8: INT8 quantized weights [N, K] int8 (flattened row-major).
        scales: Quantization scales [num_blocks] fp16.
        bias: Optional bias [N] fp16/bf16.
        block_size: Quantization block size (default 64).

    Returns:
        Output tensor with same shape as input batch + [N].
    """
    lib = _get_kernel_lib()
    if lib is not None and is_hipcc_available() and x.is_cuda:
        return _hip_int8_matmul(x, w_int8, scales, bias, block_size)

    return _triton_int8_matmul(x, w_int8, scales, bias, block_size)


def enable_hip_kernels(model: nn.Module) -> int:
    """
    Walk through model and enable HIP kernel dispatch on all QuantLinear layers.

    Sets QuantLinear.use_hip_kernel = True for layers on supported hardware.
    Falls back silently (use_triton_kernel) if HIP unavailable.

    Args:
        model: nn.Module containing QuantLinear layers.

    Returns:
        Number of QuantLinear layers now using HIP kernels.

    # NOTE: Safe to call on CPU — returns 0, doesn't crash.
    """
    hip_available = is_hip_kernel_available()
    count = 0

    for module in model.modules():
        # Check for QuantLinear by class name to avoid import issues
        class_name = module.__class__.__name__
        if class_name == "QuantLinear":
            # Set the HIP flag on the layer
            # # NOTE: We add a use_hip_kernel attribute that QuantLinear.forward() checks
            module.use_hip_kernel = hip_available

            if hip_available:
                count += 1
                logger.debug(
                    f"HIP kernel enabled for QuantLinear({module.in_features}, {module.out_features})"
                )
            else:
                # Ensure Triton fallback is enabled as safety net
                if hasattr(module, "enable_kernel"):
                    module.enable_kernel()
                logger.debug(
                    f"HIP unavailable — Triton fallback for QuantLinear({module.in_features}, {module.out_features})"
                )

    if hip_available and count > 0:
        logger.info(f"HIP kernels enabled on {count} QuantLinear layers.")
    elif not hip_available:
        logger.info(
            f"HIP kernels not available ({count} QuantLinear layers will use Triton/PyTorch)."
        )

    return count


def benchmark_hip_vs_triton(
    size: tuple = (4096, 4096),
    n_runs: int = 100,
    block_size: int = 64,
) -> Dict[str, float]:
    """
    Benchmark HIP kernel vs Triton kernel on same input.

    Times both paths and returns speedup ratio.
    Only meaningful on actual MI300X hardware.

    Args:
        size: (M, K) for input activation dimensions. Default (4096, 4096).
        n_runs: Number of timed runs per kernel. Default 100.
        block_size: Quantization block size. Default 64.

    Returns:
        Dictionary: {hip_ms, triton_ms, speedup_x, winner, hip_available}

    # NOTE: Returns NaN for hip_ms if HIP unavailable — safe on CPU.
    """
    from rocm_qlora.quantization.quant_ops import quantize_int4, NF4_VALUES
    from rocm_qlora.kernels import fused_dequant_nf4_matmul

    M, K = size
    N = 4096  # typical output feature dim

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    weight = torch.randn(N, K, dtype=torch.float32)  # quantize in FP32 for accuracy

    # Quantize weight → INT4 packed
    w_packed, scales = quantize_int4(weight.flatten(), block_size=block_size)
    w_packed = w_packed.to(device)
    scales = scales.to(device)

    hip_available = is_hip_kernel_available()

    # Warmup
    for _ in range(10):
        if hip_available:
            _ = hip_dequant_int4_matmul(x, w_packed, scales, None, block_size)
        _ = fused_dequant_nf4_matmul(x, w_packed, scales, None, block_size)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark HIP
    hip_ms = float("nan")
    if hip_available:
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = hip_dequant_int4_matmul(x, w_packed, scales, None, block_size)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        hip_ms = (end - start) / n_runs * 1000.0

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fused_dequant_nf4_matmul(x, w_packed, scales, None, block_size)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    triton_ms = (end - start) / n_runs * 1000.0

    speedup = triton_ms / hip_ms if hip_available and hip_ms > 0 else 0.0
    winner = "hip" if speedup > 1.0 else "triton"

    return {
        "hip_ms": hip_ms,
        "triton_ms": triton_ms,
        "speedup_x": round(speedup, 2),
        "winner": winner,
        "hip_available": hip_available,
    }