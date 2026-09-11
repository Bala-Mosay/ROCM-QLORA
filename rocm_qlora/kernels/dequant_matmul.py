"""
Fused Triton kernels for dequantization and matrix multiplication.
These kernels eliminate the need to materialize full FP16 weights in VRAM.
"""
import torch
import torch.nn.functional as F
from typing import Optional

try:
    import triton
    import triton.language as tl
    from rocm_qlora.kernels.kernel_config import get_kernel_config, is_triton_available
    TRITON_INSTALLED = True
except ImportError:
    TRITON_INSTALLED = False
    # Dummy objects for CPU-only environments
    class Dummy:
        def __getattr__(self, name): return lambda x: x
        @property
        def constexpr(self): return int
    triton = Dummy()
    triton.jit = lambda x: x
    tl = Dummy()

from rocm_qlora.quantization.quant_ops import dequantize_int8, dequantize_int4

# --- Triton Kernels ---

@triton.jit
def dequant_int8_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, scales_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Quantization params
    block_size,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Fused INT8 dequantization and matmul.
    B (weights) is [N, K] row-major.
    Scales is [ (N*K) // block_size ]
    """
    # Map program ID to row/col of C
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = tl.ravel_index(pid, (num_pid_m, num_pid_n))

    # Compute offsets for tiles
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to current tiles
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A tile
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        
        # Load B tile (INT8)
        b_int8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        
        # Calculate scale indices for each element in the B tile
        # scale_idx = (global_row * K + global_col) // block_size
        # global_row = offs_bn[None, :]
        # global_col = k * BLOCK_K + offs_k[:, None]
        global_row = offs_bn[None, :]
        global_col = k * BLOCK_K + offs_k[:, None]
        scale_idx = (global_row * K + global_col) // block_size
        
        # Load scales for the B tile
        scales = tl.load(scales_ptr + scale_idx)
        
        # Dequantize B
        b_fp16 = b_int8.to(tl.float16) * scales.to(tl.float16)
        
        # Matrix multiply (tl.dot expects [M, K] and [K, N])
        # b_fp16 is [BLOCK_K, BLOCK_N] (since b_ptrs was [BLOCK_K, BLOCK_N])
        acc += tl.dot(a, b_fp16)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)[None, :]
        acc += bias

    c = acc.to(tl.float16)

    # Store results
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def dequant_nf4_matmul_kernel(
    a_ptr, b_ptr, scales_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    block_size,
    # NF4 Table
    nf4_table_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Fused NF4 dequantization and matmul.
    B (weights) is packed [N, K//2] uint8.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = tl.ravel_index(pid, (num_pid_m, num_pid_n))

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # A is [M, K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is packed [N, K//2]. But for simplicity in logic, let's assume we handle unpacking.
    # If B is [N, K//2], then for a tile of [BLOCK_K, BLOCK_N], we need [BLOCK_K // 2, BLOCK_N].
    # This requires careful index math.
    
    # Load NF4 table into shared memory (SRAM)
    # Triton handles constants/small arrays well, but we'll follow the prompt's SRAM hint if possible.
    # However, tl.load from global is fine too.
    nf4_table = tl.load(nf4_table_ptr + tl.arange(0, 16))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        
        # B logic: B[n, k] where k is 0..K-1.
        # k_idx = k * BLOCK_K + offs_k[:, None]
        # n_idx = offs_bn[None, :]
        # byte_idx = (n_idx * K + k_idx) // 2
        # is_low = (k_idx % 2) == 0
        
        k_idx = k * BLOCK_K + offs_k[:, None]
        n_idx = offs_bn[None, :]
        
        # For simplicity, we assume K is even and block_size is even.
        byte_idx = (n_idx * K + k_idx) // 2
        packed_vals = tl.load(b_ptr + byte_idx, mask=(k_idx < K) & (n_idx < N), other=0)
        
        # Unpack
        is_low = (k_idx % 2) == 0
        nibble = tl.where(is_low, packed_vals & 0xF, (packed_vals >> 4) & 0xF)
        
        # Lookup
        # Note: Triton doesn't support indexing with tensors directly as easily as this, 
        # but tl.load(ptr + tensor) works.
        b_val = tl.load(nf4_table_ptr + nibble)
        
        # Scale
        scale_idx = (n_idx * K + k_idx) // block_size
        scales = tl.load(scales_ptr + scale_idx)
        
        b_fp16 = b_val.to(tl.float16) * scales.to(tl.float16)
        
        acc += tl.dot(a, b_fp16)
        
        a_ptrs += BLOCK_K * stride_ak

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)[None, :]
        acc += bias

    c = acc.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# --- Dispatch & Fallback ---

def _fallback_torch_dequant_int8_matmul(x, w_int8, scales, bias, block_size):
    # w_int8: [N, K], scales: [num_blocks]
    N, K = w_int8.shape
    w_fp16 = dequantize_int8(w_int8, scales, block_size)
    return F.linear(x, w_fp16, bias)

def _fallback_torch_dequant_nf4_matmul(x, w_packed, scales, bias, block_size):
    # w_packed can be 1D (from quantize_int4) or 2D [N, K//2].
    # Infer K from x.shape[-1] (the input feature dimension).
    K = x.shape[-1]
    # int4 packing: 2 values per uint8 byte → total elements = w_packed.numel() * 2
    N = w_packed.numel() * 2 // K
    w_fp16 = dequantize_int4(w_packed.flatten(), scales, block_size).reshape(N, K).to(x.dtype)
    if bias is not None:
        bias = bias.to(x.dtype)
    return F.linear(x, w_fp16, bias)

def _triton_dequant_int8_matmul(x, w_int8, scales, bias, block_size, cfg):
    M_orig, K = x.shape[0], x.shape[1] # Handle potentially 3D later
    # Flatten x to 2D
    x_2d = x.view(-1, K)
    M, _ = x_2d.shape
    N, _ = w_int8.shape
    
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    dequant_int8_matmul_kernel[grid](
        x_2d, w_int8, scales, output, bias,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        output.stride(0), output.stride(1),
        block_size,
        BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K,
        GROUP_SIZE_M=8,
        HAS_BIAS=bias is not None,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    
    return output.view(*x.shape[:-1], N)

def _triton_dequant_nf4_matmul(x, w_packed, scales, bias, block_size, cfg):
    from rocm_qlora.kernels.nf4_dequant import NF4_TABLE
    
    K = x.shape[-1]
    x_2d = x.view(-1, K)
    M, _ = x_2d.shape
    # w_packed can be 1D (from quantize_int4) or 2D [N, K//2]
    if w_packed.dim() == 1:
        N = w_packed.numel() * 2 // K
        w_packed_2d = w_packed.view(N, -1)
    else:
        N = w_packed.shape[0]
        w_packed_2d = w_packed
    
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    dequant_nf4_matmul_kernel[grid](
        x_2d, w_packed_2d, scales, output, bias,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_packed_2d.stride(0), w_packed_2d.stride(1),
        output.stride(0), output.stride(1),
        block_size,
        NF4_TABLE.to(x.device),
        BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K,
        GROUP_SIZE_M=8,
        HAS_BIAS=bias is not None,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    
    return output.view(*x.shape[:-1], N)

def fused_dequant_int8_matmul(x, w_int8, scales, bias=None, block_size=64):
    """
    Fused INT8 dequantization and matmul.
    If Triton is available and input is on GPU, use Triton kernel.
    Otherwise, use PyTorch fallback.
    """
    if TRITON_INSTALLED and x.is_cuda and is_triton_available():
        cfg = get_kernel_config()
        return _triton_dequant_int8_matmul(x, w_int8, scales, bias, block_size, cfg)
    else:
        return _fallback_torch_dequant_int8_matmul(x, w_int8, scales, bias, block_size)

def fused_dequant_nf4_matmul(x, w_packed, scales, bias=None, block_size=64):
    """
    Fused NF4 dequantization and matmul.
    If Triton is available and input is on GPU, use Triton kernel.
    Otherwise, use PyTorch fallback.
    """
    if TRITON_INSTALLED and x.is_cuda and is_triton_available():
        cfg = get_kernel_config()
        return _triton_dequant_nf4_matmul(x, w_packed, scales, bias, block_size, cfg)
    else:
        return _fallback_torch_dequant_nf4_matmul(x, w_packed, scales, bias, block_size)
