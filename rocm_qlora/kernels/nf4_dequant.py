"""
Standalone NF4 dequantization kernel for Triton.
Optimized for AMD ROCm via shared memory table lookups.
"""
import torch
from typing import Tuple, Optional

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

from rocm_qlora.quantization.quant_ops import NF4_VALUES

# Module-level constant for NF4 lookup
NF4_TABLE = torch.tensor(NF4_VALUES, dtype=torch.float32)

@triton.jit
def nf4_dequant_kernel(
    packed_ptr, scales_ptr, out_ptr,
    nf4_table_ptr,
    num_elements,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantizes NF4 packed bytes into FP16.
    packed_ptr: [num_elements // 2] uint8
    scales_ptr: [num_elements // block_size] float32
    out_ptr: [num_elements] float16
    """
    pid = tl.program_id(0)
    
    # We process BLOCK_SIZE elements per program
    # Each program handles BLOCK_SIZE // 2 packed bytes
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements
    
    # Load NF4 table into shared memory
    # Only need 16 values
    table_offs = tl.arange(0, 16)
    nf4_table = tl.load(nf4_table_ptr + table_offs)
    
    # Calculate byte indices and nibble positions
    # byte_idx = offs // 2
    # is_high = (offs % 2) == 1
    byte_idx = offs >> 1
    is_high = (offs & 1) == 1
    
    # Load packed bytes
    packed_vals = tl.load(packed_ptr + byte_idx, mask=mask, other=0)
    
    # Unpack nibbles
    # byte = (high << 4) | low
    indices = tl.where(is_high, (packed_vals >> 4) & 0x0F, packed_vals & 0x0F)
    
    # Table lookup (from shared memory)
    # Triton JIT will often optimize this if nf4_table is small
    vals = tl.load(nf4_table_ptr + indices)
    
    # Apply scales
    scale_idx = offs // block_size
    scales = tl.load(scales_ptr + scale_idx, mask=mask, other=1.0)
    
    dequantized = vals.to(tl.float32) * scales.to(tl.float32)
    
    # Store result
    tl.store(out_ptr + offs, dequantized, mask=mask)

def dequantize_nf4_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 64,
    original_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Dequantizes a packed NF4 tensor using a Triton GPU kernel.
    """
    if not TRITON_INSTALLED or not packed.is_cuda or not is_triton_available():
        # Fallback handled by the caller or explicitly here if needed
        # But per requirements, this function is the Triton implementation.
        from rocm_qlora.quantization.quant_ops import dequantize_int4
        return dequantize_int4(packed, scales, block_size)

    num_elements = packed.numel() * 2
    out = torch.empty(num_elements, device=packed.device, dtype=torch.bfloat16)
    
    # Use config for block size if needed, or a fixed reasonable size for this simple kernel
    # Since this is a 1D element-wise operation, BLOCK_SIZE=1024 is usually good.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    nf4_dequant_kernel[grid](
        packed, scales, out,
        NF4_TABLE.to(packed.device),
        num_elements,
        block_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    if original_shape:
        return out.view(original_shape)
    return out
