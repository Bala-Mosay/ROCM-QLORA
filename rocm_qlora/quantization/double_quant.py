"""
Double Quantization for rocm-qlora.

Quantizes the NF4 quantization constants (scales) a second time,
further reducing memory footprint without meaningful accuracy loss.

From the QLoRA paper:
  W: NF4, blocksize=64 -> scales c1 stored as FP32
  c1: re-quantized with FP8 E4M3, blocksize=256 -> c2 + c2_scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from rocm_qlora.quantization.quant_ops import quantize_int4, dequantize_int4

def _has_fp8() -> bool:
    """Checks if the current PyTorch version/hardware supports FP8."""
    return hasattr(torch, 'float8_e4m3fn')

@dataclass
class DoubleQuantState:
    """Complete state for double-quantized weights."""
    W_quant: torch.Tensor        # NF4 packed uint8, shape [ceil(N*K/2)]
    c2: torch.Tensor             # second-level quantized scales (FP8 or INT8)
    c2_scales: torch.Tensor      # scales for c2, FP32, shape [ceil(num_blocks_1/blocksize_2)]
    blocksize_1: int = 64        # NF4 block size
    blocksize_2: int = 256       # second quantization block size
    original_shape: tuple = None # (out_features, in_features)
    use_fp8_c2: bool = False     # whether c2 uses FP8 or INT8 fallback

def double_quantize(
    tensor: torch.Tensor, 
    blocksize_1: int = 64, 
    blocksize_2: int = 256
) -> DoubleQuantState:
    """
    Performs double quantization on a weight tensor.
    """
    original_shape = tensor.shape
    # 1. First-level NF4 quantization
    W_packed, c1_scales = quantize_int4(tensor, block_size=blocksize_1)
    
    # 2. Second-level quantization of c1_scales
    # c1_scales has length ceil(num_params / blocksize_1)
    num_c1 = c1_scales.numel()
    padding = (blocksize_2 - (num_c1 % blocksize_2)) % blocksize_2
    if padding > 0:
        c1_padded = F.pad(c1_scales, (0, padding))
    else:
        c1_padded = c1_scales
        
    reshaped_c1 = c1_padded.view(-1, blocksize_2)
    # Absmax scaling for c1_scales
    c2_scales = reshaped_c1.abs().max(dim=1, keepdim=True).values
    c2_scales = torch.clamp(c2_scales, min=1e-12)
    
    use_fp8 = _has_fp8()
    if use_fp8:
        # NOTE: FP8 E4M3 has a range of +/- 448.0. 
        # Since c1_scales are normalized by c2_scales, they fit in [-1, 1].
        # We scale by the maximum representable value for maximum precision.
        # But wait, torch.float8_e4m3fn is a first-class dtype.
        # We can just normalize and cast.
        normalized_c1 = reshaped_c1 / c2_scales
        # Using a simple cast to FP8. 
        # Note: In real production with ROCm 6.2, we'd use hipblasLt or tuned kernels.
        # Here we follow the PyTorch-native approach.
        c2 = normalized_c1.to(torch.float8_e4m3fn)
    else:
        # INT8 fallback
        normalized_c1 = reshaped_c1 / c2_scales
        c2 = torch.round(normalized_c1 * 127.0).to(torch.int8)
        
    return DoubleQuantState(
        W_quant=W_packed,
        c2=c2.view(-1),
        c2_scales=c2_scales.squeeze(),
        blocksize_1=blocksize_1,
        blocksize_2=blocksize_2,
        original_shape=original_shape,
        use_fp8_c2=use_fp8
    )

def double_dequantize(state: DoubleQuantState) -> torch.Tensor:
    """
    Reconstructs the original tensor from double-quantized state.
    """
    # 1. Reconstruct c1_scales from c2 + c2_scales
    c2_reshaped = state.c2.view(-1, state.blocksize_2)
    
    if state.use_fp8_c2:
        # Cast back to FP32 for math
        c1_reconstructed = c2_reshaped.to(torch.float32) * state.c2_scales.unsqueeze(-1)
    else:
        # INT8 fallback
        c1_reconstructed = (c2_reshaped.to(torch.float32) / 127.0) * state.c2_scales.unsqueeze(-1)
        
    # Remove padding added during double_quantize
    num_blocks_1 = (torch.prod(torch.tensor(state.original_shape)) + state.blocksize_1 - 1) // state.blocksize_1
    c1_final = c1_reconstructed.view(-1)[:num_blocks_1]
    
    # 2. Dequantize NF4 weights using reconstructed c1_scales
    W_fp16 = dequantize_int4(state.W_quant, c1_final, block_size=state.blocksize_1)
    
    # 3. Reshape and truncate padding
    num_elements = torch.prod(torch.tensor(state.original_shape)).item()
    return W_fp16[:num_elements].reshape(state.original_shape)

def estimate_double_quant_savings(tensor: torch.Tensor, blocksize_1: int = 64, blocksize_2: int = 256) -> Dict[str, Any]:
    """
    Estimates memory savings of double quantization.
    """
    num_params = tensor.numel()
    num_blocks_1 = (num_params + blocksize_1 - 1) // blocksize_1
    num_blocks_2 = (num_blocks_1 + blocksize_2 - 1) // blocksize_2
    
    # NF4: 4 bits per param (packed 2 per byte)
    nf4_bytes = (num_params + 1) // 2
    
    # Without DQ: scales c1 are FP32 (4 bytes)
    c1_bytes_naive = num_blocks_1 * 4
    total_naive = nf4_bytes + c1_bytes_naive
    
    # With DQ: scales c2 are FP8/INT8 (1 byte) + c2_scales are FP32 (4 bytes)
    c2_bytes = num_blocks_1 * 1
    c2_scales_bytes = num_blocks_2 * 4
    total_dq = nf4_bytes + c2_bytes + c2_scales_bytes
    
    saved = total_naive - total_dq
    saved_pct = (saved / total_naive) * 100
    
    # Bits per param calculation
    naive_bits = (total_naive * 8) / num_params
    dq_bits = (total_dq * 8) / num_params
    bits_saved = naive_bits - dq_bits
    
    return {
        "without_dq_bytes": total_naive,
        "with_dq_bytes": total_dq,
        "saved_bytes": saved,
        "saved_pct": saved_pct,
        "bits_saved_per_param": bits_saved
    }

def patch_quant_linear_for_double_quant(model: nn.Module) -> int:
    """
    Finds all INT4 QuantLinear layers and upgrades them to double quantization.
    """
    from rocm_qlora.quantization.quant_linear import QuantLinear
    count = 0
    
    # Collect layers first to avoid mutation issues
    layers_to_patch = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear) and module.bits == 4:
            layers_to_patch.append((name, module))
            
    for name, module in layers_to_patch:
        # 1. Dequantize current weight to get the FP16 base for re-quantization
        # (This is expensive but happens once at setup)
        from rocm_qlora.quantization.quant_ops import dequantize_int4
        W_fp = dequantize_int4(module.weight_quant, module.weight_scales, block_size=module.block_size)
        num_elements = module.out_features * module.in_features
        W_fp = W_fp[:num_elements].reshape(module.original_shape)
        
        # 2. Perform double quantization
        state = double_quantize(W_fp, blocksize_1=module.block_size, blocksize_2=256)
        
        # 3. Register new buffers
        module.register_buffer("c2", state.c2)
        module.register_buffer("c2_scales", state.c2_scales)
        module.blocksize_2 = state.blocksize_2
        module.use_fp8_c2 = state.use_fp8_c2
        module.use_double_quant = True
        
        # Note: weight_quant remains the same (NF4 bits), 
        # but weight_scales is now effectively redundant (replaced by c2 + c2_scales).
        # We keep weight_scales for compatibility or zero it out to save VRAM.
        # Paper suggests keeping it is not necessary.
        del module.weight_scales
        module.register_buffer("weight_scales", torch.empty(0)) # dummy
        
        count += 1
        
    return count
