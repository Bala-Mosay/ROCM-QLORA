"""
Quantization operations for INT8 and INT4 (NF4) blockwise quantization.
Pure PyTorch implementation optimized for ROCm.
"""

import torch
from typing import Tuple

# Hardcoded NF4 lookup table values as per the project contract
NF4_VALUES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

def quantize_int8(tensor: torch.Tensor, block_size: int = 64, per_channel: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (int8_tensor, scales). scales shape: [num_blocks] or [out_channels].
    Uses blockwise absmax quantization.
    """
    # Keep track of original shape for potential restoration if needed, 
    # though usually we work with flattened/reshaped weights.
    original_device = tensor.device
    original_dtype = tensor.dtype
    
    if per_channel:
        # Per-channel (per-row for Linear weights)
        # Assuming tensor is [Out, In]
        scales = tensor.abs().max(dim=1, keepdim=True).values / 127.0
        # Avoid division by zero
        scales = torch.clamp(scales, min=1e-12)
        int8_tensor = torch.round(tensor / scales).to(torch.int8)
        return int8_tensor, scales.squeeze()
    else:
        # Blockwise quantization
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()
        
        # Pad to block_size if necessary
        padding = (block_size - (num_elements % block_size)) % block_size
        if padding > 0:
            flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding))
            
        reshaped = flat_tensor.view(-1, block_size)
        scales = reshaped.abs().max(dim=1, keepdim=True).values / 127.0
        scales = torch.clamp(scales, min=1e-12)
        
        quantized = torch.round(reshaped / scales).to(torch.int8)
        
        # We return the flattened quantized tensor (including padding) and the scales
        return quantized.view(-1), scales.squeeze()

def dequantize_int8(int8_tensor: torch.Tensor, scales: torch.Tensor,
    block_size: int = 64) -> torch.Tensor:
    """
    Returns FP16/BF16 tensor reconstructed from int8 + scales.
    Supports both blockwise and per-channel scales.
    """
    # NOTE: We handle both flattened and structured tensors.
    # If it's per-channel, scales will typically match the first dimension of int8_tensor.
    if int8_tensor.dim() == 2 and scales.numel() == int8_tensor.size(0):
        # Per-channel case: [Out, In] * [Out, 1]
        return int8_tensor.to(torch.float32) * scales.view(-1, 1)
    
    # Blockwise case: we assume the tensor is flattened or can be reshaped by block_size
    # We reshape int8_tensor to [-1, block_size] to match scales
    flat_int8 = int8_tensor.flatten()
    reshaped = flat_int8.view(-1, block_size)
    
    # Ensure scales is [num_blocks, 1] for broadcasting
    dequantized = reshaped.to(torch.float32) * scales.view(-1, 1)
    
    # Return in the original shape if possible, otherwise flattened
    if int8_tensor.dim() == 2 and not (scales.numel() == int8_tensor.size(0)):
        return dequantized.view(int8_tensor.shape)
    return dequantized.view(-1)

def quantize_int4(tensor: torch.Tensor, block_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (packed_uint8, scales). Two INT4 values packed per byte.
    Uses NormalFloat 4 (NF4) quantization.
    """
    device = tensor.device
    nf4_table = torch.tensor(NF4_VALUES, device=device, dtype=torch.float32)
    
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()
    
    # Pad to block_size if necessary (must also be even for packing)
    padding = (block_size - (num_elements % block_size)) % block_size
    if padding > 0:
        flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding))
    
    # Final padding check: must be even total elements for 2-per-byte packing
    if flat_tensor.numel() % 2 != 0:
        flat_tensor = torch.nn.functional.pad(flat_tensor, (0, 1))

    reshaped = flat_tensor.view(-1, block_size)
    
    # 1. Compute scales (absmax)
    scales = reshaped.abs().max(dim=1, keepdim=True).values
    scales = torch.clamp(scales, min=1e-12)
    
    # 2. Normalize to [-1, 1]
    normalized = reshaped / scales
    
    # 3. Find nearest NF4 index
    # Use cdist for efficient distance calculation
    # normalized: [num_blocks, block_size], nf4_table: [16]
    # We need to compute distance between every element and every NF4 value
    dist = torch.abs(normalized.unsqueeze(-1) - nf4_table)
    indices = torch.argmin(dist, dim=-1).to(torch.uint8) # [num_blocks, block_size]
    
    # 4. Pack two INT4 into one uint8
    # indices is flat again for packing
    indices_flat = indices.view(-1)
    # Pack: even indices are low nibble, odd indices are high nibble (or vice versa)
    # We'll use: byte = (high << 4) | low
    low_nibbles = indices_flat[0::2]
    high_nibbles = indices_flat[1::2]
    packed = (high_nibbles << 4) | low_nibbles
    
    return packed, scales.squeeze()

def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor,
    block_size: int = 64) -> torch.Tensor:
    """
    Returns FP16/BF16 tensor from packed INT4 + scales.
    """
    device = packed.device
    nf4_table = torch.tensor(NF4_VALUES, device=device, dtype=torch.float32)
    
    # 1. Unpack uint8 into two uint8 (indices)
    low_nibbles = packed & 0x0F
    high_nibbles = packed >> 4
    
    # Interleave them back: [low0, high0, low1, high1, ...]
    indices = torch.stack([low_nibbles, high_nibbles], dim=1).view(-1)
    
    # 2. Map indices to NF4 values
    nf4_quantized = nf4_table[indices.long()]
    
    # 3. Reshape and apply scales
    reshaped = nf4_quantized.view(-1, block_size)
    dequantized = reshaped * scales.unsqueeze(-1)
    
    return dequantized.view(-1)
