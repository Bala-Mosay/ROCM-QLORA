"""
Unit tests for quantization operations.
Tests roundtrip accuracy and memory savings for INT8 and INT4 (NF4).
"""

import torch
import pytest
from rocm_qlora.quantization.quant_ops import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4
)

@pytest.mark.parametrize("block_size", [64, 128])
def test_int8_roundtrip(block_size):
    """Test INT8 blockwise quantization roundtrip accuracy."""
    torch.manual_seed(42)
    # Use random tensor with some outliers
    tensor = torch.randn(1024) * 2.0
    tensor[0] = 100.0 # Outlier
    
    q_tensor, scales = quantize_int8(tensor, block_size=block_size)
    dq_tensor = dequantize_int8(q_tensor, scales, block_size=block_size)
    
    # Check max error: should be < 1% of max abs value
    max_val = tensor.abs().max()
    error = (tensor - dq_tensor[:tensor.numel()]).abs().max()
    
    assert error < 0.01 * max_val
    assert q_tensor.dtype == torch.int8

def test_int8_per_channel():
    """Test INT8 per-channel quantization."""
    # Linear weight shape [Out, In]
    tensor = torch.randn(16, 64)
    
    q_tensor, scales = quantize_int8(tensor, per_channel=True)
    
    assert scales.numel() == 16
    assert q_tensor.shape == tensor.shape
    
    dq_tensor = dequantize_int8(q_tensor, scales)
    max_val = tensor.abs().max()
    error = (tensor - dq_tensor).abs().max()
    
    assert error < 0.01 * max_val

@pytest.mark.parametrize("block_size", [64, 128])
def test_int4_roundtrip(block_size):
    """Test INT4 (NF4) quantization roundtrip accuracy."""
    torch.manual_seed(42)
    tensor = torch.randn(1024)
    
    packed, scales = quantize_int4(tensor, block_size=block_size)
    dq_tensor = dequantize_int4(packed, scales, block_size=block_size)
    
    # NF4 is more lossy, error < 15% of max abs value due to NF4 table gaps
    max_val = tensor.abs().max()
    error = (tensor - dq_tensor[:tensor.numel()]).abs().max()
    
    assert error < 0.15 * max_val
    assert packed.dtype == torch.uint8
    # Packed size should be half of total elements (adjusted for block/even)
    assert packed.numel() * 2 >= tensor.numel()

def test_memory_savings():
    """Verify that quantized weights use significantly less memory."""
    num_elements = 1024 * 1024 # 1M params
    tensor = torch.randn(num_elements, dtype=torch.float32)
    
    # Original size (FP32)
    orig_size = tensor.element_size() * tensor.numel()
    
    # INT8 size
    q8, scales8 = quantize_int8(tensor)
    q8_size = q8.element_size() * q8.numel() + scales8.element_size() * scales8.numel()
    
    # INT4 size
    q4, scales4 = quantize_int4(tensor)
    q4_size = q4.element_size() * q4.numel() + scales4.element_size() * scales4.numel()
    
    assert q8_size < 0.35 * orig_size # 8-bit vs 32-bit (plus scales)
    assert q4_size < 0.20 * orig_size # 4-bit vs 32-bit (plus scales)

def test_device_agnostic():
    """Ensure ops work on available devices (CPU/CUDA)."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        
    for device in devices:
        tensor = torch.randn(128, device=device)
        # Just check if it runs without error
        q8, s8 = quantize_int8(tensor)
        dq8 = dequantize_int8(q8, s8)
        
        q4, s4 = quantize_int4(tensor)
        dq4 = dequantize_int4(q4, s4)
        
        assert dq8.device.type == device
        assert dq4.device.type == device

if __name__ == "__main__":
    pytest.main([__file__])
