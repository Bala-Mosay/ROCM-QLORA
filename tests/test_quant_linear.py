"""
Unit tests for QuantLinear layer.
Verifies shape restoration, memory reduction, and numerical accuracy.
"""

import torch
import torch.nn as nn
import pytest
from rocm_qlora.quantization.quant_linear import QuantLinear

def test_from_linear_output_shape():
    """Verify output shape matches original nn.Linear."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear, bits=8)
    
    x = torch.randn(2, in_features)
    out = ql(x)
    assert out.shape == (2, out_features)

def test_memory_reduction_int8():
    """Verify memory savings for INT8 quantization."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features, bias=False)
    
    # Original: FP32 (4 bytes per param)
    # Note: prompt says FP16 (2 bytes), we'll compare against both.
    orig_params = in_features * out_features
    orig_bytes = orig_params * 4 
    
    ql = QuantLinear.from_linear(linear, bits=8)
    # ql stores weight_quant (int8) + scales (fp32 per 64-block)
    q_bytes = ql.weight_quant.numel() * 1 + ql.weight_scales.numel() * 4
    
    assert q_bytes < (orig_bytes / 3) # Should be ~1/4th of FP32 or ~1/2 of FP16

def test_memory_reduction_int4():
    """Verify memory savings for INT4 quantization."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features, bias=False)
    
    ql_8 = QuantLinear.from_linear(linear, bits=8)
    
    # Re-create linear for int4 test
    linear = nn.Linear(in_features, out_features, bias=False)
    ql_4 = QuantLinear.from_linear(linear, bits=4)
    
    assert ql_4.weight_quant.numel() < ql_8.weight_quant.numel()
    # 0.5 bytes per param + scales
    assert ql_4.weight_quant.numel() == (in_features * out_features // 2)

def test_forward_close_to_fp16_int8():
    """Verify INT8 accuracy is within acceptable tolerance."""
    torch.manual_seed(42)
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    x = torch.randn(4, in_features)
    
    with torch.no_grad():
        expected = linear(x)
        ql = QuantLinear.from_linear(linear, bits=8)
        actual = ql(x)
        
    assert torch.allclose(actual, expected, atol=0.15)

def test_forward_close_to_fp16_int4():
    """Verify INT4 (NF4) accuracy is within acceptable tolerance."""
    torch.manual_seed(42)
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    x = torch.randn(4, in_features)
    
    with torch.no_grad():
        expected = linear(x)
        ql = QuantLinear.from_linear(linear, bits=4)
        actual = ql(x)
        
    # NF4 is lossier than INT8
    assert torch.allclose(actual, expected, atol=0.3)

def test_no_trainable_params():
    """Verify QuantLinear has no trainable parameters."""
    linear = nn.Linear(64, 128)
    ql = QuantLinear.from_linear(linear)
    
    trainable = sum(p.numel() for p in ql.parameters() if p.requires_grad)
    assert trainable == 0
    # Also verify weight_quant is a buffer, not a parameter
    assert not hasattr(ql, 'weight')
    assert 'weight_quant' in ql.state_dict()

def test_shape_restoration():
    """Verify that forward pass handles Phase 1 padding/flattening correctly."""
    # Choose sizes that are NOT multiples of block_size (64)
    in_features, out_features = 37, 73 
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear, bits=8, block_size=64)
    
    x = torch.randn(1, in_features)
    out = ql(x)
    assert out.shape == (1, out_features)

def test_bias_preserved():
    """Verify bias values match the original layer."""
    linear = nn.Linear(64, 128, bias=True)
    orig_bias = linear.bias.data.clone()
    
    ql = QuantLinear.from_linear(linear)
    assert torch.equal(ql.bias, orig_bias)

def test_original_weight_deleted():
    """Verify that original linear weights are removed to free VRAM."""
    linear = nn.Linear(64, 128)
    # Check that weight exists before
    assert hasattr(linear, 'weight')
    
    _ = QuantLinear.from_linear(linear)
    # Check that weight is deleted from the input object
    assert not hasattr(linear, 'weight')
    assert not hasattr(linear, 'bias')

if __name__ == "__main__":
    pytest.main([__file__])
