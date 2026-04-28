import pytest
import torch
import torch.nn as nn
import math
from rocm_qlora.quantization.double_quant import (
    double_quantize, double_dequantize,
    DoubleQuantState, estimate_double_quant_savings,
    patch_quant_linear_for_double_quant, _has_fp8
)
from rocm_qlora.quantization import QuantLinear
from rocm_qlora.lora.lora_layer import LoRALinear
from rocm_qlora import quantize_model

def test_double_quant_state_fields():
    """DoubleQuantState has all required fields."""
    state = DoubleQuantState(
        W_quant=torch.zeros(1, dtype=torch.uint8),
        c2=torch.zeros(1),
        c2_scales=torch.zeros(1),
        original_shape=(10, 10)
    )
    assert hasattr(state, "W_quant")
    assert hasattr(state, "c2")
    assert hasattr(state, "c2_scales")
    assert hasattr(state, "blocksize_1")
    assert hasattr(state, "blocksize_2")
    assert state.blocksize_1 == 64
    assert state.blocksize_2 == 256

def test_double_quantize_returns_state():
    """double_quantize(torch.randn(64,64)) returns DoubleQuantState."""
    W = torch.randn(64, 64)
    state = double_quantize(W, blocksize_1=64, blocksize_2=256)
    assert isinstance(state, DoubleQuantState)
    assert state.original_shape == (64, 64)
    assert state.W_quant.dtype == torch.uint8
    # num_blocks_1 = 4096 / 64 = 64
    # c2 shape should be num_blocks_1 = 64
    assert state.c2.numel() >= 64
    # num_blocks_2 = ceil(64 / 256) = 1
    assert state.c2_scales.numel() == 1

def test_double_dequantize_shape_matches():
    """roundtrip output shape matches input shape."""
    W = torch.randn(128, 64)
    state = double_quantize(W)
    W_rec = double_dequantize(state)
    assert W_rec.shape == W.shape

def test_double_dequantize_close_to_single():
    """double_dequantize(double_quantize(W)) within atol=0.3 of dequantize_int4(quantize_int4(W))."""
    from rocm_qlora.quantization.quant_ops import quantize_int4, dequantize_int4
    W = torch.randn(256, 256)
    
    # Single quant
    q, s = quantize_int4(W, block_size=64)
    W_single = dequantize_int4(q, s, block_size=64)[:W.numel()].reshape(W.shape)
    
    # Double quant
    state = double_quantize(W, blocksize_1=64, blocksize_2=256)
    W_double = double_dequantize(state)
    
    # Compare double to single (should be very close as double quant of scales is high precision)
    # The paper says accuracy loss is minimal.
    diff = (W_double - W_single).abs().mean()
    assert diff < 0.1

def test_memory_savings_positive():
    """estimate_double_quant_savings returns saved_bytes > 0."""
    W = torch.randn(1024, 1024) # 1M params
    savings = estimate_double_quant_savings(W)
    assert savings["saved_bytes"] > 0
    assert savings["saved_pct"] > 0

def test_bits_saved_near_paper_value():
    """bits_saved_per_param is between 0.3 and 0.5 (paper says ~0.37)."""
    W = torch.randn(4096, 4096)
    savings = estimate_double_quant_savings(W, blocksize_1=64, blocksize_2=256)
    bits_saved = savings["bits_saved_per_param"]
    # Theory: 0.5 - (8/64 + 32/(64*256)) = 0.5 - 0.12695 = 0.37305
    assert 0.35 < bits_saved < 0.40

class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)

def test_patch_returns_count():
    """patch_quant_linear_for_double_quant on FakeTransformer after quantize_model returns int > 0."""
    model = FakeTransformer()
    # Quantize to 4-bit first
    model = quantize_model(model, bits=4, target_modules=["q_proj", "v_proj"])
    
    count = patch_quant_linear_for_double_quant(model)
    assert count == 2
    
    # Check patching (handling potential LoRA wrapper)
    q_proj = model.q_proj
    if isinstance(q_proj, LoRALinear):
        q_proj = q_proj.base
        
    assert q_proj.use_double_quant is True
    assert hasattr(q_proj, "c2")
    assert hasattr(q_proj, "c2_scales")

def test_forward_with_double_quant():
    """after patching, model(x) runs without error."""
    model = FakeTransformer()
    model = quantize_model(model, bits=4)
    patch_quant_linear_for_double_quant(model)
    
    x = torch.randn(1, 128)
    out = model.q_proj(x)
    assert out.shape == (1, 128)

def test_forward_output_close_before_after():
    """forward output before/after double quant patching within atol=0.5."""
    model = FakeTransformer()
    model = quantize_model(model, bits=4)
    
    x = torch.randn(1, 128)
    out_before = model.q_proj(x).detach()
    
    patch_quant_linear_for_double_quant(model)
    out_after = model.q_proj(x).detach()
    
    diff = (out_after - out_before).abs().mean()
    assert diff < 0.1

def test_fp8_fallback_when_unavailable():
    """if _has_fp8()=False, c2 uses INT8 and function still works."""
    with mock.patch("rocm_qlora.quantization.double_quant._has_fp8", return_value=False):
        W = torch.randn(128, 128)
        state = double_quantize(W)
        assert state.use_fp8_c2 is False
        assert state.c2.dtype == torch.int8
        
        W_rec = double_dequantize(state)
        assert W_rec.shape == W.shape

import unittest.mock as mock
