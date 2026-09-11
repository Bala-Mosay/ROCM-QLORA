"""
Unit tests for LoRALinear layer.
Verifies initialization, gradient flow, and merging logic.
"""

import torch
import torch.nn as nn
import pytest
from rocm_qlora.quantization.quant_linear import QuantLinear
from rocm_qlora.lora.lora_layer import LoRALinear

def test_zero_init_exact():
    """Verify that LoRA output is exactly zero at initialization."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    x = torch.randn(2, in_features)
    base_out = ql(x)
    lora_out = lora(x)
    
    # At init, lora_B is all zeros, so lora_out should equal base_out
    assert torch.allclose(lora_out, base_out, atol=1e-6)

def test_only_lora_trainable():
    """Verify that only LoRA A and B matrices are trainable."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    trainable_params = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert len(trainable_params) == 2
    assert "lora_A" in trainable_params
    assert "lora_B" in trainable_params
    
    # Ensure base buffers are NOT parameters
    assert not any("base" in n for n in trainable_params)

def test_trainable_param_count():
    """Verify correct calculation of trainable parameters."""
    in_features, out_features, r = 64, 128, 8
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=r)
    
    # Expected: (in * r) + (out * r)
    expected = (in_features * r) + (out_features * r)
    assert lora.get_trainable_params() == expected
    assert lora.get_trainable_params() == 1536

def test_gradient_flows_lora():
    """Verify that gradients propagate to LoRA parameters."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    # Ensure B is non-zero so gradients flow to A
    with torch.no_grad():
        lora.lora_B.normal_()
        
    x = torch.randn(2, in_features)
    out = lora(x)
    loss = out.sum()
    loss.backward()
    
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.lora_A.grad.abs().sum() > 0
    assert lora.lora_B.grad.abs().sum() > 0

def test_gradient_blocked_base():
    """Verify that base QuantLinear buffers receive no gradients."""
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    x = torch.randn(2, in_features)
    out = lora(x)
    loss = out.sum()
    loss.backward()
    
    # Buffers shouldn't have grad attributes at all, but check state
    for name, buffer in lora.base_layer.named_buffers():
        assert buffer.grad is None

def test_merge_lora_output_close():
    """Verify that merging LoRA weights preserves numerical behavior."""
    torch.manual_seed(42)
    in_features, out_features = 64, 128
    linear = nn.Linear(in_features, out_features)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    # Manually set some LoRA weights to ensure they are non-zero
    with torch.no_grad():
        lora.lora_A.normal_()
        lora.lora_B.normal_()
        
    x = torch.randn(2, in_features)
    pre_merge_out = lora(x)
    
    lora.merge_lora()
    post_merge_out = lora(x)
    
    # Merging introduces re-quantization error, so we use atol=2.0
    assert torch.allclose(post_merge_out, pre_merge_out, atol=2.0)
    assert lora.merged is True
    # LoRA params should be zeroed after merge
    assert lora.lora_A.abs().sum() == 0
    assert lora.lora_B.abs().sum() == 0

def test_merge_lora_idempotent_guard():
    """Verify that merging twice raises an error."""
    linear = nn.Linear(64, 128)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    lora.merge_lora()
    with pytest.raises(RuntimeError, match="LoRA already merged"):
        lora.merge_lora()

def test_dtype_safety():
    """Verify that output dtype matches input dtype (no promotion)."""
    linear = nn.Linear(64, 128)
    ql = QuantLinear.from_linear(linear)
    lora = LoRALinear(ql, r=8)
    
    # Test with float16 (simulating mixed precision)
    x = torch.randn(2, 64).to(torch.float16)
    out = lora(x)
    
    assert out.dtype == torch.float16

if __name__ == "__main__":
    pytest.main([__file__])
