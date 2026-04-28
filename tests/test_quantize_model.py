"""
Unit tests for quantize_model orchestration.
Uses a mock transformer architecture to verify layer replacement and gradient flow.
"""

import torch
import torch.nn as nn
import pytest
from rocm_qlora.model.quantize_model import quantize_model
from rocm_qlora.lora.lora_layer import LoRALinear

class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)
        self.o_proj = nn.Linear(128, 128)
        self.ff = nn.Linear(128, 512)   # not a target module

class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeAttention() for _ in range(4)])
        self.tiny_proj = nn.Linear(32, 32) # to test skipping small layers

    def forward(self, x):
        for layer in self.layers:
            # Simple chain for testing
            x = layer.q_proj(x)
        return x

def test_target_layers_replaced():
    """Verify that only specified target modules are replaced with LoRALinear."""
    model = FakeTransformer()
    target_modules = ["q_proj", "v_proj"]
    quantize_model(model, bits=8, target_modules=target_modules)
    
    for i in range(4):
        assert isinstance(model.layers[i].q_proj, LoRALinear)
        assert isinstance(model.layers[i].v_proj, LoRALinear)
        # Check non-targets
        assert isinstance(model.layers[i].k_proj, nn.Linear)
        assert isinstance(model.layers[i].o_proj, nn.Linear)
        assert isinstance(model.layers[i].ff, nn.Linear)

def test_non_target_layers_unchanged():
    """Ensure that non-target modules remain standard nn.Linear layers."""
    model = FakeTransformer()
    quantize_model(model, bits=4, target_modules=["q_proj"])
    
    for i in range(4):
        assert isinstance(model.layers[i].k_proj, nn.Linear)
        assert isinstance(model.layers[i].ff, nn.Linear)

def test_forward_pass_runs():
    """Verify that the quantized model can perform a forward pass."""
    model = FakeTransformer()
    quantize_model(model, bits=8)
    
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == (2, 128)

def test_backward_pass_runs():
    """Verify that gradients can be computed for LoRA parameters."""
    model = FakeTransformer()
    quantize_model(model, bits=8)
    
    x = torch.randn(2, 128)
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    # Check that some gradient exists
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad

def test_only_lora_params_have_gradients():
    """Ensure that only LoRA A and B matrices receive gradients."""
    model = FakeTransformer()
    quantize_model(model, bits=4, target_modules=["q_proj"])
    
    x = torch.randn(2, 128)
    out = model(x)
    out.mean().backward()
    
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None
        else:
            # Standard nn.Module parameters (like ff.weight) should have no grad
            # or if they were turned into buffers, they won't even be in parameters()
            assert param.grad is None or not param.requires_grad

def test_trainable_param_count():
    """Verify correct calculation of trainable parameters (4 layers * 2 targets)."""
    model = FakeTransformer()
    r = 8
    # 4 layers, target q_proj and v_proj = 8 replaced layers
    # Each 128->128 layer with r=8 has (8*128 + 128*8) = 2048 params
    # Total = 8 * 2048 = 16384
    quantize_model(model, bits=8, lora_r=r, target_modules=["q_proj", "v_proj"])
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 16384

def test_invalid_bits_raises():
    """Verify that unsupported bit depths raise a ValueError."""
    model = FakeTransformer()
    with pytest.raises(ValueError, match="Only 4 and 8 bits are supported"):
        quantize_model(model, bits=3)

def test_small_layer_skipped():
    """Verify that layers smaller than 64 features are skipped."""
    model = FakeTransformer()
    # tiny_proj is 32x32
    quantize_model(model, target_modules=["tiny_proj"])
    
    assert isinstance(model.tiny_proj, nn.Linear)
    assert not isinstance(model.tiny_proj, LoRALinear)

def test_two_pass_safety():
    """Verify that calling quantize_model twice does not cause a crash."""
    model = FakeTransformer()
    quantize_model(model, bits=8, target_modules=["q_proj"])
    # Second call targets same layer
    quantize_model(model, bits=8, target_modules=["q_proj"])
    
    assert isinstance(model.layers[0].q_proj, LoRALinear)

if __name__ == "__main__":
    pytest.main([__file__])
