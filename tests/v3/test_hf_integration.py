import pytest
import unittest.mock as mock
import torch
import torch.nn as nn
from rocm_qlora.hf_integration import RocmQLoraConfig, RocmQLoraQuantizer

# Mocking transformers to avoid dependency issues during tests
def mock_hf_quantizer_available():
    fake_hf_quantizer = mock.MagicMock()
    fake_hf_quantizer.HfQuantizer = object  # base class
    return mock.patch.dict('sys.modules', {
        'transformers.quantizers.base': fake_hf_quantizer,
        'transformers.utils.quantization_config': mock.MagicMock()
    })

def test_config_instantiates_defaults():
    config = RocmQLoraConfig()
    assert config.bits == 8
    assert config.lora_r == 8
    assert config.quant_method == "rocm_qlora"

def test_config_custom_values():
    config = RocmQLoraConfig(bits=4, lora_r=16)
    assert config.bits == 4
    assert config.lora_r == 16

def test_config_invalid_bits_raises():
    with pytest.raises(ValueError, match="only supports 4 or 8 bits"):
        RocmQLoraConfig(bits=3)

def test_config_to_dict_has_all_fields():
    config = RocmQLoraConfig(bits=4, lora_r=32)
    d = config.to_dict()
    assert d["bits"] == 4
    assert d["lora_r"] == 32
    assert d["quant_method"] == "rocm_qlora"
    assert "target_modules" in d

def test_config_from_dict_roundtrip():
    original = RocmQLoraConfig(bits=4, lora_r=16, lora_alpha=32)
    reconstructed = RocmQLoraConfig.from_dict(original.to_dict())
    assert reconstructed.bits == original.bits
    assert reconstructed.lora_r == original.lora_r
    assert reconstructed.lora_alpha == original.lora_alpha

def test_config_repr_is_string():
    assert isinstance(repr(RocmQLoraConfig()), str)
    assert "RocmQLoraConfig" in repr(RocmQLoraConfig())

def test_config_absorbs_unknown_kwargs():
    # Should not crash
    config = RocmQLoraConfig(future_param="surprise")
    assert config.bits == 8

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)

def test_quantizer_is_trainable_true():
    config = RocmQLoraConfig()
    quantizer = RocmQLoraQuantizer(config)
    assert quantizer.is_trainable() is True

def test_quantizer_is_serializable_true():
    config = RocmQLoraConfig()
    quantizer = RocmQLoraQuantizer(config)
    assert quantizer.is_serializable() is True

def test_quantizer_validate_environment_no_crash():
    config = RocmQLoraConfig()
    quantizer = RocmQLoraQuantizer(config)
    # This should run without crashing even if GPU is missing
    quantizer.validate_environment()

def test_process_before_loading_replaces_linears():
    config = RocmQLoraConfig(bits=8, target_modules=["q_proj"])
    quantizer = RocmQLoraQuantizer(config)
    model = FakeModel()
    
    # Before
    assert isinstance(model.q_proj, nn.Linear)
    
    # Process
    model = quantizer._process_model_before_weight_loading(model)
    
    # After (LoRALinear wraps QuantLinear)
    from rocm_qlora.lora.lora_layer import LoRALinear
    assert isinstance(model.q_proj, LoRALinear)
    # v_proj was not in target_modules, should still be nn.Linear (skipped by quantize_model)
    assert isinstance(model.v_proj, nn.Linear)

def test_process_after_loading_returns_model():
    config = RocmQLoraConfig()
    quantizer = RocmQLoraQuantizer(config)
    model = FakeModel()
    # Mocking patch_model_attention to avoid errors on fake model
    with mock.patch("rocm_qlora.hf_integration.quantizer.patch_model_attention"):
        res = quantizer._process_model_after_weight_loading(model)
    assert isinstance(res, nn.Module)
