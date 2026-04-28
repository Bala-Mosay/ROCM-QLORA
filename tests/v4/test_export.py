"""
Export tests for rocm-qlora V4 Phase 4.

CPU-only tests - no GPU required, no safetensors required (fallback to .pt).
"""

import pytest
import os
import json
import torch
import torch.nn as nn
import tempfile

from rocm_qlora import quantize_model, LoRALinear, QuantLinear
from rocm_qlora.export import (
    merge_and_export_fp16,
    get_gguf_conversion_instructions,
    estimate_gguf_size,
    export_for_vllm_merged,
    export_lora_adapter,
    get_vllm_serve_commands,
)


@pytest.fixture
def quantized_model():
    """Create a simple quantized model for testing."""
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)
        
        def forward(self, x):
            return self.norm(self.q_proj(x))
    
    model = FakeTransformer()
    return quantize_model(model, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])


class TestGGUFExport:
    """GGUF export tests."""
    
    def test_merge_and_export_creates_file(self, quantized_model, tmp_path):
        """Test that merge_and_export_fp16 creates a file."""
        output_path = tmp_path / "out.pt"
        result = merge_and_export_fp16(quantized_model, str(output_path))
        assert os.path.exists(result), "Export file not created"
    
    def test_merge_and_export_returns_path(self, quantized_model, tmp_path):
        """Test that merge_and_export_fp16 returns a valid path."""
        output_path = tmp_path / "out.pt"
        result = merge_and_export_fp16(quantized_model, str(output_path))
        assert isinstance(result, str), "Return value should be a string path"
        assert result == str(output_path), "Return path should match input path"
    
    def test_exported_state_dict_no_lora_keys(self, quantized_model, tmp_path):
        """Test that exported state dict has no lora_A or lora_B keys."""
        output_path = tmp_path / "out.pt"
        merge_and_export_fp16(quantized_model, str(output_path))
        
        # Use weights_only=False for PyTorch 2.6+ compatibility
        state = torch.load(output_path, map_location="cpu", weights_only=False)
        for key in state.keys():
            assert 'lora_A' not in key, f"LoRA key leaked: {key}"
            assert 'lora_B' not in key, f"LoRA key leaked: {key}"
    
    def test_exported_weights_are_fp16(self, quantized_model, tmp_path):
        """Test that all weight tensors in saved file are torch.float16."""
        output_path = tmp_path / "out.pt"
        merge_and_export_fp16(quantized_model, str(output_path))
        
        # Use weights_only=False for PyTorch 2.6+ compatibility
        state = torch.load(output_path, map_location="cpu", weights_only=False)
        for key, tensor in state.items():
            assert tensor.dtype == torch.float16, f"{key} is not FP16: {tensor.dtype}"
    
    def test_exported_layer_shape_correct(self, quantized_model, tmp_path):
        """Test that q_proj weight shape is (64, 64) — matches original linear."""
        output_path = tmp_path / "out.pt"
        merge_and_export_fp16(quantized_model, str(output_path))
        
        # Use weights_only=False for PyTorch 2.6+ compatibility
        state = torch.load(output_path, map_location="cpu", weights_only=False)
        # Find q_proj weight
        q_proj_key = None
        for key in state.keys():
            if 'q_proj' in key and 'weight' in key:
                q_proj_key = key
                break
        
        assert q_proj_key is not None, "q_proj weight not found in state dict"
        assert state[q_proj_key].shape == (64, 64), f"Wrong shape: {state[q_proj_key].shape}"
    
    def test_export_no_reuse_merge_lora(self, quantized_model, tmp_path):
        """Verify LoRALinear.merged is still False after export."""
        output_path = tmp_path / "out.pt"
        merge_and_export_fp16(quantized_model, str(output_path))
        
        # Check that no LoRALinear was merged
        for name, module in quantized_model.named_modules():
            if isinstance(module, LoRALinear):
                assert not module.merged, f"LoRALinear at {name} was merged during export"
    
    def test_gguf_instructions_contains_llama_cpp(self, tmp_path):
        """Test that get_gguf_conversion_instructions contains llama.cpp."""
        output_path = tmp_path / "out.pt"
        instructions = get_gguf_conversion_instructions(str(output_path))
        assert "llama.cpp" in instructions, "Instructions should mention llama.cpp"
    
    def test_gguf_instructions_contains_q4_k_m(self, tmp_path):
        """Test that instructions contain q4_k_m recommendation."""
        output_path = tmp_path / "out.pt"
        instructions = get_gguf_conversion_instructions(str(output_path))
        assert "q4_k_m" in instructions, "Instructions should recommend q4_k_m"
    
    def test_estimate_gguf_size_returns_dict(self):
        """Test that estimate_gguf_size returns dict with required keys."""
        result = estimate_gguf_size(num_params=7_000_000_000, quant_type="q4_k_m")
        
        assert isinstance(result, dict), "Should return a dict"
        assert "estimated_size_gb" in result, "Missing estimated_size_gb"
        assert "bits_per_param" in result, "Missing bits_per_param"
        assert "quant_type" in result, "Missing quant_type"
    
    def test_estimate_gguf_size_f16_larger_than_q4(self):
        """Test that f16 estimate > q4_k_m estimate for same param count."""
        f16_result = estimate_gguf_size(num_params=7_000_000_000, quant_type="f16")
        q4_result = estimate_gguf_size(num_params=7_000_000_000, quant_type="q4_k_m")
        
        assert f16_result["estimated_size_gb"] > q4_result["estimated_size_gb"], \
            "f16 should be larger than q4_k_m"


class TestVLLMExport:
    """vLLM export tests."""
    
    def test_export_lora_creates_dir(self, quantized_model, tmp_path):
        """Test that export_lora_adapter creates output directory."""
        adapter_dir = tmp_path / "adapter"
        result = export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id")
        
        assert os.path.isdir(result), "Adapter directory not created"
    
    def test_export_lora_creates_adapter_config(self, quantized_model, tmp_path):
        """Test that adapter_config.json exists in output dir."""
        adapter_dir = tmp_path / "adapter"
        export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id")
        
        config_path = adapter_dir / "adapter_config.json"
        assert os.path.exists(config_path), "adapter_config.json not created"
    
    def test_adapter_config_peft_schema(self, quantized_model, tmp_path):
        """Test that config has peft_type, r, lora_alpha, base_model_name_or_path, target_modules."""
        adapter_dir = tmp_path / "adapter"
        export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id", lora_r=8, lora_alpha=16)
        
        config_path = adapter_dir / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        required_keys = ["peft_type", "r", "lora_alpha", "base_model_name_or_path", "target_modules"]
        for key in required_keys:
            assert key in config, f"Missing key in config: {key}"
    
    def test_adapter_config_peft_type_lora(self, quantized_model, tmp_path):
        """Test that config['peft_type'] == 'LORA'."""
        adapter_dir = tmp_path / "adapter"
        export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id")
        
        config_path = adapter_dir / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["peft_type"] == "LORA", f"peft_type should be LORA, got {config['peft_type']}"
    
    def test_adapter_config_base_model_id(self, quantized_model, tmp_path):
        """Test that config['base_model_name_or_path'] == 'base-model-id'."""
        adapter_dir = tmp_path / "adapter"
        export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id")
        
        config_path = adapter_dir / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["base_model_name_or_path"] == "base-model-id"
    
    def test_adapter_weights_only_lora_keys(self, quantized_model, tmp_path):
        """Test that saved weights contain only lora-related keys."""
        adapter_dir = tmp_path / "adapter"
        export_lora_adapter(quantized_model, str(adapter_dir), "base-model-id")
        
        # Find the weights file
        weights_file = adapter_dir / "adapter_model.safetensors"
        if not weights_file.exists():
            weights_file = adapter_dir / "adapter_model.bin"
        
        if weights_file.suffix == '.safetensors':
            from safetensors.torch import load_file
            weights = load_file(str(weights_file))
        else:
            weights = torch.load(weights_file, map_location="cpu")
        
        for key in weights.keys():
            assert 'lora_A' in key or 'lora_B' in key, f"Non-LoRA key in adapter: {key}"
    
    def test_vllm_serve_commands_contains_enable_lora(self):
        """Test that get_vllm_serve_commands contains --enable-lora."""
        commands = get_vllm_serve_commands("meta-llama/Llama-3-8B", adapter_dir="./adapter")
        assert "--enable-lora" in commands, "Commands should contain --enable-lora"
    
    def test_vllm_serve_commands_contains_hsa_var(self):
        """Test that commands contain HSA_OVERRIDE_GFX_VERSION."""
        commands = get_vllm_serve_commands("meta-llama/Llama-3-8B", adapter_dir="./adapter")
        assert "HSA_OVERRIDE_GFX_VERSION" in commands, "Commands should contain HSA_OVERRIDE_GFX_VERSION"
    
    def test_export_merged_creates_dir(self, quantized_model, tmp_path):
        """Test that export_for_vllm_merged creates output directory."""
        merged_dir = tmp_path / "merged"
        result = export_for_vllm_merged(quantized_model, str(merged_dir))
        
        assert os.path.isdir(result), "Merged directory not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])