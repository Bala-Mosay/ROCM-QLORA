import pytest
import torch
import torch.nn as nn
from unittest import mock
import os

from rocm_qlora.fp8 import (
    detect_fp8_support, FP8Config, get_fp8_dtype, get_fp8_recipe,
    FP8LinearWrapper, wrap_model_for_fp8, enable_fp8_autocast
)
from rocm_qlora.lora.lora_layer import LoRALinear
from rocm_qlora.quantization import QuantLinear

# Helper: simulate FP8-capable environment for testing the happy path
def mock_fp8_environment():
    """Mock an FP8-capable ROCm environment for testing the enabled path."""
    return mock.patch.multiple(
        'rocm_qlora.fp8.fp8_config',
        detect_fp8_support=mock.MagicMock(return_value={
            'fp8_supported': True,
            'rocm_version': '6.2.60201',
            'arch': 'cdna3',
            'fp8_dtypes_available': True,
            'transformer_engine_available': False,  # native path, no TE needed
            'reason': 'Supported',
        })
    )

# --- Tests ---

def test_detect_fp8_returns_dict():
    """detect_fp8_support() returns dict — no exception."""
    res = detect_fp8_support()
    assert isinstance(res, dict)

def test_detect_fp8_has_all_keys():
    """All required keys present in detection dict."""
    res = detect_fp8_support()
    keys = ["fp8_supported", "rocm_version", "arch", "fp8_dtypes_available", "transformer_engine_available", "reason"]
    for k in keys:
        assert k in res

def test_detect_fp8_no_crash_cpu():
    """Runs on CPU without exception."""
    detect_fp8_support()

def test_detect_fp8_reason_populated():
    """Reason field is a non-empty string."""
    res = detect_fp8_support()
    assert isinstance(res["reason"], str)
    assert len(res["reason"]) > 0

def test_detect_fp8_false_on_cpu():
    """On CPU (no ROCm 6.2), fp8_supported == False."""
    res = detect_fp8_support()
    # Assuming the current test environment is not an MI300X with ROCm 6.2
    if torch.version.hip is None or "gfx942" not in str(torch.cuda.get_device_properties(0).gcn_arch_name if torch.cuda.is_available() else ""):
        assert res["fp8_supported"] is False

def test_fp8_config_defaults():
    """FP8Config().forward_dtype == 'e4m3' and backward_dtype == 'e5m2'."""
    config = FP8Config()
    assert config.forward_dtype == "e4m3"
    assert config.backward_dtype == "e5m2"

def test_fp8_config_forward_is_e4m3():
    """Forward uses E4M3 (higher precision)."""
    assert FP8Config().forward_dtype == "e4m3"

def test_fp8_config_backward_is_e5m2():
    """Backward uses E5M2 (higher range)."""
    assert FP8Config().backward_dtype == "e5m2"

def test_get_fp8_recipe_none_without_te():
    """get_fp8_recipe(FP8Config()) returns None when TE unavailable."""
    with mock.patch.dict('sys.modules', {'transformer_engine': None}):
        res = get_fp8_recipe(FP8Config())
        assert res is None

def test_fp8_linear_wrapper_init_no_crash():
    """FP8LinearWrapper(nn.Linear(32,32)) — no exception."""
    ql = QuantLinear(32, 32)
    # Initialize buffers to prevent size-0 errors
    ql.weight_quant = torch.zeros(1024, dtype=torch.int8)
    ql.weight_scales = torch.ones(1024 // 64)
    base = LoRALinear(ql)
    FP8LinearWrapper(base)

def test_fp8_linear_wrapper_backend_bf16_on_cpu():
    """backend == 'bf16' on CPU fallback."""
    ql = QuantLinear(32, 32)
    ql.weight_quant = torch.zeros(1024, dtype=torch.int8)
    ql.weight_scales = torch.ones(1024 // 64)
    base = LoRALinear(ql)
    wrapper = FP8LinearWrapper(base)
    # On CPU/RDNA, backend must be bf16
    assert wrapper.backend == "bf16"

def test_fp8_linear_wrapper_forward_runs():
    """Forward with torch.randn(2, 32) — no exception."""
    ql = QuantLinear(32, 32)
    ql.weight_quant = torch.zeros(1024, dtype=torch.int8)
    ql.weight_scales = torch.ones(1024 // 64)
    base = LoRALinear(ql)
    wrapper = FP8LinearWrapper(base)
    x = torch.randn(2, 32)
    wrapper(x)

def test_fp8_linear_wrapper_output_shape():
    """Output shape [2, 32] for Linear(32,32)."""
    ql = QuantLinear(32, 32)
    ql.weight_quant = torch.zeros(1024, dtype=torch.int8)
    ql.weight_scales = torch.ones(1024 // 64)
    base = LoRALinear(ql)
    wrapper = FP8LinearWrapper(base)
    x = torch.randn(2, 32)
    out = wrapper(x)
    assert out.shape == (2, 32)

def test_fp8_wrapper_bf16_output_unchanged():
    """Output matches base_linear(x) within atol=1e-4 in fallback."""
    ql = QuantLinear(32, 32)
    ql.weight_quant = torch.zeros(1024, dtype=torch.int8)
    ql.weight_scales = torch.ones(1024 // 64)
    base = LoRALinear(ql)
    wrapper = FP8LinearWrapper(base)
    x = torch.randn(2, 32)
    out_wrapper = wrapper(x)
    out_base = base(x)
    assert torch.allclose(out_wrapper, out_base, atol=1e-4)

def test_wrap_model_returns_tuple():
    """wrap_model_for_fp8(nn.Linear(4,4)) returns (nn.Module, dict)."""
    # wrap_model looks for LoRALinear
    model = nn.Sequential(LoRALinear(QuantLinear(4, 4)))
    res, info = wrap_model_for_fp8(model)
    assert isinstance(res, nn.Module)
    assert isinstance(info, dict)

def test_wrap_model_info_has_keys():
    """Info dict has enabled, backend, layers_wrapped, reason."""
    model = nn.Sequential(LoRALinear(QuantLinear(4, 4)))
    _, info = wrap_model_for_fp8(model)
    for k in ["enabled", "backend", "layers_wrapped", "reason"]:
        assert k in info

def test_wrap_model_disabled_on_cpu():
    """wrap_info['enabled'] == False on CPU."""
    model = nn.Sequential(LoRALinear(QuantLinear(4, 4)))
    _, info = wrap_model_for_fp8(model)
    assert info["enabled"] is False

def test_wrap_model_reason_populated_when_disabled():
    """Reason is non-empty string when disabled."""
    model = nn.Sequential(LoRALinear(QuantLinear(4, 4)))
    _, info = wrap_model_for_fp8(model)
    assert len(info["reason"]) > 0

def test_enable_fp8_autocast_context_manager():
    """enable_fp8_autocast(nn.Linear(4,4)) returns context manager."""
    ctx = enable_fp8_autocast(nn.Linear(4, 4))
    assert hasattr(ctx, "__enter__")
    assert hasattr(ctx, "__exit__")

def test_enable_fp8_autocast_safe_on_cpu():
    """Entering context manager on CPU does not raise."""
    with enable_fp8_autocast(nn.Linear(4, 4)):
        pass

def test_env_vars_not_set_in_fallback():
    """TORCH_NCCL_HIGH_PRIORITY not set when FP8 falls back to BF16."""
    if "TORCH_NCCL_HIGH_PRIORITY" in os.environ:
        del os.environ["TORCH_NCCL_HIGH_PRIORITY"]
    model = nn.Sequential(LoRALinear(QuantLinear(4, 4)))
    wrap_model_for_fp8(model)
    assert "TORCH_NCCL_HIGH_PRIORITY" not in os.environ
