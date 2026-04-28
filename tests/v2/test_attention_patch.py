import pytest
import torch
import torch.nn as nn
import unittest.mock as mock
import sys
from rocm_qlora.attention.attention_patch import (
    detect_flash_attention, patch_model_attention,
    enable_sdpa_optimization, install_instructions
)

def mock_flash_attn_available():
    """Context manager that makes flash_attn appear importable."""
    fake_module = mock.MagicMock()
    fake_module.__version__ = "2.5.0"
    return mock.patch.dict('sys.modules', {'flash_attn': fake_module})

def mock_flash_attn_unavailable():
    """Context manager that makes flash_attn appear uninstalled."""
    return mock.patch.dict('sys.modules', {'flash_attn': None})

def test_detect_returns_dict_always():
    res = detect_flash_attention()
    assert isinstance(res, dict)

def test_detect_required_keys():
    res = detect_flash_attention()
    required = {"available", "version", "backend", "supports_backward", "arch", "triton_env_set"}
    assert all(k in res for k in required)

def test_detect_unavailable_when_not_installed():
    with mock_flash_attn_unavailable():
        res = detect_flash_attention()
        assert res["available"] is False

def test_detect_available_when_installed():
    with mock_flash_attn_available():
        res = detect_flash_attention()
        assert res["available"] is True
        assert res["version"] == "2.5.0"

def test_patch_returns_tuple():
    model = nn.Linear(4, 4)
    res = patch_model_attention(model)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], nn.Module)
    assert isinstance(res[1], dict)

def test_patch_info_has_required_keys():
    model = nn.Linear(4, 4)
    _, info = patch_model_attention(model)
    required = {"backend_chosen", "layers_patched", "warning", "install_hint"}
    assert all(k in info for k in required)

def test_patch_fallback_to_sdpa_when_unavailable():
    with mock_flash_attn_unavailable():
        model = nn.Linear(4, 4)
        _, info = patch_model_attention(model)
        assert "sdpa" in info["backend_chosen"].lower()

def test_patch_warns_on_rdna3_ck():
    # Mock arch=rdna3, FA2 available, but triton_env_set=False
    with mock_flash_attn_available():
        with mock.patch('rocm_qlora.attention.attention_patch.get_device_info') as mock_dev:
            mock_dev.return_value = {"arch": "rdna3", "triton_available": False}
            with mock.patch('os.environ.get', return_value="FALSE"):
                model = nn.Linear(4, 4)
                _, info = patch_model_attention(model)
                assert info["warning"] is not None
                assert "unsafe" in info["backend_chosen"].lower() or "sdpa" in info["backend_chosen"].lower()

def test_enable_sdpa_returns_model():
    model = nn.Linear(4, 4)
    res = enable_sdpa_optimization(model)
    assert isinstance(res, nn.Module)

def test_install_instructions_returns_string():
    res = install_instructions("cdna3")
    assert isinstance(res, str)
    assert len(res) > 0

def test_install_instructions_contains_triton_command():
    res = install_instructions("rdna3")
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE" in res

def test_no_crash_on_unknown_arch():
    with mock.patch('rocm_qlora.attention.attention_patch.get_device_info') as mock_dev:
        mock_dev.return_value = {"arch": "unknown", "triton_available": False}
        res = install_instructions()
        assert "unknown" in res.lower()
        
        model = nn.Linear(4, 4)
        patch_model_attention(model) # Should not crash
