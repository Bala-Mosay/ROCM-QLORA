import pytest
import os
import torch
import torch.nn as nn
import tempfile
import shutil
from rocm_qlora.profiling import (
    enable_tunableop, disable_tunableop, switch_to_load_only,
    load_tunableop_cache, get_tunableop_status,
    estimate_tunableop_benefit, ROCmProfiler,
)
from rocm_qlora import quantize_model

@pytest.fixture(autouse=True)
def clean_env():
    """Reset TunableOp env vars before each test."""
    keys = ["PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING",
            "PYTORCH_TUNABLEOP_FILENAME", "PYTORCH_TUNABLEOP_VERBOSE"]
    original = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)

# --- TunableOp Tests ---

def test_enable_tunableop_sets_enabled_env():
    """after call, PYTORCH_TUNABLEOP_ENABLED == '1'."""
    enable_tunableop()
    assert os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1"

def test_enable_tunableop_sets_tuning_env():
    """after call with tuning=True, PYTORCH_TUNABLEOP_TUNING == '1'."""
    enable_tunableop(tuning=True)
    assert os.environ.get("PYTORCH_TUNABLEOP_TUNING") == "1"

def test_enable_tunableop_creates_dir():
    """directory is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        enable_tunableop(output_dir=tmpdir)
        assert os.path.exists(tmpdir)

def test_enable_tunableop_returns_dict_keys():
    """has enabled, tuning, cache_path, compile_warning."""
    res = enable_tunableop()
    for k in ["enabled", "tuning", "cache_path", "compile_warning"]:
        assert k in res

def test_enable_tunableop_uses_setdefault():
    """pre-set env var not overwritten by enable_tunableop."""
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "1"
    enable_tunableop()
    assert os.environ.get("PYTORCH_TUNABLEOP_VERBOSE") == "1"

def test_disable_tunableop_sets_zero():
    """after disable, PYTORCH_TUNABLEOP_ENABLED == '0'."""
    enable_tunableop()
    disable_tunableop()
    assert os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "0"

def test_switch_to_load_only_keeps_enabled():
    """PYTORCH_TUNABLEOP_ENABLED stays '1', TUNING becomes '0'."""
    enable_tunableop(tuning=True)
    switch_to_load_only()
    assert os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1"
    assert os.environ.get("PYTORCH_TUNABLEOP_TUNING") == "0"

def test_load_nonexistent_cache():
    """load_tunableop_cache returns {loaded: False} without crash."""
    res = load_tunableop_cache("/nonexistent/path/gemm.csv")
    assert res["loaded"] is False

def test_load_existing_cache():
    """create temp CSV with 3 lines -> num_entries == 2."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("header\nline1\nline2\n")
        path = f.name
    try:
        res = load_tunableop_cache(path)
        assert res["loaded"] is True
        assert res["num_entries"] == 2
    finally:
        os.remove(path)

def test_get_status_keys():
    """returned dict has required keys."""
    status = get_tunableop_status()
    for k in ["enabled", "tuning", "cache_path", "cache_exists", "num_cached_shapes"]:
        assert k in status

def test_get_status_after_enable():
    """after enable_tunableop, status['enabled'] == True."""
    enable_tunableop()
    status = get_tunableop_status()
    assert status["enabled"] is True

def test_estimate_benefit_returns_dict():
    """returns dict with unique_gemm_shapes."""
    model = FakeTransformer()
    res = estimate_tunableop_benefit(model)
    assert isinstance(res, dict)
    assert "unique_gemm_shapes" in res

def test_estimate_benefit_nonzero_shapes():
    """unique_gemm_shapes > 0 for a model with linear layers."""
    model = FakeTransformer()
    res = estimate_tunableop_benefit(model)
    assert res["unique_gemm_shapes"] > 0

# --- ROCmProfiler Tests ---

def test_profiler_init_no_crash():
    """ROCmProfiler() instantiates without error."""
    ROCmProfiler()

def test_profiler_summary_returns_dict():
    """profile_model_summary returns dict with total_params."""
    model = FakeTransformer()
    profiler = ROCmProfiler()
    res = profiler.profile_model_summary(model, (1, 8))
    assert "total_params" in res
    assert res["total_params"] > 0

def test_profiler_summary_trainable_params_nonzero():
    """after quantize_model, trainable_params > 0."""
    model = FakeTransformer()
    model = quantize_model(model, bits=4)
    profiler = ROCmProfiler()
    res = profiler.profile_model_summary(model, (1, 8))
    assert res["trainable_params"] > 0

def test_get_rocprofiler_status_keys():
    """has available, version, install_command."""
    profiler = ROCmProfiler()
    status = profiler.get_rocprofiler_status()
    for k in ["available", "version", "install_command"]:
        assert k in status
