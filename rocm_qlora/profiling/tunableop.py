"""
TunableOp GEMM autotuning for ROCm.
"""
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def enable_tunableop(output_dir: str = "./tunableop_cache", tuning: bool = True) -> Dict[str, Any]:
    """
    Enables TunableOp environment variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "gemm_table.csv")
    
    os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
    os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "1" if tuning else "0")
    os.environ.setdefault("PYTORCH_TUNABLEOP_FILENAME", cache_path)
    os.environ.setdefault("PYTORCH_TUNABLEOP_VERBOSE", "0")
    
    from rocm_qlora.utils.compile_utils import is_compile_safe
    compile_warning = None
    if is_compile_safe():
        compile_warning = (
            "torch.compile is available on this system. "
            "Using TunableOp and torch.compile simultaneously may conflict. "
            "Recommendation: use --use_torch_compile=False when using TunableOp."
        )
        
    return {
        "enabled": True, 
        "tuning": tuning, 
        "cache_path": cache_path, 
        "compile_warning": compile_warning
    }

def disable_tunableop() -> Dict[str, Any]:
    """Disables TunableOp."""
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    return {"enabled": False}

def switch_to_load_only() -> Dict[str, Any]:
    """Stops active tuning and switches to loading the existing cache."""
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    return {
        "enabled": os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1",
        "tuning": False,
        "cache_path": os.environ.get("PYTORCH_TUNABLEOP_FILENAME", "not set")
    }

def load_tunableop_cache(cache_path: str) -> Dict[str, Any]:
    """Loads an existing TunableOp cache file."""
    if not os.path.exists(cache_path):
        return {"loaded": False, "num_entries": 0, "cache_path": cache_path}
    
    # Count entries (exclude header)
    try:
        with open(cache_path, "r") as f:
            lines = f.readlines()
            num_entries = max(0, len(lines) - 1)
    except:
        num_entries = 0
        
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = cache_path
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    
    return {"loaded": True, "num_entries": num_entries, "cache_path": cache_path}

def tunableop_warmup(
    model: nn.Module, 
    device: str, 
    seq_len: int = 512, 
    num_steps: int = 5, 
    batch_size: int = 1
) -> Dict[str, Any]:
    """Runs dummy forward passes to trigger GEMM tuning."""
    if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") != "1":
        print("[tunableop] WARNING: TunableOp is not enabled during warmup.")
        
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    times = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
            
    first = times[0]
    last = times[-1]
    speedup = first / last if last > 0 else 1.0
    
    return {
        "warmup_steps": num_steps,
        "step_times_ms": times,
        "first_step_ms": first,
        "last_step_ms": last,
        "speedup_after_warmup": speedup
    }

def get_tunableop_status() -> Dict[str, Any]:
    """Returns current status of TunableOp environment."""
    cache_path = os.environ.get("PYTORCH_TUNABLEOP_FILENAME", "not set")
    exists = os.path.exists(cache_path) if cache_path != "not set" else False
    num_cached = 0
    if exists:
        try:
            with open(cache_path, "r") as f:
                num_cached = max(0, len(f.readlines()) - 1)
        except:
            pass
            
    return {
        "enabled": os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0") == "1",
        "tuning": os.environ.get("PYTORCH_TUNABLEOP_TUNING", "0") == "1",
        "cache_path": cache_path,
        "cache_exists": exists,
        "num_cached_shapes": num_cached
    }

def estimate_tunableop_benefit(model: nn.Module) -> Dict[str, Any]:
    """Estimates tuning time based on unique GEMM shapes."""
    unique_shapes = set()
    from rocm_qlora.quantization.quant_linear import QuantLinear
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, QuantLinear)):
            unique_shapes.add((module.out_features, module.in_features))
            
    n = len(unique_shapes)
    # Estimate ~3s per unique shape on MI300X
    est_time = (n * 3) / 60
    
    return {
        "unique_gemm_shapes": n,
        "estimated_tune_time_minutes": est_time,
        "expected_speedup_range": "10-30% on MI300X"
    }
