"""
torch.compile integration with ROCm-safe fallback.
torch.compile with max-autotune on ROCm fuses the dequant+matmul
into a single TorchInductor/Triton kernel automatically.
"""
import torch
import torch.nn as nn
import logging
import time
from typing import Optional

from rocm_qlora.lora.lora_layer import LoRALinear

logger = logging.getLogger("rocm_qlora.compile_utils")

def is_compile_safe() -> bool:
    """
    Check ROCm version via torch.version.hip.
    Returns True only if ROCm >= 6.1 (earlier versions have compile instability).
    """
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        return False
    
    try:
        # Example: '6.0.2-12345'
        version_str = torch.version.hip.split('-')[0]
        major, minor = map(int, version_str.split('.')[:2])
        if major > 6:
            return True
        if major == 6 and minor >= 1:
            return True
    except Exception:
        pass
    
    return False

def compile_model(
    model: nn.Module, 
    mode: str = "max-autotune", 
    dynamic: bool = False
) -> nn.Module:
    """
    Try torch.compile(model) with fallback on failure.
    """
    if not is_compile_safe():
        logger.warning("ROCm version < 6.1 detected. Skipping torch.compile for stability.")
        return model

    try:
        logger.info(f"Compiling model with mode={mode}, dynamic={dynamic}...")
        compiled_model = torch.compile(model, mode=mode, dynamic=dynamic, backend="inductor")
        return compiled_model
    except (RuntimeError, Exception) as e:
        logger.warning(f"torch.compile failed on ROCm — falling back to eager mode. Error: {e}")
        return model

def compile_lora_only(model: nn.Module) -> nn.Module:
    """
    Only compile LoRALinear modules, not the full model.
    Safer than full model compile — smaller graph, less likely to hit ROCm compile bugs.
    """
    if not is_compile_safe():
        logger.warning("ROCm version < 6.1 detected. Skipping partial compile.")
        return model

    logger.info("Performing partial compile on LoRALinear modules...")
    compiled_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compile individual LoRA layers
            # Use 'default' mode for individual layers to avoid heavy autotuning overhead per layer
            setattr(module, "forward", torch.compile(module.forward, mode="reduce-overhead", backend="inductor"))
            compiled_count += 1
    
    logger.info(f"Successfully compiled {compiled_count} LoRA modules.")
    return model

def warmup_compiled_model(model: nn.Module, device: str = None, seq_len: int = 512, batch_size: int = 1):
    """
    Run warmup forward passes to trigger TorchInductor compilation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting warmup for compiled model (seq_len={seq_len}, device={device})...")
    model.eval()
    
    # We need to know the hidden_size. We can try to infer it from the model.
    # For LLMs, it's often model.config.hidden_size or similar, but we'll try to find a Linear layer.
    hidden_size = None
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, LoRALinear):
            hidden_size = m.in_features
            break
            
    if hidden_size is None:
        logger.warning("Could not infer hidden_size for warmup. Skipping.")
        return

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    
    start_time = time.time()
    with torch.no_grad():
        # First pass compiles
        logger.info("Warmup Pass 1 (Compiling)...")
        model(dummy_input)
        
        # Second and third verify stability
        logger.info("Warmup Pass 2...")
        model(dummy_input)
        logger.info("Warmup Pass 3...")
        model(dummy_input)
        
    end_time = time.time()
    logger.info(f"Warmup complete. Total time: {end_time - start_time:.2f}s")
    model.train()
