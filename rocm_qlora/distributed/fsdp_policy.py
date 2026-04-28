"""
FSDP wrapping policy for QuantLinear + LoRALinear models on ROCm.

Architecture constraint:
  QuantLinear weights are registered buffers — FSDP replicates them.
  LoRALinear lora_A/lora_B are parameters — FSDP shards these.

  For QLoRA: quantized buffers (INT8) are already 50% of FP16 size.
  With 4 GPUs, each GPU holds: full INT8 buffers + sharded LoRA params.
  This is memory-efficient enough for practical use without buffer sharding.

  True buffer sharding requires custom flatten/unflatten FSDP hooks,
  which are experimental and unstable on ROCm 6.x. Not implemented here.
"""
import torch
import torch.nn as nn
import functools
import logging
from typing import List, Optional, Type
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

logger = logging.getLogger("rocm_qlora.distributed.fsdp_policy")

def get_qlora_fsdp_policy(transformer_layer_class: Type[nn.Module]) -> callable:
    """
    Returns an FSDP auto-wrap policy that wraps at the transformer layer level.
    """
    return functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls={transformer_layer_class}
    )

def prepare_model_for_fsdp(
    model: nn.Module, 
    transformer_layer_class: Type[nn.Module], 
    mixed_precision_dtype: torch.dtype = torch.float16
) -> nn.Module:
    """
    Wraps the model in FSDP with optimized settings for ROCm QLoRA.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Please launch with: torchrun --nproc_per_node=N script.py"
        )

    # Build MixedPrecision policy
    # NOTE: QuantLinear buffers (INT8/NF4) are kept as-is via buffer_dtype.
    mp = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        buffer_dtype=torch.float16, # Fallback for non-quantized buffers
    )

    # Wrap in FSDP
    # sync_module_states=True ensures all ranks start with identical weights 
    # (quantization must be deterministic across ranks).
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=get_qlora_fsdp_policy(transformer_layer_class),
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
    )
    
    return fsdp_model

def get_trainable_fsdp_params(fsdp_model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Extracts trainable parameters (LoRA) from an FSDP-wrapped model.
    """
    trainable_params = []
    for name, param in fsdp_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            
    if not trainable_params:
        logger.warning("No trainable parameters found in FSDP model.")
        
    return trainable_params

def save_fsdp_lora_weights(fsdp_model: nn.Module, output_path: str, rank: int):
    """
    Gathers sharded LoRA parameters from all ranks and saves them from rank 0.
    """
    # summon_full_params must be called on ALL ranks
    with FSDP.summon_full_params(fsdp_model, writeback=False):
        if rank == 0:
            lora_state = {
                name: param.data.clone()
                for name, param in fsdp_model.named_parameters()
                if param.requires_grad
            }
            torch.save(lora_state, output_path)

def get_transformer_layer_class(model: nn.Module) -> Type[nn.Module]:
    """
    Utility to auto-detect the transformer layer class for FSDP wrapping.
    """
    model_type = getattr(model.config, 'model_type', '').lower()
    
    # Common HuggingFace model families
    layer_map = {
        'llama': 'LlamaDecoderLayer',
        'mistral': 'MistralDecoderLayer',
        'qwen2': 'Qwen2DecoderLayer',
        'phi': 'PhiDecoderLayer',
        'gemma': 'GemmaDecoderLayer',
    }
    
    target_class_name = layer_map.get(model_type)
    
    if target_class_name:
        for module in model.modules():
            if module.__class__.__name__ == target_class_name:
                return module.__class__
                
    # Fallback: return the first child of the first sequential/decoder-like module
    logger.warning(f"Could not auto-detect transformer layer class for model_type '{model_type}'. Falling back to heuristic.")
    
    # Most decoder models have a list of layers as the second module after the root
    modules = list(model.modules())
    if len(modules) > 1:
        return modules[1].__class__
        
    return type(modules[0])
