"""
Model quantization logic for rocm-qlora.
Orchestrates the conversion of standard nn.Linear layers into LoRA-augmented quantized layers.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional

from rocm_qlora.quantization.quant_linear import QuantLinear
from rocm_qlora.lora.lora_layer import LoRALinear
from rocm_qlora.utils.rocm_utils import estimate_model_memory, print_model_summary

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rocm_qlora.quantize_model")

def _get_parent_and_attr(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """
    Traverses the module tree to find the parent module and the attribute name of a target.
    
    Args:
        model: The root nn.Module.
        full_name: Dotted name of the target module (e.g., 'layers.0.self_attn.q_proj').
        
    Returns:
        tuple: (parent_module, attribute_name)
    """
    parts = full_name.split(".")
    parent = model
    for i in range(len(parts) - 1):
        parent = getattr(parent, parts[i])
    return parent, parts[-1]

def quantize_model(
    model: nn.Module,
    bits: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "v_proj"],
    block_size: int = 64,
) -> nn.Module:
    """
    Replaces target Linear layers with LoRALinear(QuantLinear) layers.
    
    Args:
        model: The model to quantize.
        bits: Bit-depth (4 or 8).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
        target_modules: List of module names to target (e.g., 'q_proj', 'v_proj').
        block_size: Quantization block size.
        
    Returns:
        nn.Module: The modified model.
    """
    if bits not in [4, 8]:
        raise ValueError(f"Invalid bit-depth: {bits}. Only 4 and 8 bits are supported.")

    # NOTE: Freeze the entire model first. LoRALinear will then add trainable parameters.
    model.requires_grad_(False)

    # NOTE: two-pass replacement (collect then apply) prevents iteration mutation bugs in nn.Module tree.
    replacements: List[Tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if any target_modules name is a suffix or exact match of the module name
            if any(name.endswith(target) for target in target_modules):
                # Skip layers that are too small to quantize meaningfully
                if min(module.in_features, module.out_features) < 64:
                    logger.warning(
                        f"Skipping layer '{name}' because it is too small "
                        f"({module.in_features}x{module.out_features} < 64)"
                    )
                    continue
                
                parent, attr = _get_parent_and_attr(model, name)
                replacements.append((parent, attr, module))

    # Apply replacements
    for parent, child_attr, original_linear in replacements:
        # 1. Convert to QuantLinear (handles FP16 -> INT8/INT4 conversion and VRAM reclaim)
        quant_linear = QuantLinear.from_linear(
            original_linear, 
            bits=bits, 
            block_size=block_size
        )
        
        # 2. Wrap with LoRALinear (handles freezing and adapter initialization)
        lora_layer = LoRALinear(
            quant_linear, 
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout
        )
        
        # 3. Inject back into model
        # NOTE: setattr on parent module is the correct nn.Module API — do not use _modules dict directly.
        setattr(parent, child_attr, lora_layer)

    # Post-quantization summary
    print("\n--- Model Quantization Summary ---")
    print_model_summary(model)
    
    mem_stats = estimate_model_memory(model, bits)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_pct:.4f}%)")
    print(f"Estimated FP16 VRAM: {mem_stats['fp16_gb']:.4f} GB")
    print(f"Estimated Quantized VRAM: {mem_stats['quantized_gb']:.4f} GB")
    print(f"Reduction: {mem_stats['reduction_pct']:.2f}%")
    print("----------------------------------\n")

    return model

def enable_all_kernels(model: nn.Module) -> int:
    """
    Walks the model and enables Triton kernels for every QuantLinear layer found.
    Returns:
        int: Number of layers where kernels were successfully enabled.
    """
    enabled_count = 0
    for name, module in model.named_modules():
        # Check both direct QuantLinear and LoRALinear wrapped ones
        target = None
        if isinstance(module, QuantLinear):
            target = module
        elif hasattr(module, "base_layer") and isinstance(module.base_layer, QuantLinear):
            target = module.base_layer
            
        if target is not None:
            if target.enable_kernel():
                enabled_count += 1
                
    return enabled_count
