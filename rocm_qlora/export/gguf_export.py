"""
GGUF export for rocm-qlora fine-tuned models.

GGUF is the file format used by llama.cpp, Ollama, and LM Studio.
Exporting enables deployment on CPU, mobile, and edge devices without
requiring ROCm or any GPU at inference time.

Export pipeline (pure Python — no llama.cpp required for steps 1-3):
  Step 1: Load model + LoRA weights
  Step 2: Dequantize + merge in FP16 (using clean export path, not merge_lora())
  Step 3: Save as safetensors or .pt (HF-compatible)
  Step 4: Convert to GGUF with llama.cpp (external tool, instructions provided)

Why not use merge_lora():
  merge_lora() re-quantizes after merging, introducing atol~2.0 error.
  Export needs full FP16 precision for GGUF to then apply its own quantization.
  We dequantize directly and add LoRA delta in FP16 — no re-quantization.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from rocm_qlora import LoRALinear, QuantLinear
from rocm_qlora.quantization.quant_ops import dequantize_int8, dequantize_int4
from rocm_qlora.quantization.double_quant import DoubleQuantState, double_dequantize


def _export_dequant_layer(lora_linear: nn.Module) -> torch.Tensor:
    """
    Dequantize base + add LoRA delta directly in FP16. No re-quantize.
    
    This is the clean export path that bypasses merge_lora() to avoid
    the atol~2.0 re-quantization error documented in V1.
    
    Args:
        lora_linear: A LoRALinear layer to export
        
    Returns:
        Merged FP16 weight tensor of shape (out_features, in_features)
    """
    if not isinstance(lora_linear, LoRALinear):
        raise TypeError(f"Expected LoRALinear, got {type(lora_linear)}")
    
    quant = lora_linear.base  # QuantLinear
    bits = quant.bits
    
    # Step 1: Dequantize base weights to FP16
    if quant.use_double_quant:
        # Handle double-quantized layers
        state = DoubleQuantState(
            W_quant=quant.weight_quant,
            c2=quant.c2,
            c2_scales=quant.c2_scales,
            blocksize_1=quant.block_size,
            blocksize_2=quant.blocksize_2,
            original_shape=quant.original_shape,
            use_fp8_c2=quant.use_fp8_c2,
        )
        W = double_dequantize(state)
    elif bits == 8:
        W = dequantize_int8(quant.weight_quant, quant.weight_scales, quant.block_size)
    else:
        # NF4 (bits == 4)
        W = dequantize_int4(quant.weight_quant, quant.weight_scales, quant.block_size)
    
    # Reshape to original layout [Out, In]
    W = W.reshape(quant.original_shape).to(torch.float16)
    
    # Step 2: Add LoRA delta in FP16 (no re-quantization)
    # delta = (B @ A) * scaling
    delta_W = (lora_linear.lora_B @ lora_linear.lora_A) * lora_linear.scaling
    delta_W = delta_W.to(torch.float16)
    
    # Add delta to base weights
    merged_W = W + delta_W
    
    return merged_W


def _build_export_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Build a HuggingFace-compatible state dict with merged LoRA weights.
    
    Walks model.named_modules() and for each LoRALinear:
      - Dequantizes base + adds LoRA delta in FP16
      - Stores with the original key (e.g., 'model.layers.0.self_attn.q_proj.weight')
    
    For regular nn.Linear (non-target layers): copies weight as-is in FP16
    For all other tensors (LayerNorm, embeddings, etc.): copies as-is
    
    Args:
        model: A quantized model with LoRALinear layers
        
    Returns:
        State dict with FP16 weights, no LoRA-specific keys
    """
    state_dict: Dict[str, torch.Tensor] = {}
    
    # First pass: collect all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            state_dict[name] = param.data.cpu().to(torch.float16)
    
    # Second pass: handle LoRALinear layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get the base layer key (without lora_A/lora_B suffix)
            base_key = name  # e.g., 'model.layers.0.self_attn.q_proj'
            
            # Compute merged FP16 weight
            merged_weight = _export_dequant_layer(module)
            
            # Store with weight suffix to match HF format
            weight_key = f"{base_key}.weight"
            state_dict[weight_key] = merged_weight
            
            # Handle bias if present
            if module.base.bias is not None:
                bias_key = f"{base_key}.bias"
                state_dict[bias_key] = module.base.bias.data.cpu().to(torch.float16)
    
    return state_dict


def merge_and_export_fp16(
    model: nn.Module, 
    output_path: str, 
    model_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export a rocm-qlora model to FP16 safetensors (or .pt fallback).
    
    Args:
        model: A quantized model with LoRALinear layers
        output_path: Path to save the exported weights
        model_config: Optional dict to save as config.json alongside
        
    Returns:
        Path to the saved file
    """
    # Guard: model.eval() + torch.no_grad() around all operations
    model.eval()
    with torch.no_grad():
        # Build export state dict
        state_dict = _build_export_state_dict(model)
        
        # Print memory stats
        total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
        print(f"[export] Total state dict size: {total_bytes / 1e9:.2f} GB")
        
        # Determine file format based on extension
        is_safetensors = output_path.endswith('.safetensors')
        
        if is_safetensors:
            # Try safetensors first
            try:
                from safetensors.torch import save_file
                save_file(state_dict, output_path)
                print(f"[export] Saved to safetensors: {output_path}")
            except ImportError:
                # Fallback to torch.save if safetensors not installed
                torch.save(state_dict, output_path)
                print(f"[export] WARNING: safetensors not installed, saved as .pt: {output_path}")
                print(f"[export] Install safetensors for faster loads: pip install safetensors")
        else:
            # .pt file - use torch.save
            torch.save(state_dict, output_path)
            print(f"[export] Saved to torch .pt: {output_path}")
        
        # Save config.json if provided
        if model_config is not None:
            config_path = os.path.join(os.path.dirname(output_path), "config.json")
            import json
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            print(f"[export] Saved config.json: {config_path}")
        
        # Print memory after
        print(f"[export] Export complete: {output_path}")
    
    return output_path


def get_gguf_conversion_instructions(fp16_path: str, model_type: str = "llama") -> str:
    """
    Returns formatted instructions for converting FP16 to GGUF format.
    
    Args:
        fp16_path: Path to the exported FP16 safetensors/pt file
        model_type: Model architecture (llama, mistral, etc.)
        
    Returns:
        Formatted multi-line string with conversion instructions
    """
    return f"""# Step 1: Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Step 2: Convert to GGUF (Q4_K_M = best quality/size tradeoff)
python convert_hf_to_gguf.py {fp16_path} --outtype q4_k_m --outfile model.gguf

# Step 3a: Run with llama.cpp
./llama-cli -m model.gguf -p "Your prompt here" -n 200

# Step 3b: Run with Ollama
# Create Modelfile with: FROM ./model.gguf
ollama create my-rocm-qlora-model -f Modelfile
ollama run my-rocm-qlora-model

# NOTE: Q4_K_M is recommended — better quality than Q4_0, similar size"""


def estimate_gguf_size(num_params: int, quant_type: str = "q4_k_m") -> Dict[str, Any]:
    """
    Estimate GGUF file size for a given parameter count and quantization type.
    
    Args:
        num_params: Total number of parameters in the model
        quant_type: GGUF quantization type (q2_k, q3_k_m, q4_0, q4_k_m, q5_k_m, q6_k, q8_0, f16, f32)
        
    Returns:
        Dict with num_params, quant_type, estimated_size_gb, bits_per_param
    """
    bits_map = {
        "q2_k": 2.6,
        "q3_k_m": 3.35,
        "q4_0": 4.0,
        "q4_k_m": 4.5,
        "q5_k_m": 5.5,
        "q6_k": 6.0,
        "q8_0": 8.0,
        "f16": 16.0,
        "f32": 32.0,
    }
    
    bits_per_param = bits_map.get(quant_type, 4.5)
    estimated_size_gb = (num_params * bits_per_param) / 8 / 1e9
    
    # GGUF has ~5% metadata overhead
    estimated_size_gb *= 1.05
    
    return {
        "num_params": num_params,
        "quant_type": quant_type,
        "estimated_size_gb": round(estimated_size_gb, 2),
        "bits_per_param": bits_per_param,
    }