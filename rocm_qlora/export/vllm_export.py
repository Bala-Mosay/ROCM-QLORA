"""
vLLM-compatible export for rocm-qlora fine-tuned models.

vLLM is the fastest LLM inference server with OpenAI-compatible API.
ROCm vLLM supports AMD MI300X with PagedAttention and continuous batching.

Two export strategies:

  Strategy A — Merged model (simpler deployment):
    Merge LoRA + dequantize → FP16 safetensors
    Load with: vllm serve {model_dir}
    Pro: single artifact, simple
    Con: loses LoRA flexibility, larger file

  Strategy B — LoRA adapter (recommended for flexibility):
    Save base model separately, save LoRA adapter in PEFT format
    Load with: vllm serve {base_model} --lora-modules name={adapter_dir}
    Pro: hot-swap adapters, smaller files, multi-LoRA serving
    Con: requires base model access at serve time

Strategy B is recommended — enables serving multiple fine-tuned variants
of the same base model with minimal memory overhead.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from rocm_qlora import LoRALinear
from rocm_qlora.export.gguf_export import _build_export_state_dict, _export_dequant_layer


def export_for_vllm_merged(
    model: nn.Module, 
    output_dir: str, 
    tokenizer: Optional[Any] = None,
    model_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export a rocm-qlora model for vLLM serving (merged weights).
    
    Args:
        model: A quantized model with LoRALinear layers
        output_dir: Directory to save the exported model
        tokenizer: Optional tokenizer to save alongside
        model_config: Optional dict to save as config.json
        
    Returns:
        Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reuse gguf_export's state dict builder
    model.eval()
    with torch.no_grad():
        state_dict = _build_export_state_dict(model)
        
        # Calculate total size and split if needed (>5GB for vLLM shard compatibility)
        total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
        print(f"[vllm_export] Total state dict size: {total_bytes / 1e9:.2f} GB")
        
        # Try safetensors first
        try:
            from safetensors.torch import save_file
            
            # Split into shards if > 5GB
            if total_bytes > 5 * 1e9:
                shard_size = 2 * 1e9  # 2GB shards
                current_shard = {}
                current_size = 0
                shard_idx = 1
                
                for key, tensor in state_dict.items():
                    tensor_size = tensor.numel() * tensor.element_size()
                    
                    if current_size + tensor_size > shard_size and current_shard:
                        # Save current shard
                        shard_path = os.path.join(output_dir, f"model-{shard_idx:05d}-of-?????.safetensors")
                        # For simplicity, save as single file if sharding not critical
                        save_file(current_shard, os.path.join(output_dir, f"model-{shard_idx:05d}.safetensors"))
                        shard_idx += 1
                        current_shard = {}
                        current_size = 0
                    
                    current_shard[key] = tensor
                    current_size += tensor_size
                
                if current_shard:
                    save_file(current_shard, os.path.join(output_dir, f"model-{shard_idx:05d}.safetensors"))
            else:
                # Single file
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                
        except ImportError:
            # Fallback to torch.save
            torch.save(state_dict, os.path.join(output_dir, "model.pt"))
            print(f"[vllm_export] WARNING: safetensors not installed, saved as .pt")
        
        # Save config.json if provided
        if model_config is not None:
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            print(f"[vllm_export] Saved config.json: {config_path}")
        
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            print(f"[vllm_export] Saved tokenizer: {output_dir}")
        
        print(f"[vllm_export] Merged export complete: {output_dir}")
    
    return output_dir


def _convert_to_peft_key(rocm_qlora_key: str) -> str:
    """
    Convert rocm-qlora key format to PEFT format.
    
    Args:
        rocm_qlora_key: e.g., 'model.layers.0.self_attn.q_proj.lora_A'
        
    Returns:
        PEFT key: e.g., 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
    """
    # PEFT format: 'base_model.model.{original_key}.weight'
    # The original key already has the full path, we prepend 'base_model.model.' 
    # and append '.weight'
    return f"base_model.model.{rocm_qlora_key}.weight"


def export_lora_adapter(
    model: nn.Module, 
    output_dir: str, 
    base_model_id: str, 
    lora_r: int = 8, 
    lora_alpha: int = 16, 
    target_modules: Optional[List[str]] = None
) -> str:
    """
    Export only the LoRA adapter weights in PEFT format for vLLM.
    
    Args:
        model: A quantized model with LoRALinear layers
        output_dir: Directory to save the adapter
        base_model_id: HuggingFace model ID of the base model
        lora_r: LoRA rank (used in config)
        lora_alpha: LoRA alpha scaling (used in config)
        target_modules: List of module names to apply LoRA to
        
    Returns:
        Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Extract only LoRA weights
    adapter_weights = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            # Convert to PEFT naming convention
            peft_key = _convert_to_peft_key(name)
            adapter_weights[peft_key] = param.data.cpu().to(torch.float16)
    
    # Save adapter weights
    try:
        from safetensors.torch import save_file
        save_file(adapter_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    except ImportError:
        torch.save(adapter_weights, os.path.join(output_dir, "adapter_model.bin"))
        print(f"[vllm_export] WARNING: safetensors not installed, saved as .bin")
    
    # Save adapter config.json with PEFT-compatible schema
    adapter_config = {
        "base_model_name_or_path": base_model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_r,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"[vllm_export] Adapter export complete: {output_dir}")
    print(f"[vllm_export] Saved adapter_config.json with peft_type=LORA")
    
    return output_dir


def get_vllm_serve_commands(
    base_model_id: str,
    adapter_dir: Optional[str] = None,
    merged_dir: Optional[str] = None,
    port: int = 8000,
    gpu_arch: str = "auto"
) -> str:
    """
    Returns formatted vLLM serve commands for the exported model.
    
    Args:
        base_model_id: HuggingFace model ID (used with adapter mode)
        adapter_dir: Path to exported LoRA adapter (Strategy B)
        merged_dir: Path to merged export (Strategy A)
        port: Port for the API server
        gpu_arch: GPU architecture (auto, gfx1100, gfx942)
        
    Returns:
        Formatted multi-line string with serve commands
    """
    # Build the commands
    commands = []
    
    commands.append("# Install vLLM with ROCm support:")
    commands.append("pip install vllm  # ROCm backend auto-detected")
    commands.append("")
    commands.append("# Required env vars for AMD GPU:")
    commands.append("# For gfx1100 (RX 7900 XTX):")
    commands.append("export HSA_OVERRIDE_GFX_VERSION=11.0.0")
    commands.append("# For gfx942 (MI300X):")
    commands.append("export HSA_OVERRIDE_GFX_VERSION=9.4.2")
    commands.append("")
    
    if merged_dir is not None:
        commands.append("# Strategy A — Merged model:")
        commands.append(f"vllm serve {merged_dir} --port {port} --dtype float16")
        commands.append("")
    
    if adapter_dir is not None:
        commands.append("# Strategy B — LoRA adapter (recommended):")
        commands.append(f"vllm serve {base_model_id} \\")
        commands.append(f"    --enable-lora \\")
        commands.append(f"    --lora-modules rocm-qlora-adapter={adapter_dir} \\")
        commands.append(f"    --port {port} \\")
        commands.append(f"    --dtype float16")
        commands.append("")
        commands.append("# Test the API:")
        commands.append(f'curl http://localhost:{port}/v1/chat/completions \\')
        commands.append(f'    -H "Content-Type: application/json" \\')
        commands.append(f'    -d \'{{"model": "rocm-qlora-adapter", "messages": [{{"role": "user", "content": "Hello"}}]}}\'')
        commands.append("")
    
    commands.append("# NOTE: HSA_OVERRIDE_GFX_VERSION is required for vLLM on consumer AMD GPUs")
    commands.append("# NOTE: --enable-lora flag required for Strategy B — vLLM disables LoRA by default")
    
    return "\n".join(commands)