"""
Model export utilities for rocm-qlora.

Closes the training → deployment loop:
  rocm-qlora (train on AMD) → GGUF (Ollama/llama.cpp) or vLLM (API server)

Quick reference:
  # GGUF export (for Ollama/LM Studio):
  merge_and_export_fp16(model, "./merged_fp16.safetensors")
  print(get_gguf_conversion_instructions("./merged_fp16.safetensors"))

  # vLLM export (for API serving — Strategy B recommended):
  export_lora_adapter(model, "./adapter", "meta-llama/Llama-3-8B")
  print(get_vllm_serve_commands("meta-llama/Llama-3-8B", adapter_dir="./adapter"))
"""

from rocm_qlora.export.gguf_export import (
    merge_and_export_fp16,
    get_gguf_conversion_instructions,
    estimate_gguf_size,
)
from rocm_qlora.export.vllm_export import (
    export_for_vllm_merged,
    export_lora_adapter,
    get_vllm_serve_commands,
)

__all__ = [
    # GGUF export
    "merge_and_export_fp16",
    "get_gguf_conversion_instructions",
    "estimate_gguf_size",
    # vLLM export
    "export_for_vllm_merged",
    "export_lora_adapter",
    "get_vllm_serve_commands",
]