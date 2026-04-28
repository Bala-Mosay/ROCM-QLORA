"""
rocm-qlora: Pure PyTorch QLoRA fine-tuning for AMD GPUs.
No bitsandbytes. No CUDA dependencies.

Quick start:
    from rocm_qlora import quantize_model, check_rocm
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("...", torch_dtype=torch.float16)
    model = quantize_model(model, bits=8, lora_r=8)
    model = model.to("cuda")
    # Ready for QLoRA fine-tuning on AMD GPU

    # Or with HF plugin (requires transformers >= 4.40):
    from rocm_qlora.hf_integration import RocmQLoraConfig
    model = AutoModelForCausalLM.from_pretrained(
        "...", quantization_config=RocmQLoraConfig(bits=8)
    )
"""
import os

# V1 core — immutable API
from rocm_qlora.model.quantize_model import quantize_model
from rocm_qlora.quantization.quant_linear import QuantLinear
from rocm_qlora.lora.lora_layer import LoRALinear
from rocm_qlora.utils.rocm_utils import check_rocm

# V2 additions
from rocm_qlora.kernels import (
    fused_dequant_int8_matmul, fused_dequant_nf4_matmul,
    get_device_info, is_triton_available,
)
from rocm_qlora.model.quantize_model import enable_all_kernels
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.data import build_packed_dataset, PackedSequenceCollator
from rocm_qlora.attention import (
    detect_flash_attention, patch_model_attention, enable_sdpa_optimization,
)
from rocm_qlora.utils.compile_utils import (
    compile_model, compile_lora_only, is_compile_safe,
)

# V3 additions
from rocm_qlora.trainers import (
    ROCmSFTTrainer, SFTConfig,
    ROCmDPOTrainer, DPOConfig,
    ROCmGRPOTrainer, GRPOConfig,
)
from rocm_qlora.hf_integration import RocmQLoraConfig, RocmQLoraQuantizer

# V4 additions
from rocm_qlora.export import (
    merge_and_export_fp16,
    get_gguf_conversion_instructions,
    estimate_gguf_size,
    export_for_vllm_merged,
    export_lora_adapter,
    get_vllm_serve_commands,
)

__version__ = "4.0.0"

__all__ = [
    # V1
    "quantize_model", "QuantLinear", "LoRALinear", "check_rocm",
    # V2
    "fused_dequant_int8_matmul", "fused_dequant_nf4_matmul",
    "get_device_info", "is_triton_available", "enable_all_kernels",
    "PagedAdamW",
    "build_packed_dataset", "PackedSequenceCollator",
    "detect_flash_attention", "patch_model_attention", "enable_sdpa_optimization",
    "compile_model", "compile_lora_only", "is_compile_safe",
    # V3
    "ROCmSFTTrainer", "SFTConfig",
    "ROCmDPOTrainer", "DPOConfig",
    "ROCmGRPOTrainer", "GRPOConfig",
    "RocmQLoraConfig", "RocmQLoraQuantizer",
    # V4
    "merge_and_export_fp16",
    "get_gguf_conversion_instructions",
    "estimate_gguf_size",
    "export_for_vllm_merged",
    "export_lora_adapter",
    "get_vllm_serve_commands",
    "__version__",
]

if not os.environ.get("ROCM_QLORA_QUIET"):
    _info = check_rocm()
    _device_info = get_device_info()
    print(
        f"[rocm-qlora {__version__}] "
        f"GPU: {_info['device_name']} | "
        f"ROCm: {_info['version']} | "
        f"Triton: {'V' if is_triton_available() else 'X'} | "
        f"Arch: {_device_info.get('arch', 'unknown')}"
    )
