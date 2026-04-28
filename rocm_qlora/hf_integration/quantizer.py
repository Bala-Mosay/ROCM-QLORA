"""
HuggingFace HfQuantizer implementation for rocm-qlora.

Plugs into transformers' quantizer registry so that:
    AutoModelForCausalLM.from_pretrained(..., quantization_config=RocmQLoraConfig())
automatically invokes rocm-qlora's quantization pipeline.
"""
import torch
import torch.nn as nn
import transformers
from typing import Any, Dict, List, Optional

from rocm_qlora.model.quantize_model import quantize_model, enable_all_kernels
from rocm_qlora.attention import patch_model_attention
from rocm_qlora.kernels import is_triton_available
from rocm_qlora.utils.rocm_utils import check_rocm

try:
    from transformers.quantizers.base import HfQuantizer
except ImportError:
    class HfQuantizer:
        def __init__(self, quantization_config, **kwargs):
            raise ImportError(
                "transformers >= 4.40 required for HF plugin. "
                "Use quantize_model() directly instead."
            )

class RocmQLoraQuantizer(HfQuantizer):
    """
    Quantizer class for rocm-qlora.
    """
    requires_calibration: bool = False
    required_packages: List[str] = ["rocm_qlora"]

    def validate_environment(self, *args, **kwargs):
        """Check hardware and software requirements."""
        if not torch.cuda.is_available():
            print("[rocm-qlora] WARNING: No GPU detected. Training will be extremely slow.")
        
        if torch.version.hip is None:
            print("[rocm-qlora] WARNING: ROCm (HIP) not detected. Performance may be sub-optimal.")
            
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.40.0"):
            raise ImportError(
                f"transformers >= 4.40.0 required for rocm-qlora plugin. "
                f"Found {transformers.__version__}. Please upgrade."
            )
            
        # Log ROCm info
        if not torch.cuda.is_available():
             print("[rocm-qlora] Running in CPU/Mock mode.")
        else:
             check_rocm()

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs) -> nn.Module:
        """Modify model structure BEFORE weights are loaded."""
        config = self.quantization_config
        
        # This replaces Linear layers with QuantLinear + LoRALinear
        return quantize_model(
            model,
            bits=config.bits,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            block_size=config.block_size
        )

    def _process_model_after_weight_loading(self, model: nn.Module, **kwargs) -> nn.Module:
        """Post-load optimizations once weights are in memory."""
        config = self.quantization_config
        
        # 1. Enable Triton kernels
        if config.use_triton_kernels and is_triton_available():
            enable_all_kernels(model)
            
        # 2. Patch Flash Attention
        if config.use_flash_attention:
            patch_model_attention(model, verbose=False)
            
        # 3. Print final report
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[rocm-qlora] Model ready. Trainable: {trainable:,} / Total: {total:,}")
        
        return model

    def is_serializable(self) -> bool:
        return True

    def is_trainable(self, model: nn.Module = None) -> bool:
        return True
