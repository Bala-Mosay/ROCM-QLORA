"""
HuggingFace-compatible quantization config for rocm-qlora.

Enables the from_pretrained() interface:

    from rocm_qlora.hf_integration import RocmQLoraConfig
    from transformers import AutoModelForCausalLM

    config = RocmQLoraConfig(bits=8, lora_r=16)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B",
        quantization_config=config,
    )
    # model is now quantized + LoRA-ready on ROCm
"""
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

try:
    from transformers.utils.quantization_config import QuantizationConfigMixin
except ImportError:
    class QuantizationConfigMixin:
        """Fallback for older transformers versions."""
        def to_dict(self):
            # Simple fallback for asdict-like behavior if not using dataclass
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

@dataclass
class RocmQLoraConfig(QuantizationConfigMixin):
    """
    Configuration for rocm-qlora quantization.
    """
    quant_method: str = "rocm_qlora" # Required for HF registry lookup
    
    def __init__(
        self,
        bits: int = 8,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        block_size: int = 64,
        use_triton_kernels: bool = True,
        use_paged_optimizer: bool = True,
        use_flash_attention: bool = True,
        use_packing: bool = True,
        **kwargs,
    ):
        self.bits = bits
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules if target_modules is not None else ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.block_size = block_size
        self.use_triton_kernels = use_triton_kernels
        self.use_paged_optimizer = use_paged_optimizer
        self.use_flash_attention = use_flash_attention
        self.use_packing = use_packing
        
        if self.bits not in [4, 8]:
            raise ValueError(f"rocm-qlora only supports 4 or 8 bits, got {self.bits}")

    def to_dict(self) -> Dict[str, Any]:
        """Returns all init params as a plain dict."""
        return {
            "quant_method": self.quant_method,
            "bits": self.bits,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "block_size": self.block_size,
            "use_triton_kernels": self.use_triton_kernels,
            "use_paged_optimizer": self.use_paged_optimizer,
            "use_flash_attention": self.use_flash_attention,
            "use_packing": self.use_packing
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RocmQLoraConfig":
        """Reconstruct from to_dict() output."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return (f"RocmQLoraConfig(bits={self.bits}, lora_r={self.lora_r}, "
                f"lora_alpha={self.lora_alpha}, target_modules={self.target_modules})")
