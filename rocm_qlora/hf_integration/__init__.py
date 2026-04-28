"""
HuggingFace integration for rocm-qlora.

Enables the from_pretrained() interface with automatic quantization:

    from rocm_qlora.hf_integration import RocmQLoraConfig
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=RocmQLoraConfig(bits=8, lora_r=8),
    )
"""
from rocm_qlora.hf_integration.config import RocmQLoraConfig
from rocm_qlora.hf_integration.quantizer import RocmQLoraQuantizer

__all__ = ["RocmQLoraConfig", "RocmQLoraQuantizer"]
