"""
FP8 linear compute layer for rocm-qlora.

Storage:  QuantLinear (INT8/NF4)
Compute:  FP8 matmul (TransformerEngine or Native)
Adapters: BF16
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import contextlib
from typing import Dict, Any, Tuple, Optional

from rocm_qlora.fp8.fp8_config import detect_fp8_support, FP8Config, get_fp8_dtype, get_fp8_recipe
from rocm_qlora.lora.lora_layer import LoRALinear

class FP8LinearWrapper(nn.Module):
    """
    Wraps LoRALinear to use FP8 compute while keeping quantized storage.
    """
    def __init__(self, base_lora_linear: nn.Module, config: Optional[FP8Config] = None):
        super().__init__()
        self.config = config if config is not None else FP8Config()
        self.base = base_lora_linear
        self.fp8_info = detect_fp8_support()
        
        # Determine Backend
        if not self.config.enabled or not self.fp8_info['fp8_supported']:
            self.backend = "bf16"
        elif self.fp8_info['transformer_engine_available'] and self.config.use_transformer_engine:
            self.backend = "transformer_engine"
            self.te_recipe = get_fp8_recipe(self.config)
        elif self.fp8_info['fp8_dtypes_available']:
            self.backend = "native"
        else:
            self.backend = "bf16"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "transformer_engine":
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=True, fp8_recipe=self.te_recipe):
                return self.base(x)
        
        elif self.backend == "native":
            # NOTE: native torch._scaled_mm is highly experimental on ROCm.
            # We dequantize the base weight to FP16 first.
            # Then cast x and weight to FP8 for matmul.
            
            # 1. Base Weight (QuantLinear) -> FP16
            # We don't want to call self.base(x) yet because that adds LoRA too.
            # We need the quantized base weight dequantized.
            from rocm_qlora.quantization.quant_ops import dequantize_int8, dequantize_int4
            ql = self.base.base_layer
            if ql.use_double_quant:
                from rocm_qlora.quantization.double_quant import double_dequantize, DoubleQuantState
                state = DoubleQuantState(ql.weight_quant, ql.c2, ql.c2_scales, ql.block_size, ql.blocksize_2, ql.original_shape, ql.use_fp8_c2)
                w_fp16 = double_dequantize(state).to(x.dtype)
            elif ql.bits == 8:
                w_fp16 = dequantize_int8(ql.weight_quant, ql.weight_scales, ql.block_size).view(ql.original_shape).to(x.dtype)
            else:
                w_fp16 = dequantize_int4(ql.weight_quant, ql.weight_scales, ql.block_size).view(ql.original_shape).to(x.dtype)
            
            # 2. Cast to FP8
            f8_dtype = get_fp8_dtype(self.config.forward_dtype)
            x_f8 = x.to(f8_dtype)
            w_f8 = w_fp16.to(f8_dtype)
            
            # 3. Scaled Matmul (Conceptual fallback if scaled_mm unavailable)
            try:
                if hasattr(torch, '_scaled_mm'):
                    # ROCm 6.2+ supports this
                    # torch._scaled_mm requires 2D inputs — reshape if needed
                    original_shape = x_f8.shape
                    if x_f8.dim() > 2:
                        x_f8_2d = x_f8.reshape(-1, x_f8.shape[-1])
                    else:
                        x_f8_2d = x_f8
                    scale_x = torch.tensor([1.0], device=x.device)
                    scale_w = torch.tensor([1.0], device=x.device)
                    base_out = torch._scaled_mm(x_f8_2d, w_f8.t(), scale_x, scale_w, out_dtype=x.dtype)
                    base_out = base_out.reshape(*original_shape[:-1], -1)
                else:
                    # Fallback to standard matmul if scaled_mm missing in this build
                    base_out = F.linear(x_f8.to(x.dtype), w_f8.to(x.dtype))
            except RuntimeError:
                # FP8 GEMM not supported by HIPBLAS on this hardware — fall back to BF16
                base_out = F.linear(x.to(w_fp16.dtype), w_fp16)
            
            # 4. Add Bias and LoRA
            if ql.bias is not None:
                base_out = base_out + ql.bias.to(x.dtype)
            
            # LoRA contribution (always in BF16/FP16)
            # result = base_out + (x @ A.T @ B.T * scaling)
            lora_out = self.base.lora_dropout(x) @ self.base.lora_A.t().to(x.dtype) @ self.base.lora_B.t().to(x.dtype) * self.base.scaling
            return base_out + lora_out.to(base_out.dtype)

        else:
            # BF16 Fallback (V3 behavior)
            return self.base(x)

    def extra_repr(self) -> str:
        return f"backend={self.backend}, fp8_supported={self.fp8_info['fp8_supported']}"

def wrap_model_for_fp8(model: nn.Module, config: Optional[FP8Config] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Wraps all LoRA layers in the model with FP8 compute wrappers.
    """
    config = config if config is not None else FP8Config()
    fp8_info = detect_fp8_support()
    
    status = {
        "enabled": False,
        "backend": "bf16",
        "layers_wrapped": 0,
        "reason": fp8_info["reason"]
    }
    
    if not fp8_info["fp8_supported"] and config.enabled:
        return model, status

    # Two-pass replacement
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            to_replace.append((name, module))
            
    for name, module in to_replace:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        
        wrapped = FP8LinearWrapper(module, config)
        setattr(parent, child_name, wrapped)
        status["layers_wrapped"] += 1
        status["backend"] = wrapped.backend
        status["enabled"] = wrapped.backend != "bf16"

    # Set ROCm environment variables for FP8 + FSDP stability
    if status["enabled"]:
        os.environ.setdefault("TORCH_NCCL_HIGH_PRIORITY", "1")
        os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
        
    return model, status

def enable_fp8_autocast(model: nn.Module, config: Optional[FP8Config] = None):
    """
    Context manager for FP8 autocast.
    """
    config = config if config is not None else FP8Config()
    try:
        import transformer_engine.pytorch as te
        if config.use_transformer_engine:
            recipe = get_fp8_recipe(config)
            return te.fp8_autocast(enabled=True, fp8_recipe=recipe)
    except ImportError:
        pass
    return contextlib.nullcontext()
