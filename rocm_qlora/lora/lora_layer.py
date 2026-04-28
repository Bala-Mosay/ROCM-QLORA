"""
LoRA (Low-Rank Adaptation) layer implementation for ROCm.
Wraps QuantLinear layers with trainable low-rank adapters.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from rocm_qlora.quantization.quant_linear import QuantLinear
from rocm_qlora.quantization.quant_ops import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4
)

class LoRALinear(nn.Module):
    """
    Wraps a QuantLinear layer with LoRA adapters (A and B matrices).
    Supports merging the adapters back into the quantized base weight.
    """
    def __init__(
        self, 
        quant_linear: QuantLinear, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.base = quant_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        
        # NOTE: Freeze base model weights immediately.
        # QuantLinear uses buffers, but we ensure no parameters accidentally remain trainable.
        for p in self.base.parameters():
            p.requires_grad_(False)
            
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(r, quant_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(quant_linear.out_features, r))
        
        # NOTE: lora_B initialized to zero ensures lora_out is zero at step 0.
        # lora_A kaiming initialization follows PEFT library conventions.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base_output + (x @ A.T @ B.T * scaling)
        """
        if self.merged:
            return self.base(x)
            
        # Standard LoRA path
        base_out = self.base(x)
        
        # Adapter path: [Batch, In] @ [In, R] @ [R, Out] -> [Batch, Out]
        # We use transpose to match linear layer weight conventions
        lora_out = self.lora_dropout(x) @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T * self.scaling
        
        # NOTE: Casting to base_out.dtype (e.g., BF16/FP16) to prevent promotion to FP32.
        return base_out + lora_out.to(base_out.dtype)

    def merge_lora(self) -> None:
        """
        Merges LoRA weights into the base QuantLinear layer.
        This involves dequantizing, adding the delta, and re-quantizing.
        """
        if self.merged:
            raise RuntimeError("LoRA already merged")
            
        # 1. Compute delta_W = (B @ A) * scaling
        # Result shape: [out_features, in_features]
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        
        # 2. Dequantize base weight directly from buffers (bypassing forward logic)
        if self.base.bits == 8:
            base_weight_fp = dequantize_int8(
                self.base.weight_quant,
                self.base.weight_scales,
                block_size=self.base.block_size
            )
        else:
            base_weight_fp = dequantize_int4(
                self.base.weight_quant,
                self.base.weight_scales,
                block_size=self.base.block_size
            )
            
        # Reshape to original layout [Out, In]
        num_elements = self.base.out_features * self.base.in_features
        base_weight_fp = base_weight_fp[:num_elements].reshape(self.base.original_shape)
        
        # 3. Add delta and re-quantize
        # NOTE: Re-quantization introduces small additional error but is necessary for merged inference.
        merged_weight = base_weight_fp + delta_W.to(base_weight_fp.dtype)
        
        # Flatten for re-quantization ops
        flat_merged = merged_weight.flatten()
        
        if self.base.bits == 8:
            q_weight, scales = quantize_int8(flat_merged, block_size=self.base.block_size)
        else:
            q_weight, scales = quantize_int4(flat_merged, block_size=self.base.block_size)
            
        # Store back into base buffers
        self.base.weight_quant = q_weight
        self.base.weight_scales = scales
        
        # 4. Cleanup and state update
        self.merged = True
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def get_trainable_params(self) -> int:
        """Returns total number of trainable parameters in the LoRA layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.lora_alpha}, scaling={self.scaling:.4f}, merged={self.merged}"
