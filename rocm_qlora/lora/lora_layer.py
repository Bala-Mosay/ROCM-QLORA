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
        self.base_layer = quant_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        
        # NOTE: Freeze base model weights immediately.
        # QuantLinear uses buffers, but we ensure no parameters accidentally remain trainable.
        for p in self.base_layer.parameters():
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
            return self.base_layer(x)
            
        # Standard LoRA path
        base_out = self.base_layer(x)
        
        # Adapter path: [Batch, In] @ [In, R] @ [R, Out] -> [Batch, Out]
        # We use transpose to match linear layer weight conventions
        lora_out = self.lora_dropout(x) @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T * self.scaling
        
        # NOTE: Casting to base_out.dtype (e.g., BF16/FP16) to prevent promotion to FP32.
        return base_out + lora_out.to(base_out.dtype)

    def merge_lora(self) -> None:
        """
        Merges LoRA weights into the base QuantLinear layer in FP16.
        No re-quantization — stores merged FP16 weight for zero-error inference.
        Use unmerge_lora() to revert and continue training.
        """
        if self.merged:
            raise RuntimeError("LoRA already merged")
            
        # 1. Compute delta_W = (B @ A) * scaling
        # Result shape: [out_features, in_features]
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        
        # 2. Dequantize base weight directly from buffers (bypassing forward logic)
        if self.base_layer.bits == 8:
            base_weight_fp = dequantize_int8(
                self.base_layer.weight_quant,
                self.base_layer.weight_scales,
                block_size=self.base_layer.block_size
            )
        else:
            base_weight_fp = dequantize_int4(
                self.base_layer.weight_quant,
                self.base_layer.weight_scales,
                block_size=self.base_layer.block_size
            )
            
        # Reshape to original layout [Out, In]
        num_elements = self.base_layer.out_features * self.base_layer.in_features
        base_weight_fp = base_weight_fp[:num_elements].reshape(self.base_layer.original_shape)
        
        # 3. Add delta in FP16 — NO re-quantization (avoids atol~2.0 error)
        merged_weight = (base_weight_fp + delta_W.to(base_weight_fp.dtype)).to(torch.float16)
        
        # Store in FP16 merged mode
        self.base_layer.merged_fp16_weight = merged_weight
        self.base_layer.merged_mode = True
        
        # 4. Cleanup and state update
        self.merged = True
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def unmerge_lora(self) -> None:
        """
        Reverts merge_lora() — restores original quantized weights and clears FP16 buffer.
        Allows continuing training after merged inference.
        """
        if not self.merged:
            raise RuntimeError("LoRA is not merged — nothing to unmerge")
            
        # Clear FP16 merged mode
        self.base_layer.merged_fp16_weight = None
        self.base_layer.merged_mode = False
        
        # Restore state
        self.merged = False

    def get_trainable_params(self) -> int:
        """Returns total number of trainable parameters in the LoRA layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.lora_alpha}, scaling={self.scaling:.4f}, merged={self.merged}"
