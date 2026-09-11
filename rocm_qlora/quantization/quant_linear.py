"""
Quantized Linear layer implementation for ROCm.
Handles storage of quantized weights and dynamic dequantization during forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from rocm_qlora.quantization.quant_ops import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4
)

class QuantLinear(nn.Module):
    """
    A Linear layer that stores weights in quantized format.
    Weights are dequantized on-the-fly during the forward pass.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bits: int = 8, 
        bias: bool = True, 
        block_size: int = 64
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.original_shape: Optional[Tuple[int, ...]] = (out_features, in_features)
        
        # NOTE: We use buffers instead of parameters so the optimizer ignores them.
        # They will still move with .to(device) and be saved in the state_dict.
        self.register_buffer("weight_quant", torch.empty(0, dtype=torch.int8 if bits == 8 else torch.uint8))
        self.register_buffer("weight_scales", torch.empty(0))
        self.register_buffer("weight_zeros", torch.zeros(1)) # Reserved for future asymmetric quantization
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None
            
        # V2: Kernel integration
        self.use_triton_kernel: bool = False  # off by default, enabled via enable_kernel()
        # V4: Double quantization (attributes set by patch_quant_linear_for_double_quant)
        self.use_double_quant: bool = False

    @classmethod
    def from_linear(
        cls, 
        linear: nn.Linear, 
        bits: int = 8, 
        block_size: int = 64
    ) -> "QuantLinear":
        """
        Converts a standard nn.Linear layer into a QuantLinear layer.
        Immediately reclaims VRAM by deleting the original weights.
        """
        instance = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            bias=linear.bias is not None,
            block_size=block_size
        )
        
        # Store original shape for correct dequantization later
        instance.original_shape = linear.weight.shape
        
        # NOTE: Flattening to 1D before quantization to ensure Phase 1 ops work correctly.
        flat_weight = linear.weight.data.flatten()
        
        if bits == 8:
            q_weight, scales = quantize_int8(flat_weight, block_size=block_size)
        elif bits == 4:
            q_weight, scales = quantize_int4(flat_weight, block_size=block_size)
        else:
            raise ValueError(f"Unsupported bit depth: {bits}")
            
        instance.weight_quant = q_weight
        instance.weight_scales = scales
        
        if linear.bias is not None:
            instance.bias = linear.bias.data.clone()
            
        # NOTE: Critical VRAM reclaim step.
        del linear.weight
        if linear.bias is not None:
            del linear.bias
            
        return instance

    def enable_kernel(self) -> bool:
        """Enable Triton kernel for forward pass. Returns True if successful."""
        from rocm_qlora.kernels import is_triton_available
        if is_triton_available():
            self.use_triton_kernel = True
            return True
        return False

    def enable_double_quant(self) -> bool:
        """Enable double quantization for this layer. Returns True if successful."""
        from rocm_qlora.quantization.double_quant import patch_quant_linear_for_double_quant
        # Wrap in a mini-model so the utility function can find it
        class Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
        
        wrapper = Wrapper(self)
        count = patch_quant_linear_for_double_quant(wrapper)
        return count > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic dequantization.
        """
        # V5 HIP kernel dispatch (MI300X)
        if getattr(self, "use_hip_kernel", False) and x.is_cuda:
            from rocm_qlora.hip_kernels import hip_dequant_int4_matmul, hip_dequant_int8_matmul
            if self.bits == 8:
                return hip_dequant_int8_matmul(
                    x, self.weight_quant, self.weight_scales,
                    self.bias, self.block_size
                )
            else:
                return hip_dequant_int4_matmul(
                    x, self.weight_quant, self.weight_scales,
                    self.bias, self.block_size
                )

        if self.use_triton_kernel and x.is_cuda:
            from rocm_qlora.kernels import fused_dequant_int8_matmul, fused_dequant_nf4_matmul
            if self.bits == 8:
                return fused_dequant_int8_matmul(
                    x, self.weight_quant, self.weight_scales,
                    self.bias, self.block_size
                )
            else:
                return fused_dequant_nf4_matmul(
                    x, self.weight_quant, self.weight_scales,
                    self.bias, self.block_size
                )

        # 1. Dequantize based on bit depth
        if self.use_double_quant and self.bits == 4:
            if not hasattr(self, 'c2') or not hasattr(self, 'c2_scales'):
                raise RuntimeError(
                    "Double quantization attributes not initialized. "
                    "Call patch_quant_linear_for_double_quant() first."
                )
            from rocm_qlora.quantization.double_quant import double_dequantize, DoubleQuantState
            state = DoubleQuantState(
                W_quant=self.weight_quant,
                c2=self.c2,
                c2_scales=self.c2_scales,
                blocksize_1=self.block_size,
                blocksize_2=self.blocksize_2,
                original_shape=self.original_shape,
                use_fp8_c2=self.use_fp8_c2,
            )
            weight_fp = double_dequantize(state)
        elif self.bits == 8:
            weight_fp = dequantize_int8(
                self.weight_quant, 
                self.weight_scales, 
                block_size=self.block_size
            )
        else:
            weight_fp = dequantize_int4(
                self.weight_quant, 
                self.weight_scales, 
                block_size=self.block_size
            )
            
        # 2. Restore original shape and cast to input dtype for mixed precision support
        # We truncate to match the expected number of elements in case of padding.
        num_elements = self.out_features * self.in_features
        weight_fp = weight_fp[:num_elements].reshape(self.original_shape)
        weight_fp = weight_fp.to(x.dtype)
        
        # 3. Perform standard linear operation
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight_fp, bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, block_size={self.block_size}"
