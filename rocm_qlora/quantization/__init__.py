"""
Quantization module for rocm-qlora.
"""

from rocm_qlora.quantization.quant_ops import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4
)
from rocm_qlora.quantization.quant_linear import QuantLinear
from rocm_qlora.quantization.double_quant import (
    double_quantize, double_dequantize,
    DoubleQuantState, estimate_double_quant_savings,
    patch_quant_linear_for_double_quant,
)

__all__ = [
    "quantize_int8", "dequantize_int8",
    "quantize_int4", "dequantize_int4",
    "QuantLinear",
    "double_quantize", "double_dequantize",
    "DoubleQuantState", "estimate_double_quant_savings",
    "patch_quant_linear_for_double_quant",
]
