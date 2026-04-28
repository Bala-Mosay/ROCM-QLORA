import pytest
import torch
import torch.nn as nn
from rocm_qlora.kernels import (
    get_kernel_config, 
    get_device_info, 
    is_triton_available,
    fused_dequant_int8_matmul,
    fused_dequant_nf4_matmul,
    dequantize_nf4_triton
)
from rocm_qlora.kernels.kernel_config import TritonKernelConfig
from rocm_qlora.quantization.quant_ops import (
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4
)
from rocm_qlora.quantization.quant_linear import QuantLinear

def test_kernel_config_returns_dataclass():
    config = get_kernel_config()
    assert isinstance(config, TritonKernelConfig)
    assert hasattr(config, "BLOCK_M")
    assert hasattr(config, "wavefront_size")

def test_device_info_keys():
    info = get_device_info()
    assert isinstance(info, dict)
    expected_keys = {"arch", "compute_units", "vram_gb", "wavefront_size", "triton_available"}
    assert expected_keys.issubset(set(info.keys()))

def test_int8_fallback_matches_v1():
    # Setup test data
    M, N, K = 4, 128, 64
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    block_size = 32
    
    # Quantize using v1
    w_quant, scales = quantize_int8(w.flatten(), block_size=block_size)
    w_quant = w_quant.reshape(N, K)
    
    # V1 Manual: dequant + F.linear
    w_fp16 = dequantize_int8(w_quant, scales, block_size=block_size).reshape(N, K)
    expected = torch.nn.functional.linear(x, w_fp16)
    
    # V2 Fallback (fused_dequant_int8_matmul on CPU)
    actual = fused_dequant_int8_matmul(x, w_quant, scales, block_size=block_size)
    
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

def test_nf4_fallback_matches_v1():
    M, N, K = 4, 128, 64
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    block_size = 32
    
    # Quantize using v1
    w_packed, scales = quantize_int4(w.flatten(), block_size=block_size)
    
    # V1 Manual
    w_fp16 = dequantize_int4(w_packed, scales, block_size=block_size).reshape(N, K)
    expected = torch.nn.functional.linear(x, w_fp16)
    
    # V2 Fallback (fused_dequant_nf4_matmul on CPU)
    w_packed_2d = w_packed.reshape(N, K // 2)
    actual = fused_dequant_nf4_matmul(x, w_packed_2d, scales, block_size=block_size)
    
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

def test_output_shape_2d():
    M, N, K = 4, 128, 64
    x = torch.randn(M, K)
    w_quant = torch.zeros(N, K, dtype=torch.int8)
    scales = torch.ones( (N*K) // 64 )
    
    output = fused_dequant_int8_matmul(x, w_quant, scales, block_size=64)
    assert output.shape == (M, N)

def test_output_shape_3d():
    B, S, N, K = 2, 8, 128, 64
    x = torch.randn(B, S, K)
    w_quant = torch.zeros(N, K, dtype=torch.int8)
    scales = torch.ones( (N*K) // 64 )
    
    output = fused_dequant_int8_matmul(x, w_quant, scales, block_size=64)
    assert output.shape == (B, S, N)

def test_bias_applied():
    M, N, K = 4, 64, 32
    x = torch.randn(M, K)
    w_quant = torch.zeros(N, K, dtype=torch.int8)
    scales = torch.ones( (N*K) // 32 )
    bias = torch.ones(N) * 5.0
    
    out_no_bias = fused_dequant_int8_matmul(x, w_quant, scales, bias=None, block_size=32)
    out_with_bias = fused_dequant_int8_matmul(x, w_quant, scales, bias=bias, block_size=32)
    
    # Difference should be exactly the bias
    diff = out_with_bias - out_no_bias
    for i in range(M):
        torch.testing.assert_close(diff[i], bias)

def test_enable_kernel_returns_bool():
    layer = QuantLinear(64, 64)
    res = layer.enable_kernel()
    assert isinstance(res, bool)
    if res:
        assert layer.use_triton_kernel is True

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_kernel_output_matches_fallback():
    if not is_triton_available():
        pytest.skip("Triton not available")
        
    device = "cuda"
    M, N, K = 16, 128, 128
    block_size = 64
    
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    
    # Setup quantized weights
    w_quant, scales = quantize_int8(w.flatten(), block_size=block_size)
    w_quant = w_quant.reshape(N, K).to(device)
    scales = scales.to(device)
    
    # Force Triton
    cfg = get_kernel_config()
    from rocm_qlora.kernels.dequant_matmul import _triton_dequant_int8_matmul, _fallback_torch_dequant_int8_matmul
    
    actual_triton = _triton_dequant_int8_matmul(x, w_quant, scales, None, block_size, cfg)
    expected_fallback = _fallback_torch_dequant_int8_matmul(x, w_quant, scales, None, block_size)
    
    torch.testing.assert_close(actual_triton, expected_fallback, atol=0.05, rtol=0.05)
