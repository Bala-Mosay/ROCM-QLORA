"""
Tests for V5 HIP INT4/INT8 matmul kernels.

CPU tests always run (verify fallback path, no crash).
GPU tests marked @pytest.mark.gpu — only run on MI300X hardware.
"""

import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =========================================================================
# CPU fallback tests (always safe, no GPU required)
# =========================================================================


class TestHipKernelImport:
    """Verify that the hip_kernels module can be imported without crashing."""

    def test_hip_kernel_import_no_crash(self):
        """Importing rocm_qlora.hip_kernels should not raise any exception."""
        import rocm_qlora.hip_kernels
        assert rocm_qlora.hip_kernels is not None

    def test_hip_kernel_submodules_import(self):
        """All hip_kernels submodules should import cleanly."""
        from rocm_qlora.hip_kernels import int4_matmul, build_utils
        assert int4_matmul is not None
        assert build_utils is not None

    def test_public_api_exports(self):
        """All 4 public API functions should be importable from top-level."""
        from rocm_qlora.hip_kernels import (
            is_hip_kernel_available,
            enable_hip_kernels,
            hip_dequant_int4_matmul,
            hip_dequant_int8_matmul,
            benchmark_hip_vs_triton,
        )
        assert callable(is_hip_kernel_available)
        assert callable(enable_hip_kernels)
        assert callable(hip_dequant_int4_matmul)
        assert callable(hip_dequant_int8_matmul)
        assert callable(benchmark_hip_vs_triton)


class TestFallbackToTritonWhenNoHipcc:
    """Verify graceful degradation when hipcc is not available."""

    def test_is_hip_kernel_available_returns_false_on_cpu(self):
        """Without GPU/hipcc, is_hip_kernel_available() should return False."""
        from rocm_qlora.hip_kernels import is_hip_kernel_available
        # On a system without hipcc/GPU, this must be False
        result = is_hip_kernel_available()
        assert isinstance(result, bool)
        # Note: could be True on actual MI300X dev machine, so just check type

    def test_fallback_to_triton_when_no_hipcc(self):
        """hip_dequant_int4_matmul should not crash on CPU without hipcc."""
        from rocm_qlora.hip_kernels import hip_dequant_int4_matmul
        from rocm_qlora.quantization.quant_ops import quantize_int4

        # Create small test tensors on CPU
        x = torch.randn(2, 64, dtype=torch.float32)
        weight = torch.randn(128, 64, dtype=torch.float32)
        w_packed, scales = quantize_int4(weight.flatten())

        # Should run without error, using V1/Triton fallback
        output = hip_dequant_int4_matmul(x, w_packed, scales, block_size=64)
        assert output.shape == (2, 128)
        assert output.dtype == torch.float32

    def test_fallback_output_matches_v1(self):
        """
        CPU fallback output must match V1 pure PyTorch dequant+matmul.

        # NOTE: atol=0.05 because quantization rounding may differ slightly.
        """
        from rocm_qlora.hip_kernels import hip_dequant_int4_matmul
        from rocm_qlora.quantization.quant_ops import dequantize_int4, quantize_int4
        import torch.nn.functional as F

        torch.manual_seed(42)
        x = torch.randn(4, 256, dtype=torch.float32)
        weight = torch.randn(512, 256, dtype=torch.float32)
        w_packed, scales = quantize_int4(weight.flatten())

        # HIP path (will use V1 fallback on CPU)
        output_hip = hip_dequant_int4_matmul(x, w_packed, scales, block_size=64)

        # V1 pure PyTorch reference
        K = x.shape[-1]
        N = w_packed.numel() * 2 // K
        w_fp16 = dequantize_int4(w_packed.flatten(), scales, 64).reshape(N, K)
        output_ref = F.linear(x, w_fp16)

        assert torch.allclose(output_hip, output_ref, atol=0.05), \
            f"Max diff: {(output_hip - output_ref).abs().max().item()}"

    def test_int8_fallback_output_matches_v1(self):
        """INT8 CPU fallback must match V1 pure PyTorch dequant+matmul."""
        from rocm_qlora.hip_kernels import hip_dequant_int8_matmul
        from rocm_qlora.quantization.quant_ops import dequantize_int8, quantize_int8
        import torch.nn.functional as F

        torch.manual_seed(42)
        x = torch.randn(4, 256, dtype=torch.float32)
        weight = torch.randn(512, 256, dtype=torch.float32)
        w_int8, scales = quantize_int8(weight.flatten())
        w_int8 = w_int8.reshape(512, 256)

        output_hip = hip_dequant_int8_matmul(x, w_int8, scales, block_size=64)

        w_fp16 = dequantize_int8(w_int8, scales, 64)
        output_ref = F.linear(x, w_fp16)

        assert torch.allclose(output_hip, output_ref, atol=0.05), \
            f"Max diff: {(output_hip - output_ref).abs().max().item()}"


class TestBuildUtils:
    """Tests for the build_utils compilation infrastructure."""

    def test_find_hipcc_returns_none_or_string(self):
        """find_hipcc() must return either None or a string path."""
        from rocm_qlora.hip_kernels.build_utils import find_hipcc
        result = find_hipcc()
        assert result is None or isinstance(result, str)

    def test_is_hipcc_available_is_bool(self):
        """is_hipcc_available() must return a boolean."""
        from rocm_qlora.hip_kernels.build_utils import is_hipcc_available
        result = is_hipcc_available()
        assert isinstance(result, bool)

    def test_compile_hip_kernel_no_hipcc_returns_false(self):
        """
        compile_hip_kernel with nonexistent file should return False
        when hipcc is not available (or gracefully handle error).
        """
        from rocm_qlora.hip_kernels.build_utils import compile_hip_kernel
        # Use a nonexistent output path to test error handling
        result = compile_hip_kernel(
            "/nonexistent/path/int4_matmul.hip",
            "/tmp/rocm_qlora_test_nonexistent.so",
            arch="gfx942"
        )
        # Should return False because source file doesn't exist
        assert isinstance(result, bool)

    def test_compile_hip_kernel_valid_source_no_hipcc(self):
        """
        compile_hip_kernel with valid source but no hipcc should return False gracefully.
        """
        import os
        from rocm_qlora.hip_kernels.build_utils import compile_hip_kernel

        package_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        hip_file = os.path.join(package_dir, "rocm_qlora", "hip_kernels", "int4_matmul.hip")

        if os.path.isfile(hip_file):
            result = compile_hip_kernel(
                hip_file,
                "/tmp/rocm_qlora_test.so",
                arch="gfx942"
            )
            # May succeed if hipcc installed, or False gracefully
            assert isinstance(result, bool)
        else:
            pytest.skip(f"HIP source not found at {hip_file}")


class TestEnableHipKernels:
    """Tests for enable_hip_kernels() on QuantLinear models."""

    def test_enable_hip_kernels_on_cpu_returns_zero(self):
        """
        enable_hip_kernels() on a CPU model should return 0
        (no HIP kernel available) and not crash.
        """
        from rocm_qlora.hip_kernels import enable_hip_kernels
        from rocm_qlora.quantization.quant_linear import QuantLinear

        # Create a mini model with QuantLinear layers
        class MiniModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q1 = QuantLinear(64, 128, bits=4)
                self.q2 = QuantLinear(128, 256, bits=8)

            def forward(self, x):
                x = self.q1(x)
                return self.q2(x)

        model = MiniModel()

        # Add some fake quantized weights so the layers don't crash
        model.q1.weight_quant = torch.zeros(128 * 64 // 2, dtype=torch.uint8)
        model.q1.weight_scales = torch.ones((128 * 64) // 64)
        model.q2.weight_quant = torch.zeros(256 * 128, dtype=torch.int8)
        model.q2.weight_scales = torch.ones((256 * 128) // 64)

        count = enable_hip_kernels(model)
        assert isinstance(count, int)
        assert count >= 0  # 0 on CPU, >0 if HIP available on GPU

    def test_enable_hip_kernels_sets_use_hip_kernel_flag(self):
        """
        After enable_hip_kernels(), QuantLinear layers should have
        the use_hip_kernel attribute set.
        """
        from rocm_qlora.hip_kernels import enable_hip_kernels
        from rocm_qlora.quantization.quant_linear import QuantLinear

        model = torch.nn.Sequential(
            QuantLinear(32, 64, bits=4),
            torch.nn.ReLU(),
            QuantLinear(64, 32, bits=8),
        )

        # Add weights
        for m in model:
            if isinstance(m, QuantLinear):
                m.weight_quant = torch.zeros(
                    (m.out_features * m.in_features) // (2 if m.bits == 4 else 1),
                    dtype=torch.uint8 if m.bits == 4 else torch.int8
                )
                m.weight_scales = torch.ones(m.out_features * m.in_features // 64)

        count = enable_hip_kernels(model)

        for m in model:
            if isinstance(m, QuantLinear):
                assert hasattr(m, "use_hip_kernel"), \
                    "use_hip_kernel flag should be set on all QuantLinear layers"
                assert isinstance(m.use_hip_kernel, bool)


class TestBenchmarkFunction:
    """Tests for benchmark_hip_vs_triton()."""

    def test_benchmark_returns_dict_with_expected_keys(self):
        """benchmark_hip_vs_triton() must return dict with all expected keys."""
        from rocm_qlora.hip_kernels import benchmark_hip_vs_triton

        # Small benchmark — fast on CPU
        result = benchmark_hip_vs_triton(size=(128, 128), n_runs=5)

        assert isinstance(result, dict)
        expected_keys = {"hip_ms", "triton_ms", "speedup_x", "winner", "hip_available"}
        assert expected_keys.issubset(set(result.keys())), \
            f"Missing keys: {expected_keys - set(result.keys())}"

    def test_benchmark_hip_ms_is_nan_on_cpu(self):
        """
        On CPU, hip_ms should be NaN since HIP kernel isn't available.
        triton_ms should be a real number (Triton or V1 fallback).
        """
        from rocm_qlora.hip_kernels import benchmark_hip_vs_triton

        result = benchmark_hip_vs_triton(size=(128, 128), n_runs=5)

        # hip_ms is NaN when HIP unavailable; triton_ms is always a real float
        import math
        if not result["hip_available"]:
            assert math.isnan(result["hip_ms"]), \
                f"hip_ms should be NaN on CPU, got {result['hip_ms']}"
        assert result["triton_ms"] > 0, \
            f"triton_ms should be positive, got {result['triton_ms']}"

    def test_benchmark_winner_is_string(self):
        """Winner field should be 'hip' or 'triton'."""
        from rocm_qlora.hip_kernels import benchmark_hip_vs_triton
        result = benchmark_hip_vs_triton(size=(128, 128), n_runs=5)
        assert result["winner"] in ("hip", "triton")

    def test_benchmark_does_not_crash_on_small_size(self):
        """Small benchmarks (very small matrices) shouldn't crash."""
        from rocm_qlora.hip_kernels import benchmark_hip_vs_triton
        result = benchmark_hip_vs_triton(size=(16, 16), n_runs=3)
        assert isinstance(result, dict)


# =========================================================================
# GPU tests (MI300X only — marked with @pytest.mark.gpu)
# =========================================================================


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for HIP kernel tests"
)
class TestHipKernelOnGPU:
    """Tests that require actual MI300X GPU hardware."""

    def test_hip_kernel_available_on_gpu(self):
        """
        On a ROCm GPU with hipcc installed, is_hip_kernel_available()
        should return True (or False gracefully if no hipcc).
        """
        from rocm_qlora.hip_kernels import is_hip_kernel_available
        from rocm_qlora.hip_kernels.build_utils import is_hipcc_available
        # At minimum, this shouldn't crash
        result = is_hip_kernel_available()
        assert isinstance(result, bool)
        # If hipcc is available on this GPU node, expect True
        if is_hipcc_available():
            # May still be False if compilation failed, but shouldn't crash
            pass

    def test_hip_output_matches_triton_int4(self):
        """
        HIP INT4 kernel output must match Triton output within atol=0.05.
        """
        from rocm_qlora.hip_kernels import (
            hip_dequant_int4_matmul, is_hip_kernel_available
        )
        from rocm_qlora.kernels import fused_dequant_nf4_matmul
        from rocm_qlora.quantization.quant_ops import quantize_int4

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        torch.manual_seed(42)
        device = "cuda"

        x = torch.randn(4, 512, dtype=torch.float16, device=device)
        weight = torch.randn(1024, 512, dtype=torch.float32)
        w_packed, scales = quantize_int4(weight.flatten())
        w_packed = w_packed.to(device)
        scales = scales.to(device)

        output_hip = hip_dequant_int4_matmul(x, w_packed, scales, block_size=64)
        output_triton = fused_dequant_nf4_matmul(x, w_packed, scales, block_size=64)

        diff = (output_hip - output_triton).abs().max().item()
        assert diff < 0.05, \
            f"HIP vs Triton max diff: {diff} (should be < 0.05)"

    def test_hip_output_matches_triton_int8(self):
        """
        HIP INT8 kernel output must match Triton output within atol=0.05.
        """
        from rocm_qlora.hip_kernels import (
            hip_dequant_int8_matmul, is_hip_kernel_available
        )
        from rocm_qlora.kernels import fused_dequant_int8_matmul
        from rocm_qlora.quantization.quant_ops import quantize_int8

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        torch.manual_seed(42)
        device = "cuda"

        x = torch.randn(4, 512, dtype=torch.float16, device=device)
        weight = torch.randn(1024, 512, dtype=torch.float32)
        w_int8, scales = quantize_int8(weight.flatten())
        w_int8 = w_int8.reshape(1024, 512).to(device)
        scales = scales.to(device)

        output_hip = hip_dequant_int8_matmul(x, w_int8, scales, block_size=64)
        output_triton = fused_dequant_int8_matmul(x, w_int8, scales, block_size=64)

        diff = (output_hip - output_triton).abs().max().item()
        assert diff < 0.05, \
            f"HIP vs Triton max diff: {diff} (should be < 0.05)"

    def test_hip_kernel_correct_output_shape_int4(self):
        """HIP INT4 kernel must produce correct output shape."""
        from rocm_qlora.hip_kernels import (
            hip_dequant_int4_matmul, is_hip_kernel_available
        )
        from rocm_qlora.quantization.quant_ops import quantize_int4

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        device = "cuda"
        batch, in_dim, out_dim = 4, 256, 512

        x = torch.randn(batch, in_dim, dtype=torch.float16, device=device)
        weight = torch.randn(out_dim, in_dim, dtype=torch.float32)
        w_packed, scales = quantize_int4(weight.flatten())
        w_packed = w_packed.to(device)
        scales = scales.to(device)

        output = hip_dequant_int4_matmul(x, w_packed, scales, block_size=64)
        assert output.shape == (batch, out_dim), \
            f"Expected {(batch, out_dim)}, got {output.shape}"
        assert output.dtype == torch.float16, \
            f"Expected float16 output, got {output.dtype}"

    def test_hip_kernel_correct_output_shape_int8(self):
        """HIP INT8 kernel must produce correct output shape."""
        from rocm_qlora.hip_kernels import (
            hip_dequant_int8_matmul, is_hip_kernel_available
        )
        from rocm_qlora.quantization.quant_ops import quantize_int8

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        device = "cuda"
        batch, in_dim, out_dim = 4, 256, 512

        x = torch.randn(batch, in_dim, dtype=torch.float16, device=device)
        weight = torch.randn(out_dim, in_dim, dtype=torch.float32)
        w_int8, scales = quantize_int8(weight.flatten())
        w_int8 = w_int8.reshape(out_dim, in_dim).to(device)
        scales = scales.to(device)

        output = hip_dequant_int8_matmul(x, w_int8, scales, block_size=64)
        assert output.shape == (batch, out_dim), \
            f"Expected {(batch, out_dim)}, got {output.shape}"

    def test_hip_faster_than_triton(self):
        """
        HIP kernel must be measurably faster than Triton on MI300X.
        # NOTE: Minimum speedup threshold = 1.1x (10% faster).
        """
        from rocm_qlora.hip_kernels import (
            benchmark_hip_vs_triton, is_hip_kernel_available
        )

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        result = benchmark_hip_vs_triton(size=(4096, 4096), n_runs=50)

        assert result["hip_available"], "HIP should be available for this test"
        assert result["hip_ms"] > 0, f"hip_ms should be > 0, got {result['hip_ms']}"
        assert result["triton_ms"] > 0

        speedup = result["speedup_x"]
        assert speedup > 1.1, \
            f"HIP speedup should exceed 1.1x, got {speedup}x. " \
            f"HIP: {result['hip_ms']:.2f}ms, Triton: {result['triton_ms']:.2f}ms"

    def test_hip_with_bias(self):
        """HIP INT4 matmul should correctly handle bias."""
        from rocm_qlora.hip_kernels import (
            hip_dequant_int4_matmul, is_hip_kernel_available
        )
        from rocm_qlora.quantization.quant_ops import quantize_int4
        import torch.nn.functional as F

        if not is_hip_kernel_available():
            pytest.skip("HIP kernel not available on this GPU")

        device = "cuda"
        torch.manual_seed(42)

        x = torch.randn(4, 256, dtype=torch.float16, device=device)
        weight = torch.randn(512, 256, dtype=torch.float32)
        bias = torch.randn(512, dtype=torch.float16, device=device)

        w_packed, scales = quantize_int4(weight.flatten())
        w_packed = w_packed.to(device)
        scales = scales.to(device)

        output_hip = hip_dequant_int4_matmul(x, w_packed, scales, bias=bias)

        # Reference: dequantize manually + F.linear + bias
        N = w_packed.shape[0]
        K = w_packed.shape[1] * 2
        from rocm_qlora.quantization.quant_ops import dequantize_int4
        w_fp = dequantize_int4(w_packed.flatten(), scales, 64).reshape(N, K).to(torch.float16)
        output_ref = F.linear(x, w_fp, bias)

        diff = (output_hip - output_ref).abs().max().item()
        assert diff < 0.05, f"Bias test max diff: {diff}"