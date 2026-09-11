#!/usr/bin/env python3
"""
V5 CPU Smoke Test — HIP INT4/INT8 Matmul Kernels

Validates that the V5 HIP kernel module:
  1. Imports without errors
  2. Has correct public API exports
  3. Gracefully falls back to Triton/V1 on CPU
  4. enable_hip_kernels() returns 0 on CPU (no crash)
  5. All three fallback tiers (HIP→Triton→PyTorch) are accessible
  6. Output correctness matches V1 reference

No GPU required. No internet downloads. Safe for CI.

Usage:
    python smoke_test_v5.py
"""

import os
import sys
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# --- Global tracking ---
PASSED = 0
FAILED = 0
ERRORS = []  # list of (check_name, error_message)


def check(name, condition, detail=""):
    """Record a pass/fail check."""
    global PASSED, FAILED, ERRORS
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        msg = f"{name}: {detail}"
        ERRORS.append(msg)
        print(f"  [FAIL] {name} — {detail}")


def section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =========================================================================
# Test 1: Module Importability
# =========================================================================

section("Test 1: Module Importability")

try:
    import rocm_qlora.hip_kernels
    _import_main_ok = True
except Exception:
    traceback.print_exc()
    _import_main_ok = False
check("rocm_qlora.hip_kernels import", _import_main_ok)

try:
    from rocm_qlora.hip_kernels import (
        is_hip_kernel_available,
        enable_hip_kernels,
        hip_dequant_int4_matmul,
        hip_dequant_int8_matmul,
        benchmark_hip_vs_triton,
    )
    _api_import_ok = True
except Exception:
    traceback.print_exc()
    _api_import_ok = False
check("Public API functions importable", _api_import_ok)

try:
    from rocm_qlora.hip_kernels import build_utils, int4_matmul
    _submodule_ok = True
except Exception:
    traceback.print_exc()
    _submodule_ok = False
check("Submodules (build_utils, int4_matmul) import", _submodule_ok)


# =========================================================================
# Test 2: Public API Signatures
# =========================================================================

section("Test 2: Public API Signatures")

try:
    from rocm_qlora.hip_kernels import (
        is_hip_kernel_available,
        enable_hip_kernels,
        hip_dequant_int4_matmul,
        hip_dequant_int8_matmul,
        benchmark_hip_vs_triton,
    )
except ImportError:
    print("  [SKIP] API not available — import test already failed")
    check("is_hip_kernel_available is callable", False, "import failed")
    check("enable_hip_kernels is callable", False, "import failed")
    check("hip_dequant_int4_matmul is callable", False, "import failed")
    check("hip_dequant_int8_matmul is callable", False, "import failed")
    check("benchmark_hip_vs_triton is callable", False, "import failed")
else:
    check("is_hip_kernel_available() is callable", callable(is_hip_kernel_available))
    check("enable_hip_kernels() is callable", callable(enable_hip_kernels))
    check("hip_dequant_int4_matmul() is callable", callable(hip_dequant_int4_matmul))
    check("hip_dequant_int8_matmul() is callable", callable(hip_dequant_int8_matmul))
    check("benchmark_hip_vs_triton() is callable", callable(benchmark_hip_vs_triton))


# =========================================================================
# Test 3: CPU Fallback — INT4 Matmul
# =========================================================================

section("Test 3: INT4 CPU Fallback Matmul")

try:
    from rocm_qlora.hip_kernels import hip_dequant_int4_matmul
    from rocm_qlora.quantization.quant_ops import quantize_int4, dequantize_int4

    torch.manual_seed(42)
    x = torch.randn(4, 256, dtype=torch.float32)
    weight = torch.randn(512, 256, dtype=torch.float32)
    w_packed, scales = quantize_int4(weight.flatten())

    # Run HIP path (will use V1 fallback on CPU)
    output = hip_dequant_int4_matmul(x, w_packed, scales, block_size=64)
    check("INT4 shape correct", output.shape == (4, 512),
          f"got {output.shape}")

    # Reference via V1 dequant + F.linear
    # w_packed is 1D from quantize_int4, infer N/K
    K = x.shape[-1]
    N = w_packed.numel() * 2 // K
    w_fp16 = dequantize_int4(w_packed.flatten(), scales, 64).reshape(N, K).to(torch.float32)
    output_ref = F.linear(x, w_fp16)

    max_diff = (output - output_ref).abs().max().item()
    check("INT4 output matches V1 reference", max_diff < 0.05,
          f"max diff = {max_diff:.6f}")

except Exception as e:
    traceback.print_exc()
    check("INT4 fallback test ran", False, str(e))


# =========================================================================
# Test 4: CPU Fallback — INT8 Matmul
# =========================================================================

section("Test 4: INT8 CPU Fallback Matmul")

try:
    from rocm_qlora.hip_kernels import hip_dequant_int8_matmul
    from rocm_qlora.quantization.quant_ops import quantize_int8, dequantize_int8

    torch.manual_seed(42)
    x = torch.randn(4, 256, dtype=torch.float32)
    weight = torch.randn(512, 256, dtype=torch.float32)
    w_int8, scales = quantize_int8(weight.flatten())
    w_int8 = w_int8.reshape(512, 256)

    output = hip_dequant_int8_matmul(x, w_int8, scales, block_size=64)
    check("INT8 shape correct", output.shape == (4, 512),
          f"got {output.shape}")

    w_fp16 = dequantize_int8(w_int8, scales, 64).to(torch.float32)
    output_ref = F.linear(x, w_fp16)

    max_diff = (output - output_ref).abs().max().item()
    check("INT8 output matches V1 reference", max_diff < 0.05,
          f"max diff = {max_diff:.6f}")

except Exception as e:
    traceback.print_exc()
    check("INT8 fallback test ran", False, str(e))


# =========================================================================
# Test 5: enable_hip_kernels() on CPU model
# =========================================================================

section("Test 5: enable_hip_kernels() on CPU")

try:
    from rocm_qlora.hip_kernels import enable_hip_kernels
    from rocm_qlora.quantization.quant_linear import QuantLinear

    # Build a trivial QuantLinear model
    q1 = QuantLinear(64, 128, bits=4)
    q2 = QuantLinear(128, 256, bits=8)
    model = torch.nn.Sequential(q1, torch.nn.ReLU(), q2)

    # Seed fake quantized weights
    q1.weight_quant = torch.zeros(128 * 64 // 2, dtype=torch.uint8)
    q1.weight_scales = torch.ones((128 * 64) // 64)
    q2.weight_quant = torch.zeros(256 * 128, dtype=torch.int8)
    q2.weight_scales = torch.ones((256 * 128) // 64)

    count = enable_hip_kernels(model)
    check("enable_hip_kernels returns int", isinstance(count, int))
    # On CPU, should return 0 (no HIP kernel available)
    check("enable_hip_kernels returns 0 on CPU", count == 0,
          f"got {count} (expected 0 on CPU)")

    # Check use_hip_kernel flag was set on both layers
    flags_ok = all(
        hasattr(m, "use_hip_kernel") and not m.use_hip_kernel
        for m in model if isinstance(m, QuantLinear)
    )
    check("use_hip_kernel flag set (False) on QuantLinear layers",
          flags_ok, "flag missing or True on CPU")

except Exception as e:
    traceback.print_exc()
    check("enable_hip_kernels test ran", False, str(e))


# =========================================================================
# Test 6: is_hip_kernel_available() returns False
# =========================================================================

section("Test 6: HIP availability check")

try:
    from rocm_qlora.hip_kernels import is_hip_kernel_available
    available = is_hip_kernel_available()
    check("is_hip_kernel_available() returns bool", isinstance(available, bool))
    # On CPU without HIP, should be False
    # (May be True on MI300X dev machine — either is acceptable)
    check("is_hip_kernel_available() does not crash", True)
except Exception as e:
    traceback.print_exc()
    check("is_hip_kernel_available() test ran", False, str(e))


# =========================================================================
# Test 7: benchmark_hip_vs_triton() on CPU
# =========================================================================

section("Test 7: benchmark_hip_vs_triton()")

try:
    from rocm_qlora.hip_kernels import benchmark_hip_vs_triton
    result = benchmark_hip_vs_triton(size=(128, 128), n_runs=3)
    check("benchmark returns dict", isinstance(result, dict))
    expected_keys = {"hip_ms", "triton_ms", "speedup_x", "winner", "hip_available"}
    missing = expected_keys - set(result.keys())
    check("benchmark dict has all expected keys", len(missing) == 0,
          f"missing: {missing}")
    check("triton_ms > 0", result["triton_ms"] > 0,
          f"triton_ms = {result['triton_ms']}")
    check("winner is 'hip' or 'triton'", result["winner"] in ("hip", "triton"),
          f"winner = {result['winner']}")

except Exception as e:
    traceback.print_exc()
    check("benchmark test ran", False, str(e))


# =========================================================================
# Test 8: build_utils checks
# =========================================================================

section("Test 8: build_utils infrastructure")

try:
    from rocm_qlora.hip_kernels.build_utils import (
        find_hipcc,
        is_hipcc_available,
        compile_hip_kernel,
        get_or_compile_int4_kernel,
    )
    hipcc = find_hipcc()
    check("find_hipcc() returns None or str", hipcc is None or isinstance(hipcc, str))
    has_hipcc = is_hipcc_available()
    check("is_hipcc_available() returns bool", isinstance(has_hipcc, bool))

    # Try get_or_compile — should not crash
    lib = get_or_compile_int4_kernel()
    check("get_or_compile_int4_kernel() does not crash", True)
    check("get_or_compile returns None or CDLL", lib is None or hasattr(lib, "launch_int4_dequant_matmul"))

except Exception as e:
    traceback.print_exc()
    check("build_utils test ran", False, str(e))


# =========================================================================
# Test 9: HIP source file exists
# =========================================================================

section("Test 9: HIP source file integrity")

try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    hip_file = os.path.join(module_dir, "rocm_qlora", "hip_kernels", "int4_matmul.hip")
    build_py = os.path.join(module_dir, "rocm_qlora", "hip_kernels", "build_utils.py")
    wrapper_py = os.path.join(module_dir, "rocm_qlora", "hip_kernels", "int4_matmul.py")
    init_py = os.path.join(module_dir, "rocm_qlora", "hip_kernels", "__init__.py")

    check("int4_matmul.hip exists", os.path.isfile(hip_file))
    check("build_utils.py exists", os.path.isfile(build_py))
    check("int4_matmul.py exists", os.path.isfile(wrapper_py))
    check("__init__.py exists", os.path.isfile(init_py))
except Exception as e:
    traceback.print_exc()
    check("File integrity test ran", False, str(e))


# =========================================================================
# FINAL RESULT
# =========================================================================

section("FINAL RESULT")

total = PASSED + FAILED
print(f"\n  Checks: {total} total")
print(f"  Passed: {PASSED}")
print(f"  Failed: {FAILED}")

if FAILED == 0:
    print(f"\n  STATUS: READY FOR GPU VALIDATION")
    print(f"  All {PASSED}/{total} CPU checks passed.")
    print(f"  Next step: Run on MI300X with --gpu flag.")
else:
    print(f"\n  STATUS: FAILED — {FAILED}/{total} checks failed")
    for err in ERRORS:
        print(f"    • {err}")

print()

sys.exit(FAILED)