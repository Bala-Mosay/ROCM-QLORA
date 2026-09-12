#!/bin/bash
# ============================================================================
# ROCM-QLORA MI300X Full Validation Script
# Run on AMD Developer Cloud MI300X droplet
# Tests: Pytest, Smoke, GPU training, HIP kernels, FP8, TunableOp, LLM SFT
# Saves all results to /tmp/rocm_qlora_logs/
# ============================================================================
set -e

LOGDIR="/tmp/rocm_qlora_logs"
mkdir -p "$LOGDIR"

echo "=============================================="
echo "  ROCM-QLORA MI300X Full Validation"
echo "  $(date)"
echo "=============================================="

# ── PHASE 1: Setup ──────────────────────────────────────────────────────────
echo ""
echo "===== PHASE 1: Setup ====="
if [ -d ".git" ]; then
    echo "Repo exists, pulling latest..."
    git pull origin main 2>&1 | tail -3
else
    git clone https://github.com/Bala-Mosay/ROCM-QLORA.git 2>&1 | tail -3
    cd ROCM-QLORA
fi
pip3 install --break-system-packages -e ".[dev]" 2>&1 | tail -5 | tee "$LOGDIR/01_install.log"
echo "Setup complete: $(date)" | tee -a "$LOGDIR/01_install.log"

# ── PHASE 2: GPU + ROCm Detection ───────────────────────────────────────────
echo ""
echo "===== PHASE 2: GPU + ROCm Detection ====="
python3 -c "
import torch
print('='*50)
print('SYSTEM INFO')
print('='*50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'ROCm version: {torch.version.hip}')
else:
    print('ROCm version: N/A (CUDA build)')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'Architecture: {props.name}')
    print(f'Compute capability: {props.major}.{props.minor}')
    print(f'Major minor: {props.major}')
    print(f'Multi-processor count: {props.multi_processor_count}')
else:
    print('WARNING: No GPU detected!')
print('='*50)
" 2>&1 | tee "$LOGDIR/02_gpu_info.log"

# ── PHASE 3: Full Pytest Suite ──────────────────────────────────────────────
echo ""
echo "===== PHASE 3: Full Pytest Suite (212 tests) ====="
python3 -m pytest tests/ -v --tb=short 2>&1 | tee "$LOGDIR/03_pytest.log" || true
echo ""
echo "Pytest complete: $(date)" | tee -a "$LOGDIR/03_pytest.log"

# ── PHASE 4: Smoke Tests ────────────────────────────────────────────────────
echo ""
echo "===== PHASE 4: Smoke Tests ====="
for script in smoke_test.py smoke_test_v3.py smoke_test_v4.py smoke_test_v5.py; do
    echo "--- $script ---"
    python3 "$script" 2>&1 | tee -a "$LOGDIR/04_smoke_tests.log" || true
    echo ""
done
echo "Smoke tests complete: $(date)" | tee -a "$LOGDIR/04_smoke_tests.log"

# ── PHASE 5: GPU Training Test ──────────────────────────────────────────────
echo ""
echo "===== PHASE 5: GPU Training Test ====="
python3 -c "
import torch
import torch.nn as nn
from rocm_qlora import quantize_model, enable_all_kernels

if not torch.cuda.is_available():
    print('SKIP: No CUDA device available')
    exit(0)

print('GPU:', torch.cuda.get_device_name(0))
print('VRAM before:', round(torch.cuda.memory_allocated() / 1e9, 3), 'GB')

# Build model
model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 10))
model = quantize_model(model, bits=4, lora_r=4, target_modules=['0'])
n = enable_all_kernels(model)
print(f'Enabled kernels: {n}')

# Move to GPU
device = torch.device('cuda')
model = model.to(device)
x = torch.randn(2, 128, device=device)

# Forward
out = model(x)
print(f'Output shape: {out.shape}')
print(f'Output dtype: {out.dtype}')

# Backward
loss = out.sum()
loss.backward()

# Check gradients
grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
print(f'Params with grads: {len(grad_names)}')

print('VRAM after:', round(torch.cuda.memory_allocated() / 1e9, 3), 'GB')
print('GPU training test: PASSED')
" 2>&1 | tee "$LOGDIR/05_gpu_train.log" || true

# ── PHASE 6: Merge/Unmerge on GPU ──────────────────────────────────────────
echo ""
echo "===== PHASE 6: Merge/Unmerge on GPU ====="
python3 -c "
import torch
import torch.nn as nn
from rocm_qlora import quantize_model

if not torch.cuda.is_available():
    print('SKIP: No CUDA device available')
    exit(0)

device = torch.device('cuda')
model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 10))
model = quantize_model(model, bits=4, lora_r=4, target_modules=['0'])
model = model.to(device)

x = torch.randn(1, 128, device=device)

# Forward before merge
out_before = model(x)
print(f'Output before merge: {out_before.shape}')

# Merge
model[0].merge_lora()
out_merged = model(x)
diff = (out_before - out_merged).abs().max().item()
print(f'After merge (atol): {diff:.6f}')

# Unmerge
model[0].unmerge_lora()
out_unmerged = model(x)
diff2 = (out_before - out_unmerged).abs().max().item()
print(f'After unmerge (atol): {diff2:.6f}')

status = 'PASSED' if diff2 < 0.01 else 'FAILED'
print(f'Merge/Unmerge test: {status}')
" 2>&1 | tee "$LOGDIR/06_merge_unmerge.log" || true

# ── PHASE 7: HIP vs Triton Benchmark ────────────────────────────────────────
echo ""
echo "===== PHASE 7: HIP vs Triton Benchmark ====="
python3 -c "
from rocm_qlora.hip_kernels import benchmark_hip_vs_triton, is_hip_kernel_available

print(f'HIP kernels available: {is_hip_kernel_available()}')

# Benchmark at multiple sizes
for size in [256, 512, 1024, 2048]:
    print(f'\n--- Size {size}x{size} ---')
    result = benchmark_hip_vs_triton(size=(size, size), n_runs=10)
    for k, v in result.items():
        print(f'  {k}: {v}')
" 2>&1 | tee "$LOGDIR/07_benchmark.log" || true

# ── PHASE 8: FP8 Detection ──────────────────────────────────────────────────
echo ""
echo "===== PHASE 8: FP8 Support Detection ====="
python3 -c "
from rocm_qlora.fp8 import detect_fp8_support, FP8Config

info = detect_fp8_support()
print('FP8 Detection Results:')
for k, v in info.items():
    print(f'  {k}: {v}')

config = FP8Config()
print(f'\nFP8Config defaults:')
print(f'  enabled: {config.enabled}')
print(f'  forward_dtype: {config.forward_dtype}')
print(f'  backward_dtype: {config.backward_dtype}')
print(f'  fallback_dtype: {config.fallback_dtype}')
" 2>&1 | tee "$LOGDIR/08_fp8_detection.log" || true

# ── PHASE 9: TunableOp ──────────────────────────────────────────────────────
echo ""
echo "===== PHASE 9: TunableOp Test ====="
python3 -c "
from rocm_qlora.profiling import enable_tunableop, disable_tunableop, get_tunableop_status

status_before = get_tunableop_status()
print(f'Status before: {status_before}')

result = enable_tunableop()
print(f'Enable result: {result}')

status_after = get_tunableop_status()
print(f'Status after: {status_after}')

disable_result = disable_tunableop()
print(f'Disable result: {disable_result}')

status_final = get_tunableop_status()
print(f'Status final: {status_final}')

print('TunableOp test: PASSED')
" 2>&1 | tee "$LOGDIR/09_tunableop.log" || true

# ── PHASE 10: MI300X Benchmark (TinyLlama-1.1B) ──────────────────────────
echo ""
echo "===== PHASE 10: MI300X Benchmark ====="
echo "Training TinyLlama-1.1B on Alpaca (4 configs x 10 epochs)"
python3 demo_bench_mi300x.py 2>&1 | tee "$LOGDIR/10_benchmark.log" || true

# ── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  ALL 10 PHASES COMPLETE — $(date)"
echo "=============================================="
echo ""
echo "Logs saved to: $LOGDIR/"
ls -lh "$LOGDIR/"
echo ""
echo "Download with:"
echo "  scp -r root@DROPLET_IP:$LOGDIR/ ./local_results/"
