#!/bin/bash
# ============================================================================
# ROCM-QLORA Cloud Test Script
# Run on AMD GPU droplet (MI300X recommended)
# Saves all results to /tmp/*.log
# ============================================================================
set -e

LOGDIR="/tmp/rocm_qlora_logs"
mkdir -p "$LOGDIR"

echo "=============================================="
echo "  ROCM-QLORA Cloud Test — $(date)"
echo "=============================================="

# ── PHASE 1: Setup ──────────────────────────────────────────────────────────
echo ""
echo "===== PHASE 1: Installing dependencies ====="
pip install -e ".[dev]" 2>&1 | tail -10 | tee "$LOGDIR/01_install.log"

# ── PHASE 2: GPU Info ───────────────────────────────────────────────────────
echo ""
echo "===== PHASE 2: GPU Detection ====="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('HIP version:', getattr(torch.version, 'hip', 'N/A'))
print('PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('VRAM:', round(props.total_mem / 1e9, 1), 'GB')
    print('Compute capability:', props.major, '.', props.minor)
    print('Architecture:', props.name)
else:
    print('WARNING: No CUDA device detected. GPU tests will be skipped.')
" 2>&1 | tee "$LOGDIR/02_gpu_info.log"

# ── PHASE 3: Pytest ─────────────────────────────────────────────────────────
echo ""
echo "===== PHASE 3: Full Pytest Suite ====="
python -m pytest tests/ -v --tb=short 2>&1 | tee "$LOGDIR/03_pytest.log" || true

# ── PHASE 4: Smoke Tests ────────────────────────────────────────────────────
echo ""
echo "===== PHASE 4: Smoke Tests ====="
echo "--- smoke_test.py ---"
python smoke_test.py 2>&1 | tee "$LOGDIR/04a_smoke_v1.log" || true

echo "--- smoke_test_v3.py ---"
python smoke_test_v3.py 2>&1 | tee "$LOGDIR/04b_smoke_v3.log" || true

echo "--- smoke_test_v4.py ---"
python smoke_test_v4.py 2>&1 | tee "$LOGDIR/04c_smoke_v4.log" || true

echo "--- smoke_test_v5.py ---"
python smoke_test_v5.py 2>&1 | tee "$LOGDIR/04d_smoke_v5.log" || true

# ── PHASE 5: GPU Training Test ──────────────────────────────────────────────
echo ""
echo "===== PHASE 5: GPU Training Test ====="
python -c "
import torch
import torch.nn as nn
from rocm_qlora import quantize_model, enable_all_kernels
from rocm_qlora.optim import PagedAdamW

if not torch.cuda.is_available():
    print('SKIP: No CUDA device available')
    exit(0)

print('Device: cuda')
print('GPU:', torch.cuda.get_device_name(0))

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

# Backward
loss = out.sum()
loss.backward()

# Check gradients
grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
print(f'Params with grads: {len(grad_names)}')
print(f'Grad params: {grad_names[:5]}')

print('GPU training test: PASSED')
" 2>&1 | tee "$LOGDIR/05_gpu_train.log" || true

# ── PHASE 6: Benchmark ──────────────────────────────────────────────────────
echo ""
echo "===== PHASE 6: HIP vs Triton Benchmark ====="
python -c "
from rocm_qlora.hip_kernels import benchmark_hip_vs_triton
result = benchmark_hip_vs_triton()
for k, v in result.items():
    print(f'{k}: {v}')
" 2>&1 | tee "$LOGDIR/06_benchmark.log" || true

# ── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  ALL PHASES COMPLETE — $(date)"
echo "=============================================="
echo "Logs saved to: $LOGDIR/"
ls -la "$LOGDIR/"
echo ""
echo "Download with:"
echo "  scp -r root@DROPLET_IP:$LOGDIR/ ./local_results/"
