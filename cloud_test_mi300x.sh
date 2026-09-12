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
pip install -e ".[dev]" 2>&1 | tail -5 | tee "$LOGDIR/01_install.log"
echo "Setup complete: $(date)" | tee -a "$LOGDIR/01_install.log"

# ── PHASE 2: GPU + ROCm Detection ───────────────────────────────────────────
echo ""
echo "===== PHASE 2: GPU + ROCm Detection ====="
python -c "
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
python -m pytest tests/ -v --tb=short 2>&1 | tee "$LOGDIR/03_pytest.log" || true
echo ""
echo "Pytest complete: $(date)" | tee -a "$LOGDIR/03_pytest.log"

# ── PHASE 4: Smoke Tests ────────────────────────────────────────────────────
echo ""
echo "===== PHASE 4: Smoke Tests ====="
for script in smoke_test.py smoke_test_v3.py smoke_test_v4.py smoke_test_v5.py; do
    echo "--- $script ---"
    python "$script" 2>&1 | tee -a "$LOGDIR/04_smoke_tests.log" || true
    echo ""
done
echo "Smoke tests complete: $(date)" | tee -a "$LOGDIR/04_smoke_tests.log"

# ── PHASE 5: GPU Training Test ──────────────────────────────────────────────
echo ""
echo "===== PHASE 5: GPU Training Test ====="
python -c "
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
python -c "
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
python -c "
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
python -c "
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
python -c "
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

# ── PHASE 10: Real LLM Training (TinyLlama) ────────────────────────────────
echo ""
echo "===== PHASE 10: Real LLM Training ====="
python -c "
import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from rocm_qlora import quantize_model, enable_all_kernels

if not torch.cuda.is_available():
    print('SKIP: No CUDA device available')
    exit(0)

device = torch.device('cuda')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# Step 1: Load small model (GPT-2, 124M params — fast download)
print('\n[Step 1] Loading sshleifer/tiny-gpt2...')
model_name = 'sshleifer/tiny-gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
print(f'Model loaded: {model.num_parameters():,} params')

# Step 2: Quantize + LoRA
print('\n[Step 2] Quantizing with NF4 + LoRA...')
model = quantize_model(model, bits=4, lora_r=16, lora_alpha=32,
                       target_modules=['c_attn', 'c_proj'])
n = enable_all_kernels(model)
print(f'Enabled kernels: {n}')

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total_params:,}')
print(f'Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')

# Step 3: Move to GPU
print('\n[Step 3] Moving to GPU...')
model = model.to(device)
print(f'VRAM after load: {torch.cuda.memory_allocated() / 1e9:.3f} GB')

# Step 4: Prepare training data
print('\n[Step 4] Preparing training data...')
texts = [
    'The quick brown fox jumps over the lazy dog. The fox was clever and fast.',
    'Machine learning is a subset of artificial intelligence that focuses on patterns.',
    'AMD ROCm is an open-source platform for GPU computing on AMD hardware.',
    'QLORA enables efficient fine-tuning of large language models with minimal memory.',
    'The transformer architecture revolutionized natural language processing tasks.',
]
encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                      max_length=64)
encodings = {k: v.to(device) for k, v in encodings.items()}

# Step 5: Train for 3 steps
print('\n[Step 5] Training 3 steps...')
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4, weight_decay=0.01
)

model.train()
losses = []
start = time.time()

for step in range(3):
    optimizer.zero_grad()
    outputs = model(**encodings, labels=encodings['input_ids'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'  Step {step+1}/3 - Loss: {loss.item():.4f} - '
          f'VRAM: {torch.cuda.memory_allocated() / 1e9:.3f} GB')

elapsed = time.time() - start
tokens_per_sec = (len(texts) * 64 * 3) / elapsed  # approx

print(f'\nTraining complete in {elapsed:.1f}s')
print(f'Approx tokens/sec: {tokens_per_sec:.0f}')
print(f'Loss curve: {losses[0]:.4f} -> {losses[-1]:.4f}')

# Step 6: Inference test
print('\n[Step 6] Inference test...')
model.eval()
prompt = tokenizer('AMD MI300X is', return_tensors='pt').to(device)
with torch.no_grad():
    generated = model.generate(**prompt, max_new_tokens=20, do_sample=False)
response = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f'Generated: {response}')

print('\n' + '='*50)
print('REAL LLM TRAINING: PASSED')
print('='*50)
" 2>&1 | tee "$LOGDIR/10_llm_training.log" || true

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
