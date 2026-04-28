# rocm-qlora

![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen)
![ROCm](https://img.shields.io/badge/ROCm-6.2%2B-red)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

QLoRA fine-tuning for AMD GPUs — no bitsandbytes, no CUDA, 117 tests passing.

## The Problem

bitsandbytes has CUDA-compiled kernels at its core. The ROCm port is broken and unmaintained, leaving AMD GPU owners locked out of QLoRA fine-tuning entirely. This project solves it with pure PyTorch — every quantization kernel, optimizer, and attention kernel runs natively on ROCm without any CUDA dependencies.

## Supported Hardware

| GPU | Architecture | ROCm | Status |
|-----|--------------|------|--------|
| RX 7900 XTX / XT / GRE | gfx1100 (RDNA3) | 6.2+ | ✅ Supported |
| MI300X | gfx942 (CDNA3) | 6.2+ | ✅ Supported + FP8 |
| MI300A | gfx940 (CDNA3) | 6.2+ | ✅ Supported + FP8 |
| MI250 / MI250X | gfx90a (CDNA2) | 6.0+ | ⚠️ No FP8 |
| RX 6000 series | gfx1030 (RDNA2) | 6.0+ | ⚠️ Untested |

## Installation

```bash
# Install PyTorch with ROCm 6.2 support
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.2
```

```bash
git clone https://github.com/yourusername/rocm-qlora
cd rocm-qlora
pip install -e .

# Verify installation
python smoke_test_v4.py  # should print 20/20 passed
```

## Quick Start

```python
from transformers import AutoModelForCausalLM
from rocm_qlora import quantize_model, check_rocm

check_rocm()  # prints GPU info

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16
)
model = quantize_model(model, bits=8, lora_r=8,
                       target_modules=["q_proj", "v_proj"])
model = model.to("cuda")
# Ready for fine-tuning. Base weights frozen. Only LoRA trains.
```

## What's Inside

| Module | What it does |
|--------|---------------|
| rocm_qlora.quantization | NF4/INT8 blockwise quantization + double quantization |
| rocm_qlora.lora | LoRALinear with zero-init contract and gradient isolation |
| rocm_qlora.kernels | Triton fused dequant+matmul (RDNA3 + CDNA3 configs) |
| rocm_qlora.optim | PagedAdamW — 0 VRAM optimizer states (CPU pinned) |
| rocm_qlora.data | Sequence packing — 87% efficiency, 2.1x throughput |
| rocm_qlora.attention | Flash Attention 2 with training-safe backend detection |
| rocm_qlora.fp8 | FP8 training — 88% speedup on MI300X, BF16 fallback |
| rocm_qlora.distributed | FSDP multi-GPU via ROCm RCCL |
| rocm_qlora.trainers | SFT + DPO + GRPO alignment trainers |
| rocm_qlora.hf_integration | from_pretrained(quantization_config=RocmQLoraConfig()) |
| rocm_qlora.profiling | TunableOp GEMM autotuning + ROCm profiler |
| rocm_qlora.export | GGUF (Ollama) + vLLM adapter export |

## Memory Savings

| Model | FP16 | INT8 | NF4 | NF4 + Double Quant |
|-------|------|------|-----|-------------------|
| TinyLlama 1.1B | 2.2 GB | 1.1 GB | 0.6 GB | 0.55 GB |
| LLaMA-2 7B | 14.0 GB | 7.0 GB | 3.8 GB | 3.5 GB |
| LLaMA-2 13B | 26.0 GB | 13.0 GB | 7.1 GB | 6.5 GB |
| LLaMA-2 70B | 140.0 GB | 70.0 GB | 39.3 GB | 36.1 GB |

LoRA overhead: +8MB (r=8, 1B model) to +24MB (r=8, 7B model) — negligible.

## Full Pipeline

```python
# Step 1: Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=RocmQLoraConfig(bits=4, lora_r=16),
)

# Step 2: SFT fine-tuning
trainer = ROCmSFTTrainer(model, tokenizer, dataset, SFTConfig())
trainer.train()

# Step 3: DPO preference alignment
dpo = ROCmDPOTrainer(model, ref_model, tokenizer, pref_data, DPOConfig())
dpo.train()

# Step 4: GRPO reasoning (DeepSeek-R1 style — no labels needed)
grpo = ROCmGRPOTrainer(model, ref, tokenizer,
                       reward_fn=GRPOTrainer.built_in_reward_fns()["math"],
                       config=GRPOConfig())
grpo.train()

# Step 5: Export
export_lora_adapter(model, "./adapter", "meta-llama/Llama-3-8B")

# Step 6: Serve
# vllm serve meta-llama/Llama-3-8B --enable-lora --lora-modules ours=./adapter
```

### Multi-GPU

```bash
torchrun --nproc_per_node=4 train_v3_fsdp.py \
    --model_id meta-llama/Llama-3-8B \
    --bits 8 --lora_r 16
```

### All Optimizations

```bash
python train_v2.py \
    --bits 4 \
    --use_double_quant True \
    --use_fp8 True \
    --use_tunableop True \
    --use_packing True \
    --use_flash_attention True \
    --use_paged_optimizer True
```

## Project Structure

```
rocm-qlora/
├── rocm_qlora/           # Core library
│   ├── quantization/     # NF4/INT8 quantization ops
│   ├── lora/             # LoRA layer implementation
│   ├── kernels/          # Triton fused kernels
│   ├── optim/            # PagedAdamW optimizer
│   ├── data/             # Sequence packing & collators
│   ├── attention/        # Flash Attention patching
│   ├── fp8/              # FP8 training support
│   ├── distributed/      # FSDP multi-GPU training
│   ├── trainers/         # SFT, DPO, GRPO trainers
│   ├── hf_integration/  # HuggingFace plugin
│   ├── profiling/        # TunableOp & ROCm profiler
│   └── export/           # GGUF & vLLM export
├── tests/                # Test suite (117 tests)
│   ├── v2/               # V2 performance tests
│   ├── v3/               # V3 capability tests
│   └── v4/               # V4 advanced tests
├── smoke_test_v4.py      # CPU-only smoke test
├── train.py              # Basic training script
├── train_v2.py           # Optimized training script
├── train_v3_fsdp.py      # Multi-GPU FSDP training
└── demo.py               # Quick demo script
```

## Running Tests

```bash
# CPU-only smoke test (no GPU needed, no downloads)
python smoke_test_v4.py

# Full test suite
pytest tests/ -v

# By phase
pytest tests/v2/ -v  # V2 performance tests
pytest tests/v3/ -v  # V3 capability tests
pytest tests/v4/ -v  # V4 advanced tests
```

## Limitations & Known Issues

- **merge_lora() re-quantization error**: merge_lora() introduces re-quantization error (atol≈2.0) — for inference only, not training. Export pipeline uses clean dequantization path instead.

- **FP8 hardware requirement**: FP8 training requires MI300X/MI300A with ROCm 6.2+. Silently falls back to BF16 on all other hardware.

- **RDNA3 Flash Attention limitation**: RDNA3 (RX 7900) Flash Attention 2 CK backend has no backward pass — blocked automatically, torch SDPA used instead.

- **PagedAdamW page size**: No paged optimizer for gradient spikes > page_size_mb — increase --page_size_mb if OOM occurs.

- **device_map not supported**: device_map="auto" not supported — single or multi-GPU via FSDP only.

- **TunableOp + torch.compile conflict**: TunableOp and torch.compile conflict on some ROCm versions — use one at a time.

- **ROCm version requirement**: ROCm < 6.0 not supported.

## Roadmap

- [ ] HIP assembly kernels for INT4 matmul (beyond Triton)
- [ ] Multi-node training (Slurm + RCCL)
- [ ] FP8 + FSDP combined (MI300X cluster training)
- [ ] Speculative decoding for vLLM serving
- [ ] FP8 inference (not just training)
- [ ] torchao integration for additional quantization formats

## License

MIT

---

Built for the AMD community. No CUDA required.
