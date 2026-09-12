"""
rocm-qlora MI300X Benchmark Demo
=================================
Trains TinyLlama-1.1B on Alpaca with different configurations.
Compares: Triton vs no Triton, FP8 vs BF16, TunableOp vs default.

Usage:
    export HF_TOKEN=hf_...
    python3 demo_bench_mi300x.py
"""

import os
import sys
import json
import time
import copy
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_SIZE = 500       # Alpaca samples to use
MAX_SEQ_LEN = 512
NUM_EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
SEED = 42
LOG_EVERY = 20  # steps between log lines


@dataclass
class BenchResult:
    config_name: str
    total_time_s: float = 0.0
    peak_vram_gb: float = 0.0
    final_loss: float = 0.0
    loss_curve: List[float] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    trainable_params: int = 0
    total_params: int = 0
    notes: str = ""


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_alpaca(sample):
    if sample["input"].strip():
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def prepare_alpaca(tokenizer, max_length=512, num_samples=500):
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_samples}]")

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for sample in raw:
        text = format_alpaca(sample)
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[ids == tokenizer.pad_token_id] = -100

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(labels)

    dataset = TensorDataset(
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list),
    )
    return dataset


def run_training_config(
    config_name: str,
    bits: int = 4,
    use_triton: bool = True,
    use_paged_opt: bool = True,
    use_packing: bool = True,
    use_fp8: bool = False,
    use_tunableop: bool = False,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    max_seq_len: int = MAX_SEQ_LEN,
    lr: float = LR,
) -> BenchResult:
    """Run a single training configuration and return metrics."""
    from rocm_qlora import quantize_model, enable_all_kernels
    from rocm_qlora.optim import PagedAdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    result = BenchResult(config_name=config_name)
    device = torch.device("cuda")

    set_seed(SEED)

    # ── Load model ──
    print(f"\n{'='*60}")
    print(f"  CONFIG: {config_name}")
    print(f"{'='*60}")
    print(f"  Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    fp16_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  FP16 VRAM: {fp16_vram:.2f} GB")

    # ── Quantize ──
    print(f"  Quantizing to NF4 + LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    model = quantize_model(
        model,
        bits=bits,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )

    result.total_params = sum(p.numel() for p in model.parameters())
    result.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {result.total_params:,} total, {result.trainable_params:,} trainable "
          f"({100*result.trainable_params/result.total_params:.2f}%)")

    # ── Enable kernels ──
    if use_triton:
        n = enable_all_kernels(model)
        print(f"  Triton kernels: {n} layers enabled")
    else:
        print(f"  Triton kernels: DISABLED")

    # ── FP8 ──
    if use_fp8:
        from rocm_qlora.fp8 import detect_fp8_support, wrap_model_for_fp8, FP8Config
        fp8_info = detect_fp8_support()
        if fp8_info["fp8_supported"]:
            fp8_config = FP8Config(
                use_transformer_engine=fp8_info["transformer_engine_available"]
            )
            model, wrap_info = wrap_model_for_fp8(model, fp8_config)
            if wrap_info["enabled"]:
                print(f"  FP8: ENABLED ({wrap_info['backend']}, {wrap_info['layers_wrapped']} layers)")
                result.notes = f"FP8 backend: {wrap_info['backend']}"
            else:
                print(f"  FP8: SKIPPED ({wrap_info['reason']})")
                use_fp8 = False
        else:
            print(f"  FP8: NOT SUPPORTED ({fp8_info['reason']})")
            use_fp8 = False

    # ── TunableOp ──
    if use_tunableop:
        from rocm_qlora.profiling import enable_tunableop, switch_to_load_only, get_tunableop_status
        tun_result = enable_tunableop("./tunableop_cache")
        print(f"  TunableOp: ENABLED (cache: {tun_result['cache_path']})")

    # ── Move to GPU ──
    model = model.to(device)
    model.gradient_checkpointing_enable()

    post_move_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after quant+move: {post_move_vram:.2f} GB")

    # ── Prepare data ──
    print(f"  Preparing Alpaca ({DATASET_SIZE} samples)...")
    dataset = prepare_alpaca(tokenizer, max_length=max_seq_len, num_samples=DATASET_SIZE)

    if use_packing:
        from rocm_qlora.data import build_packed_dataset, PackedSequenceCollator, compute_packing_efficiency
        texts = []
        for sample in dataset:
            # Reconstruct text from tokens (for packing we need raw text)
            pass
        # Packing needs raw texts - reload from dataset
        from datasets import load_dataset
        raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{DATASET_SIZE}]")
        texts = [format_alpaca(s) for s in raw]
        packed, efficiency = build_packed_dataset(texts, tokenizer, max_length=max_seq_len)
        collator = PackedSequenceCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        dataloader = DataLoader(packed, batch_size=batch_size, collate_fn=collator, shuffle=True)
        print(f"  Packing: {efficiency['efficiency_pct']:.1f}% efficiency, "
              f"{efficiency['original_samples']}->{efficiency['packed_samples']} packs")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"  Packing: DISABLED")

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if use_paged_opt:
        optimizer = PagedAdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"  Optimizer: PagedAdamW")
    else:
        from torch.optim import AdamW
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"  Optimizer: AdamW (standard)")

    total_steps = (len(dataloader) // grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(10, total_steps // 10), num_training_steps=total_steps
    )

    print(f"  Steps: {total_steps} ({len(dataloader)} batches/epoch, {grad_accum} grad_accum)")
    print(f"  Training...")

    # ── Training loop ──
    model.train()
    global_step = 0
    total_tokens = 0
    start_time = time.time()
    peak_vram = 0.0
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            # Count non-ignored tokens for throughput
            labels = batch.get("labels", batch.get("input_ids"))
            total_tokens += (labels != -100).sum().item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_loss = loss.item() * grad_accum
                epoch_loss += current_loss
                epoch_steps += 1

                if global_step % LOG_EVERY == 0:
                    vram = torch.cuda.memory_allocated() / 1e9
                    peak_vram = max(peak_vram, vram)
                    elapsed = time.time() - start_time
                    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"    Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                          f"Loss: {current_loss:.4f} | LR: {lr_now:.2e} | "
                          f"VRAM: {vram:.2f} GB | {tok_per_sec:.0f} tok/s")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_losses.append(avg_epoch_loss)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_epoch_loss:.4f} | "
              f"Elapsed: {elapsed:.1f}s")

        # TunableOp: switch to load-only after first epoch
        if use_tunableop and epoch == 0:
            from rocm_qlora.profiling import switch_to_load_only
            switch_to_load_only()
            print(f"  TunableOp: switched to load-only mode")

    total_time = time.time() - start_time
    final_loss = epoch_losses[-1] if epoch_losses else 0.0

    result.total_time_s = round(total_time, 1)
    result.peak_vram_gb = round(peak_vram, 3)
    result.final_loss = round(final_loss, 4)
    result.loss_curve = [round(l, 4) for l in epoch_losses]
    result.total_tokens = total_tokens
    result.tokens_per_sec = round(total_tokens / total_time, 1) if total_time > 0 else 0

    print(f"\n  RESULT: {config_name}")
    print(f"    Time: {result.total_time_s}s | Peak VRAM: {result.peak_vram_gb} GB")
    print(f"    Loss: {result.loss_curve[0]:.4f} -> {result.final_loss:.4f}")
    print(f"    Throughput: {result.tokens_per_sec:.0f} tokens/sec")

    # Cleanup
    del model, optimizer, scheduler, dataloader
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return result


def print_comparison_table(results: List[BenchResult]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  MI300X BENCHMARK RESULTS — TinyLlama-1.1B on Alpaca ({NUM_EPOCHS} epochs)")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"{'='*80}")

    # Header
    header = f"{'Config':<28} {'Time':>8} {'tok/s':>8} {'Peak VRAM':>10} {'Loss Start':>11} {'Loss Final':>11}"
    print(header)
    print("-" * 80)

    for r in results:
        loss_start = r.loss_curve[0] if r.loss_curve else 0.0
        row = (
            f"{r.config_name:<28} "
            f"{r.total_time_s:>7.1f}s "
            f"{r.tokens_per_sec:>7.0f} "
            f"{r.peak_vram_gb:>9.2f}GB "
            f"{loss_start:>10.4f} "
            f"{r.final_loss:>10.4f}"
        )
        print(row)

    print("-" * 80)

    # Speedup analysis
    if len(results) >= 2:
        baseline = results[0]  # Full config (Triton)
        print(f"\n  Speedup Analysis:")
        for r in results[1:]:
            if baseline.tokens_per_sec > 0 and r.tokens_per_sec > 0:
                speedup = baseline.tokens_per_sec / r.tokens_per_sec
                if speedup > 1:
                    print(f"    {baseline.config_name} is {speedup:.2f}x faster than {r.config_name}")
                else:
                    print(f"    {r.config_name} is {1/speedup:.2f}x faster than {baseline.config_name}")

    # Loss convergence
    print(f"\n  Loss Convergence:")
    for r in results:
        if len(r.loss_curve) >= 2:
            improvement = r.loss_curve[0] - r.final_loss
            pct = (improvement / r.loss_curve[0]) * 100 if r.loss_curve[0] != 0 else 0
            print(f"    {r.config_name}: {r.loss_curve[0]:.4f} -> {r.final_loss:.4f} "
                  f"(↓{improvement:.4f}, {pct:.1f}% reduction)")

    print(f"{'='*80}\n")


def main():
    print("=" * 60)
    print("  rocm-qlora MI300X Benchmark")
    print(f"  {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB VRAM")
    print(f"  PyTorch {torch.__version__} | "
          f"ROCm {torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else 'N/A'}")
    print("=" * 60)

    results = []

    # Config A: Full features (Triton + PagedAdamW + packing)
    results.append(run_training_config(
        config_name="NF4+LoRA+Triton+Paged+Pack",
        bits=4,
        use_triton=True,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=False,
        use_tunableop=False,
    ))

    # Config B: No Triton, no paged opt, no packing (baseline comparison)
    results.append(run_training_config(
        config_name="NF4+LoRA (no Triton)",
        bits=4,
        use_triton=False,
        use_paged_opt=False,
        use_packing=False,
        use_fp8=False,
        use_tunableop=False,
    ))

    # Config C: Triton + FP8 (MI300X FP8)
    results.append(run_training_config(
        config_name="NF4+LoRA+Triton+FP8",
        bits=4,
        use_triton=True,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=True,
        use_tunableop=False,
    ))

    # Config D: Triton + TunableOp
    results.append(run_training_config(
        config_name="NF4+LoRA+Triton+TunableOp",
        bits=4,
        use_triton=True,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=False,
        use_tunableop=True,
    ))

    # Print comparison table
    print_comparison_table(results)

    # Save results to JSON
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mi300x_benchmark.json")

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
        "pytorch_version": torch.__version__,
        "rocm_version": torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else "N/A",
        "model": MODEL_ID,
        "dataset": f"Alpaca ({DATASET_SIZE} samples)",
        "epochs": NUM_EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
        "configs": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
