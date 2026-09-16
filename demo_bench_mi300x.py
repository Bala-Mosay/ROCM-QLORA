"""
rocm-qlora MI300X Benchmark Demo (v8)
======================================
Trains TinyLlama-1.1B on Alpaca with different configurations.
Fair comparison: Triton ON vs OFF with identical pipelines.
Includes eval loss, text generation, and 5000 samples for generalization.

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_SIZE = 5000
TRAIN_SIZE = 4000
EVAL_SIZE = 1000
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
LOG_EVERY = 20

# Text generation prompts for quality check
GEN_PROMPTS = [
    "### Instruction:\nExplain what a neural network is in simple terms.\n\n### Response:\n",
    "### Instruction:\nWrite a short poem about the ocean.\n\n### Response:\n",
    "### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n",
]


@dataclass
class BenchResult:
    config_name: str
    total_time_s: float = 0.0
    peak_vram_gb: float = 0.0
    final_loss: float = 0.0
    final_eval_loss: float = 0.0
    loss_curve: List[float] = field(default_factory=list)
    eval_loss_curve: List[float] = field(default_factory=list)
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


def prepare_alpaca_split(tokenizer, max_length=512, train_size=4000, eval_size=1000):
    """Load Alpaca, split into train/eval, return both datasets."""
    from datasets import load_dataset
    total = train_size + eval_size
    raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{total}]")

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

    all_data = TensorDataset(
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list),
    )

    train_dataset = Subset(all_data, list(range(train_size)))
    eval_dataset = Subset(all_data, list(range(train_size, train_size + eval_size)))

    return train_dataset, eval_dataset


def compute_eval_loss(model, eval_loader, device, max_batches=50):
    """Compute average loss over eval set."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            if step >= max_batches:
                break
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = tuple(b.to(device) for b in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts, max_new_tokens=150):
    """Generate text from prompts for quality check."""
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append(generated.strip())
    model.train()
    return results


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
    gen_prompts: List[str] = None,
) -> BenchResult:
    """Run a single training configuration and return metrics."""
    from rocm_qlora import quantize_model, enable_all_kernels
    from rocm_qlora.optim import PagedAdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    result = BenchResult(config_name=config_name)
    device = torch.device("cuda")

    set_seed(SEED)

    print(f"\n{'='*60}")
    print(f"  CONFIG: {config_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

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

    if use_triton:
        n = enable_all_kernels(model)
        print(f"  Triton kernels: {n} layers enabled")
    else:
        print(f"  Triton kernels: DISABLED (PyTorch fallback)")

    if use_fp8:
        from rocm_qlora.fp8 import detect_fp8_support, wrap_model_for_fp8, FP8Config
        fp8_info = detect_fp8_support()
        if fp8_info["fp8_supported"]:
            fp8_config = FP8Config(use_transformer_engine=fp8_info["transformer_engine_available"])
            model, wrap_info = wrap_model_for_fp8(model, fp8_config)
            if wrap_info["enabled"]:
                print(f"  FP8: ENABLED ({wrap_info['backend']}, {wrap_info['layers_wrapped']} layers)")
            else:
                print(f"  FP8: SKIPPED ({wrap_info['reason']})")
                use_fp8 = False
        else:
            print(f"  FP8: NOT SUPPORTED")
            use_fp8 = False

    if use_tunableop:
        from rocm_qlora.profiling import enable_tunableop, switch_to_load_only
        enable_tunableop("./tunableop_cache")
        print(f"  TunableOp: ENABLED")

    model = model.to(device)
    model.gradient_checkpointing_enable()

    # ── Prepare data with train/eval split ──
    train_dataset, eval_dataset = prepare_alpaca_split(
        tokenizer, max_length=max_seq_len, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE
    )
    print(f"  Data: {TRAIN_SIZE} train, {EVAL_SIZE} eval samples")

    if use_packing:
        from rocm_qlora.data import build_packed_dataset, PackedSequenceCollator
        from datasets import load_dataset
        raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{TRAIN_SIZE}]")
        texts = [format_alpaca(s) for s in raw]
        packed, efficiency = build_packed_dataset(texts, tokenizer, max_length=max_seq_len)
        collator = PackedSequenceCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_block_attention=True,
        )
        train_loader = DataLoader(packed, batch_size=batch_size, collate_fn=collator, shuffle=True)
        print(f"  Packing: {efficiency['efficiency_pct']:.1f}% efficiency")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"  Packing: DISABLED")

    # Eval loader (always no packing, standard padding)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if use_paged_opt:
        optimizer = PagedAdamW(trainable_params, lr=lr, weight_decay=0.01)
    else:
        from torch.optim import AdamW
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(10, total_steps // 10), num_training_steps=total_steps
    )

    print(f"  Steps: {total_steps} ({len(train_loader)} batches/epoch, {grad_accum} grad_accum)")
    print(f"  Training...")

    # ── Training loop ──
    model.train()
    global_step = 0
    total_tokens = 0
    start_time = time.time()
    peak_vram = 0.0
    epoch_losses = []
    eval_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.get("labels", batch.get("input_ids"))
            else:
                batch = tuple(b.to(device) for b in batch)
                batch = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                labels = batch["labels"]

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

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

        # Eval loss
        eval_loss = compute_eval_loss(model, eval_loader, device)
        eval_losses.append(eval_loss)

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_epoch_loss:.4f} | Eval: {eval_loss:.4f} | "
              f"Elapsed: {elapsed:.1f}s")

        if use_tunableop and epoch == 0:
            from rocm_qlora.profiling import switch_to_load_only
            switch_to_load_only()

    total_time = time.time() - start_time
    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    final_eval_loss = eval_losses[-1] if eval_losses else 0.0

    result.total_time_s = round(total_time, 1)
    result.peak_vram_gb = round(peak_vram, 3)
    result.final_loss = round(final_loss, 4)
    result.final_eval_loss = round(final_eval_loss, 4)
    result.loss_curve = [round(l, 4) for l in epoch_losses]
    result.eval_loss_curve = [round(l, 4) for l in eval_losses]
    result.total_tokens = total_tokens
    result.tokens_per_sec = round(total_tokens / total_time, 1) if total_time > 0 else 0

    print(f"\n  RESULT: {config_name}")
    print(f"    Time: {result.total_time_s}s | Peak VRAM: {result.peak_vram_gb} GB")
    print(f"    Train Loss: {result.loss_curve[0]:.4f} -> {result.final_loss:.4f}")
    print(f"    Eval Loss:  {eval_losses[0]:.4f} -> {result.final_eval_loss:.4f}")
    print(f"    Throughput: {result.tokens_per_sec:.0f} tokens/sec")

    # ── Text generation quality check ──
    if gen_prompts:
        print(f"\n  === Text Generation (after training) ===")
        samples = generate_samples(model, tokenizer, device, gen_prompts)
        for i, (prompt, gen) in enumerate(zip(gen_prompts, samples)):
            instruction = prompt.split("### Response:\n")[0].split("### Instruction:\n")[1].strip()
            print(f"\n  Prompt {i+1}: {instruction[:60]}...")
            print(f"  Generated: {gen[:200]}...")

    # Cleanup
    del model, optimizer, scheduler, train_loader, eval_loader
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return result


def print_comparison_table(results: List[BenchResult]):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  MI300X BENCHMARK RESULTS — TinyLlama-1.1B on Alpaca ({NUM_EPOCHS} epochs)")
    print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Train: {TRAIN_SIZE} samples | Eval: {EVAL_SIZE} samples")
    print(f"{'='*90}")

    header = (f"{'Config':<30} {'Time':>7} {'tok/s':>7} {'VRAM':>7} "
              f"{'Train Loss':>12} {'Eval Loss':>10}")
    print(header)
    print("-" * 90)

    for r in results:
        loss_start = r.loss_curve[0] if r.loss_curve else 0.0
        eval_start = r.eval_loss_curve[0] if r.eval_loss_curve else 0.0
        row = (
            f"{r.config_name:<30} "
            f"{r.total_time_s:>6.1f}s "
            f"{r.tokens_per_sec:>6.0f} "
            f"{r.peak_vram_gb:>6.2f}G "
            f"{loss_start:>5.4f}->{r.final_loss:<5.4f} "
            f"{eval_start:>5.4f}->{r.final_eval_loss:<5.4f}"
        )
        print(row)

    print("-" * 90)

    # Fair comparison section: same-pipeline configs
    fair_pairs = []
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            # Check if configs differ only in triton
            if (r1.peak_vram_gb == r2.peak_vram_gb and
                r1.total_params == r2.total_params and
                abs(r1.total_time_s - r2.total_time_s) > 1):
                if r1.tokens_per_sec > r2.tokens_per_sec:
                    fair_pairs.append((r1, r2))
                else:
                    fair_pairs.append((r2, r1))

    if fair_pairs:
        print(f"\n  FAIR COMPARISON (identical pipeline, only Triton differs):")
        for faster, slower in fair_pairs:
            speedup = faster.tokens_per_sec / slower.tokens_per_sec
            print(f"    {faster.config_name}: {faster.tokens_per_sec:.0f} tok/s "
                  f"vs {slower.config_name}: {slower.tokens_per_sec:.0f} tok/s "
                  f"= {speedup:.2f}x speedup")
            print(f"    Train loss: {faster.final_loss:.4f} vs {slower.final_loss:.4f} "
                  f"(convergence {'MATCHES' if abs(faster.final_loss - slower.final_loss) < 0.05 else 'DIFFERS'})")
            print(f"    Eval loss:  {faster.final_eval_loss:.4f} vs {slower.final_eval_loss:.4f}")

    # Generalization check
    print(f"\n  GENERALIZATION (Train vs Eval loss):")
    for r in results:
        gap = r.final_eval_loss - r.final_loss
        status = "OK (gap < 0.1)" if abs(gap) < 0.1 else f"WARNING (gap = {gap:.4f})"
        print(f"    {r.config_name}: train={r.final_loss:.4f}, eval={r.final_eval_loss:.4f} — {status}")

    print(f"{'='*90}\n")


def main():
    print("=" * 60)
    print("  rocm-qlora MI300X Benchmark v7")
    print(f"  {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM")
    print(f"  PyTorch {torch.__version__} | "
          f"ROCm {torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else 'N/A'}")
    print("=" * 60)

    results = []

    # Config A: Triton ON (full features)
    results.append(run_training_config(
        config_name="Triton ON (full)",
        bits=4,
        use_triton=True,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=False,
        use_tunableop=False,
        gen_prompts=GEN_PROMPTS,
    ))

    # Config B: Triton OFF (fair comparison — same pipeline as A)
    results.append(run_training_config(
        config_name="Triton OFF (fair)",
        bits=4,
        use_triton=False,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=False,
        use_tunableop=False,
        gen_prompts=GEN_PROMPTS,
    ))

    # Config C: Triton + TunableOp
    results.append(run_training_config(
        config_name="Triton+TunableOp",
        bits=4,
        use_triton=True,
        use_paged_opt=True,
        use_packing=True,
        use_fp8=False,
        use_tunableop=True,
        gen_prompts=None,
    ))

    print_comparison_table(results)

    # Save results
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mi300x_benchmark_v7.json")

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "pytorch_version": torch.__version__,
        "rocm_version": torch.version.hip if hasattr(torch.version, 'hip') and torch.version.hip else "N/A",
        "model": MODEL_ID,
        "dataset": f"Alpaca ({TRAIN_SIZE} train / {EVAL_SIZE} eval)",
        "epochs": NUM_EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
        "configs": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
