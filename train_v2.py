"""
Upgraded training script for rocm-qlora (V2).
Includes Triton kernels, Paged AdamW, and torch.compile integration.
"""
import os
import json
import time
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset

# rocm-qlora V1 imports
from rocm_qlora.model.quantize_model import quantize_model, enable_all_kernels
from rocm_qlora.utils.rocm_utils import check_rocm, get_memory_stats

# rocm-qlora V2 imports
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.utils.compile_utils import compile_lora_only, warmup_compiled_model

def format_sample(sample):
    """Formats Alpaca dataset samples into instruction-following prompts."""
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

def parse_args():
    parser = argparse.ArgumentParser(description="rocm-qlora V2 Training")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    # V2 Phase 1 & 2 Specific Args
    parser.add_argument("--use_paged_optimizer", type=bool, default=True)
    parser.add_argument("--use_triton_kernels", type=bool, default=True)
    parser.add_argument("--use_torch_compile", type=bool, default=False)
    parser.add_argument("--compile_warmup_steps", type=int, default=3)
    
    # V2 Phase 3 Specific Args
    parser.add_argument("--use_packing", type=bool, default=True)
    
    # V2 Phase 4 Specific Args
    parser.add_argument("--use_flash_attention", type=bool, default=True)
    
    # V4 Phase 1 & 2 Specific Args
    parser.add_argument("--use_fp8", default=False, type=bool,
        help="Enable FP8 training. Requires ROCm 6.2+ and MI300X. "
             "Falls back to BF16 silently on other hardware.")
    parser.add_argument("--use_double_quant", default=False, type=bool,
        help="Enable double quantization on NF4 layers. "
             "Saves ~0.37 bits/param. Requires --bits=4.")
    
    # V4 Phase 3 Specific Args
    parser.add_argument("--use_tunableop", default=False, type=bool,
        help="Enable TunableOp GEMM autotuning. "
             "First run tunes kernels (~2-5s per unique GEMM shape). "
             "Subsequent runs load cached optimal kernels. "
             "Do not combine with --use_torch_compile.")
    parser.add_argument("--tunableop_cache_dir", default="./tunableop_cache", type=str,
        help="Directory for TunableOp GEMM kernel cache CSV.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # STEP 0: TunableOp GEMM autotuning (must be set before any CUDA ops)
    if args.use_tunableop:
        from rocm_qlora.profiling import enable_tunableop, get_tunableop_status
        if args.use_torch_compile:
            print("[tunableop] WARNING: --use_torch_compile and --use_tunableop "
                  "may conflict. Recommend disabling --use_torch_compile.")
        result = enable_tunableop(args.tunableop_cache_dir)
        print(f"[tunableop] ENABLED | Cache: {result['cache_path']}")
        if result['compile_warning']:
            print(f"[tunableop] {result['compile_warning']}")
        status = get_tunableop_status()
        if status['cache_exists']:
            print(f"[tunableop] Loading {status['num_cached_shapes']} cached GEMM shapes")
        else:
            print(f"[tunableop] No cache found — will tune on first run (slower first epoch)")
    
    # 1. Setup ROCm environment
    info = check_rocm()
    print(f"\n[rocm-qlora V2] Device: {info['device_name']} | ROCm: {info['version']} | Available: {info['available']}")
    
    device = "cuda" if info["available"] else "cpu"
    
    # 2. Model Loading
    print(f"Loading tokenizer and model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # NOTE: Load to CPU first as per V1 contract
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    pre_q_mem = get_memory_stats()
    print(f"VRAM before quantization: {pre_q_mem['allocated_gb']:.2f}GB allocated")
    
    # 3. Quantization
    model = quantize_model(
        model, 
        bits=args.bits, 
        lora_r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # STEP 5b: Double quantization (NF4 layers only, CPU)
    if args.use_double_quant:
        if args.bits != 4:
            print("[dq] WARNING: --use_double_quant requires --bits=4. Skipping.")
        else:
            from rocm_qlora.quantization.double_quant import (
                patch_quant_linear_for_double_quant, estimate_double_quant_savings
            )
            n_dq = patch_quant_linear_for_double_quant(model)
            # Create a small representative tensor for estimate
            savings = estimate_double_quant_savings(
                torch.randn(64, 64)  # representative estimate
            )
            print(f"[dq] Double quant enabled: {n_dq} layers | "
                  f"~{savings['bits_saved_per_param']:.3f} bits/param saved")
    
    # 4. V2 Optimization: Triton Kernels
    if args.use_triton_kernels:
        enabled_count = enable_all_kernels(model)
        print(f"Enabled Triton kernels for {enabled_count} layers.")
    
    # 5. Flash Attention configuration (V2 Phase 4)
    # NOTE: Must be set before model moves to device
    fa_info = {"available": False, "backend": "none", "supports_backward": False}
    if args.use_flash_attention:
        from rocm_qlora.attention import (
            detect_flash_attention, patch_model_attention,
            install_instructions,
        )
        fa_info = detect_flash_attention()
        print(f"[attention] FA2 available: {fa_info['available']} | "
              f"Backend: {fa_info['backend']} | "
              f"Backward safe: {fa_info['supports_backward']}")

        model, patch_info = patch_model_attention(model, verbose=True)

        if patch_info['warning']:
            print(f"[attention] WARNING: {patch_info['warning']}")
        if patch_info['install_hint']:
            print(f"[attention] To install FA2: see install_instructions()")

        print(f"[attention] Using: {patch_info['backend_chosen']}")
    
    # Print status line for startup config
    if not args.use_flash_attention:
        print("Flash Attention:  DISABLED (use_flash_attention=False)")
    elif not fa_info['available']:
         print("Flash Attention:  torch SDPA (FA2 not installed — run install_instructions())")
    elif not fa_info['supports_backward']:
         print(f"Flash Attention:  torch SDPA (RDNA3 CK unsafe — set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE)")
    else:
         print(f"Flash Attention:  {fa_info['backend']} backend | backward: YES")

    # 6. Move to GPU and enable checkpointing
    model = model.to(device)
    if info["available"]:
        model.gradient_checkpointing_enable()
        
    # STEP 7b: FP8 wrapping (MI300X only — BF16 fallback on other hardware)
    if args.use_fp8:
        from rocm_qlora.fp8 import detect_fp8_support, wrap_model_for_fp8, FP8Config
        fp8_support = detect_fp8_support()
        fp8_config = FP8Config(
            use_transformer_engine=fp8_support['transformer_engine_available']
        )
        model, wrap_info = wrap_model_for_fp8(model, fp8_config)
        if wrap_info['enabled']:
            print(f"[fp8] ENABLED | Backend: {wrap_info['backend']} | "
                  f"Layers wrapped: {wrap_info['layers_wrapped']}")
        else:
            print(f"[fp8] SKIPPED — {wrap_info['reason']}")
            print(f"[fp8] Continuing with BF16 (no performance change)")
    
    # 6. V2 Optimization: torch.compile
    if args.use_torch_compile and info["available"]:
        model = compile_lora_only(model)
        # Warmup
        warmup_compiled_model(model, device=device, seq_len=args.max_seq_length)
        
    # STEP 10b: TunableOp warmup (populate GEMM table)
    if args.use_tunableop and not get_tunableop_status()['cache_exists']:
        from rocm_qlora.profiling import tunableop_warmup, switch_to_load_only
        print("[tunableop] Running GEMM tuning warmup...")
        warmup_result = tunableop_warmup(
            model, device=device,
            seq_len=args.max_seq_length,
            num_steps=5,
        )
        speedup = warmup_result['speedup_after_warmup']
        print(f"[tunableop] Warmup complete | "
              f"First step: {warmup_result['first_step_ms']:.0f}ms | "
              f"Last step: {warmup_result['last_step_ms']:.0f}ms | "
              f"Speedup after warmup: {speedup:.2f}x")
        # Switch to load-only after warmup — no more tuning overhead
        switch_to_load_only()
        print("[tunableop] Switched to load-only mode for training run")
    
    post_q_mem = get_memory_stats()
    print(f"VRAM after quantization: {post_q_mem['allocated_gb']:.2f}GB allocated")
    
    # 7. Dataset Preparation (V2 Phase 3)
    raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
    formatted_texts = [format_sample(s) for s in raw_dataset]

    if args.use_packing:
        from rocm_qlora.data import (
            build_packed_dataset, PackedSequenceCollator, compute_packing_efficiency
        )
        packed_samples, efficiency = build_packed_dataset(
            raw_texts=formatted_texts,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
        print(f"[data] Packing efficiency: {efficiency['efficiency_pct']:.1f}%")
        print(f"[data] {efficiency['original_samples']} samples -> "
              f"{efficiency['packed_samples']} packs "
              f"({efficiency['compression_ratio']:.2f}x compression)")

        collator = PackedSequenceCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_block_attention=True,
        )
        dataset_obj = packed_samples  # already list of dicts
        dataloader = DataLoader(
            dataset_obj, batch_size=args.batch_size,
            collate_fn=collator, shuffle=True,
        )
        print(f"Sequence packing: ENABLED ({efficiency['efficiency_pct']:.1f}% efficiency, "
              f"{efficiency['original_samples']}->{efficiency['packed_samples']} packs)")
    else:
        # V1 fallback: pad to max_seq_length
        print(f"Sequence packing: DISABLED (padding to {args.max_seq_length})")
        
    # V4 Status Updates
    if args.use_double_quant and args.bits == 4:
        print("Double Quant:      ENABLED (8.3% memory saving on NF4)")
    else:
        print("Double Quant:      DISABLED")
        
    if args.use_fp8:
        from rocm_qlora.fp8 import detect_fp8_support
        s = detect_fp8_support()
        if s['fp8_supported']:
            backend_name = "TransformerEngine" if s['transformer_engine_available'] else "PyTorch native"
            print(f"FP8 Training:      ENABLED ({backend_name}, {s['arch'].upper()})")
        else:
            print(f"FP8 Training:      DISABLED — {s['reason']}")
    else:
        print("FP8 Training:      DISABLED")
        
    if args.use_tunableop:
        from rocm_qlora.profiling import get_tunableop_status
        status = get_tunableop_status()
        print(f"TunableOp:         ENABLED (cache: {status['cache_path']})")
    else:
        print("TunableOp:         DISABLED")

    if not args.use_packing:
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        
        for sample in raw_dataset:
            text = format_sample(sample)
            tokens = tokenizer(
                text, 
                max_length=args.max_seq_length, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            ids = tokens["input_ids"].squeeze(0)
            mask = tokens["attention_mask"].squeeze(0)
            
            labels = ids.clone()
            labels[ids == tokenizer.pad_token_id] = -100
            
            input_ids_list.append(ids)
            attention_masks_list.append(mask)
            labels_list.append(labels)
            
        train_dataset = TensorDataset(
            torch.stack(input_ids_list),
            torch.stack(attention_masks_list),
            torch.stack(labels_list)
        )
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 8. Optimizer and Scheduler
    total_steps = (len(dataloader) // args.grad_accum_steps) * args.epochs
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.use_paged_optimizer:
        print("Using PagedAdamW optimizer (offloading states to CPU pinned memory)...")
        optimizer = PagedAdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=total_steps
    )
    
    # 9. Training Loop
    print(f"\nStarting training for {args.epochs} epochs ({total_steps} steps)...")
    model.train()
    global_step = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                **batch
            )
            
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch: {epoch} | Step: {global_step}/{total_steps} | "
                        f"Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {current_lr:.2e}"
                    )
                
                if global_step % 50 == 0:
                    stats = get_memory_stats()
                    paged_info = ""
                    if hasattr(optimizer, "get_memory_stats"):
                        p_stats = optimizer.get_memory_stats()
                        paged_info = f" | Paged: {p_stats['cpu_pinned_mb']:.1f}MB (CPU)"
                    print(f"  VRAM: Allocated={stats['allocated_gb']:.2f}GB{paged_info}")
    
    total_time = time.time() - start_time
    
    # 10. Saving Artifacts
    print("\nTraining complete. Saving artifacts...")
    os.makedirs("./outputs", exist_ok=True)
    
    lora_state_dict = {
        name: param.data for name, param in model.named_parameters() 
        if param.requires_grad
    }
    torch.save(lora_state_dict, "./outputs/lora_weights_v2.pt")
    
    config = vars(args)
    with open("./outputs/training_config_v2.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Final Loss: {loss.item() * args.grad_accum_steps:.4f}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Saved LoRA weights to ./outputs/lora_weights_v2.pt")
