# Launch command:
# torchrun --nproc_per_node=NUM_GPUS train_v3_fsdp.py \
#     --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#     --bits 8 --lora_r 8 --lora_alpha 16 \
#     --target_modules q_proj,v_proj,k_proj,o_proj \
#     --use_packing True --use_triton_kernels True \
#     --use_flash_attention True \
#     --num_epochs 3 --batch_size 2 --grad_accum 4

"""
Multi-GPU training script for rocm-qlora (V3).
Uses FSDP for model sharding and RCCL for communication.
"""
import os
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset

# rocm-qlora imports
from rocm_qlora.model.quantize_model import quantize_model, enable_all_kernels
from rocm_qlora.utils.rocm_utils import get_memory_stats, check_rocm
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.data import (
    build_packed_dataset, PackedSequenceCollator
)
from rocm_qlora.attention import (
    detect_flash_attention, patch_model_attention
)
from rocm_qlora.distributed import (
    init_distributed, setup_device, cleanup_distributed,
    print_on_main, barrier, prepare_model_for_fsdp,
    get_transformer_layer_class, get_trainable_fsdp_params,
    save_fsdp_lora_weights
)

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
    parser = argparse.ArgumentParser(description="rocm-qlora V3 FSDP Training")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    # V2 Flags
    parser.add_argument("--use_triton_kernels", type=bool, default=True)
    parser.add_argument("--use_packing", type=bool, default=True)
    parser.add_argument("--use_flash_attention", type=bool, default=True)
    
    # V3 FSDP Specific Args
    parser.add_argument("--fsdp_mixed_precision", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--save_path", type=str, default="./outputs/lora_fsdp.pt")
    parser.add_argument("--page_size_mb", type=int, default=64)

    return parser.parse_args()

def main():
    # STEP 1: Init distributed
    rank, local_rank, world_size = init_distributed()
    device = setup_device(local_rank)
    print_on_main(f"[fsdp] Training on {world_size} GPUs (RCCL backend)", rank)

    # STEP 2: Parse args
    args = parse_args()
    
    # STEP 3: Load model to CPU on ALL ranks
    print_on_main(f"[fsdp] Loading tokenizer and model: {args.model_id} to CPU...", rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # STEP 4: Quantization on CPU (Identical across ranks)
    print_on_main(f"[fsdp] Quantizing model ({args.bits}-bit) on CPU...", rank)
    target_modules = args.target_modules.split(",")
    model = quantize_model(
        model, 
        bits=args.bits, 
        lora_r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        target_modules=target_modules,
        block_size=args.block_size
    )
    
    # STEP 5: Flash Attention config BEFORE FSDP wrap
    if args.use_flash_attention:
        model, patch_info = patch_model_attention(model, verbose=(rank == 0))
        print_on_main(f"[fsdp] Attention: {patch_info['backend_chosen']}", rank)
        
    # STEP 6: FSDP wrap — moves model to device internally
    print_on_main("[fsdp] Applying FSDP wrap...", rank)
    dtype = torch.bfloat16 if args.fsdp_mixed_precision == "bf16" else torch.float16
    transformer_layer_class = get_transformer_layer_class(model)
    model = prepare_model_for_fsdp(model, transformer_layer_class, dtype)
    
    # STEP 7: Enable Triton kernels AFTER FSDP wrap
    if args.use_triton_kernels:
        n_kernels = enable_all_kernels(model)
        print_on_main(f"[fsdp] Triton kernels enabled for {n_kernels} layers", rank)
        
    # STEP 8: Gradient Checkpointing
    model.gradient_checkpointing_enable()
    
    # STEP 9: Dataset with packing + DistributedSampler
    packed_samples = None
    efficiency = None
    
    if rank == 0:
        print("[fsdp] Preparing packed dataset on Rank 0...")
        raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
        formatted = [format_sample(s) for s in raw_dataset]
        packed_samples, efficiency = build_packed_dataset(
            formatted, tokenizer, args.max_seq_length
        )
        print(f"[fsdp] Packing efficiency: {efficiency['efficiency_pct']:.1f}% | "
              f"{efficiency['original_samples']} -> {efficiency['packed_samples']} packs "
              f"({efficiency['compression_ratio']:.2f}x compression)")
    
    # Wait for Rank 0 to finish data prep
    barrier(rank)
    
    # For single-node multi-GPU, we can just share the object if it was created in the same script flow,
    # but with torchrun/multiprocessing, we typically need to broadcast or re-create.
    # To keep it simple and reliable for this phase, we re-create on other ranks (identical seed/tokenizer).
    if rank != 0:
        raw_dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
        formatted = [format_sample(s) for s in raw_dataset]
        packed_samples, _ = build_packed_dataset(
            formatted, tokenizer, args.max_seq_length
        )

    collator = PackedSequenceCollator(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_block_attention=True,
    )
    sampler = torch.utils.data.DistributedSampler(
        packed_samples, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        packed_samples, batch_size=args.batch_size,
        sampler=sampler, collate_fn=collator,
    )

    # STEP 10: Optimizer — PagedAdamW on LoRA params only
    trainable = get_trainable_fsdp_params(model)
    print_on_main(f"[fsdp] Trainable parameters: {sum(p.numel() for p in trainable):,}", rank)
    optimizer = PagedAdamW(trainable, lr=args.lr, weight_decay=0.01)
    
    # STEP 11: Scheduler
    total_steps = (len(dataloader) // args.grad_accum) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps
    )
    
    # STEP 12: Training Loop
    print_on_main(f"\n[fsdp] Starting training for {args.num_epochs} epochs...", rank)
    global_step = 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch) # Critical for shuffle correctness
        model.train()
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    print_on_main(
                        f"Epoch: {epoch} | Step: {global_step}/{total_steps} | "
                        f"Loss: {loss.item() * args.grad_accum:.4f} | Rank: {rank}", rank
                    )
                
                if global_step % 50 == 0 and rank == 0:
                    stats = get_memory_stats()
                    print(f"  VRAM (Rank 0): {stats['allocated_gb']:.2f}GB allocated | {stats['reserved_gb']:.2f}GB reserved")

    total_time = time.time() - start_time
    print_on_main(f"\n[fsdp] Training complete in {total_time/60:.2f} minutes.", rank)

    # STEP 13: Save LoRA weights — rank 0 only, gather from all ranks first
    barrier(rank)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print_on_main(f"[fsdp] Saving consolidated LoRA weights to {args.save_path}...", rank)
    save_fsdp_lora_weights(model, args.save_path, rank)
    
    # STEP 14: Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
