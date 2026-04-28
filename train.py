"""
Training script for rocm-qlora.
Fine-tunes TinyLlama on ROCm hardware using blockwise quantization and LoRA.
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset

from rocm_qlora.model.quantize_model import quantize_model
from rocm_qlora.utils.rocm_utils import check_rocm, get_memory_stats

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

def main():
    # 1. Setup ROCm environment
    info = check_rocm()
    print(f"\n[rocm-qlora] Device: {info['device_name']} | ROCm: {info['version']} | Available: {info['available']}")
    if not info["available"]:
        print("[WARNING] No GPU detected — training will run on CPU and be very slow")
    
    device = "cuda" if info["available"] else "cpu"
    
    # 2. Model Loading Configuration
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading tokenizer and model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # TinyLlama has no pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    
    # NOTE: Load to CPU first to avoid device_map dispatch conflicts with gradient checkpointing on ROCm.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    pre_q_mem = get_memory_stats()
    print(f"VRAM before quantization: {pre_q_mem['allocated_gb']:.2f}GB allocated")
    
    # 3. Quantization
    # We target all major linear layers for maximum memory reduction.
    model = quantize_model(
        model, 
        bits=8, 
        lora_r=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # NOTE: Move to GPU after quantization, then enable checkpointing.
    model = model.to(device)
    if info["available"]:
        model.gradient_checkpointing_enable()
    
    post_q_mem = get_memory_stats()
    print(f"VRAM after quantization: {post_q_mem['allocated_gb']:.2f}GB allocated")
    
    # 4. Dataset Preparation
    print("Loading and processing dataset (Alpaca subset)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")
    
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    
    for sample in dataset:
        text = format_sample(sample)
        tokens = tokenizer(
            text, 
            max_length=512, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        ids = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)
        
        # Labels: copy input_ids, but set pad token positions to -100 to ignore in loss
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
    
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 5. Training Setup
    num_epochs = 3
    grad_accum_steps = 4
    total_steps = (len(dataloader) // grad_accum_steps) * num_epochs
    
    # Collect only LoRA parameters for optimization
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=total_steps
    )
    
    # 6. Training Loop
    print(f"\nStarting training for {num_epochs} epochs ({total_steps} steps)...")
    model.train()
    global_step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for step, (b_input_ids, b_mask, b_labels) in enumerate(dataloader):
            b_input_ids = b_input_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
            
            outputs = model(
                input_ids=b_input_ids, 
                attention_mask=b_mask, 
                labels=b_labels
            )
            
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch: {epoch} | Step: {global_step}/{total_steps} | "
                        f"Loss: {loss.item() * grad_accum_steps:.4f} | LR: {current_lr:.2e}"
                    )
                
                if global_step % 50 == 0:
                    stats = get_memory_stats()
                    print(f"  VRAM: Allocated={stats['allocated_gb']:.2f}GB | Reserved={stats['reserved_gb']:.2f}GB")
    
    total_time = time.time() - start_time
    
    # 7. Saving Artifacts
    print("\nTraining complete. Saving artifacts...")
    os.makedirs("./outputs", exist_ok=True)
    
    # NOTE: Save ONLY LoRA weights using full dotted paths from root model.
    lora_state_dict = {
        name: param.data for name, param in model.named_parameters() 
        if param.requires_grad
    }
    torch.save(lora_state_dict, "./outputs/lora_weights.pt")
    
    config = {
        "model_id": model_id,
        "bits": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "block_size": 64,
        "lr": 2e-4,
        "epochs": num_epochs,
        "batch_size": 2,
        "grad_accum_steps": grad_accum_steps,
        "max_length": 512
    }
    
    with open("./outputs/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Final Loss: {loss.item() * grad_accum_steps:.4f}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Saved LoRA weights to ./outputs/lora_weights.pt")
    print(f"Saved configuration to ./outputs/training_config.json")

if __name__ == "__main__":
    main()
