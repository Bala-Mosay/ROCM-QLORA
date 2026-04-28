"""
Supervised Fine-Tuning trainer for rocm-qlora.

Wraps the V2 training pipeline into a clean class API.
Supports both single-GPU and multi-GPU FSDP training.
Integrates all V2 optimizations: Triton kernels, PagedAdamW,
sequence packing, and Flash Attention.
"""
import os
import time
import math
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# rocm-qlora imports
from rocm_qlora.model.quantize_model import enable_all_kernels
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.data import PackedSequenceCollator, build_packed_dataset
from rocm_qlora.utils.rocm_utils import get_memory_stats
from rocm_qlora.utils.compile_utils import compile_lora_only, is_compile_safe
from rocm_qlora.distributed import (
    get_trainable_fsdp_params, save_fsdp_lora_weights,
    is_main_process, print_on_main, barrier
)

@dataclass
class SFTConfig:
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_seq_length: int = 512
    warmup_steps: int = 10
    max_grad_norm: float = 0.3
    use_packing: bool = True
    use_paged_optimizer: bool = True
    use_triton_kernels: bool = True
    use_torch_compile: bool = False
    log_every: int = 10
    save_every: int = 0        # 0 = only save at end
    eval_every: int = 0        # 0 = no mid-training eval
    page_size_mb: int = 256
    rank: int = 0              # distributed rank, 0 for single GPU
    world_size: int = 1        # 1 for single GPU

class ROCmSFTTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer, 
        train_dataset: List[Dict[str, Any]], 
        config: SFTConfig, 
        eval_dataset: Optional[List[Dict[str, Any]]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.eval_dataset = eval_dataset
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Detect FSDP
        self.is_fsdp = self._check_fsdp(model)
        
        # 1. Apply optimizations
        if config.use_triton_kernels:
            # NOTE: enable_all_kernels after FSDP wrap — confirmed in Phase 1
            n = enable_all_kernels(model)
            if is_main_process(self.rank):
                print(f"[sft] Enabled Triton kernels for {n} layers")
                
        if config.use_torch_compile and is_compile_safe():
            self.model = compile_lora_only(model)
            
        # 2. Build Dataloader
        if config.use_packing:
            # We assume train_dataset is already packed or we pack it here if it's raw
            # For simplicity in this API, we expect list of dicts. 
            # If not already containing input_ids, we'd need to pack.
            # Here we assume the user provides a list that either needs collating or is already packed.
            pass
            
        self.collator = PackedSequenceCollator(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Sampler
        if self.world_size > 1:
            self.sampler = torch.utils.data.DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=True
            )
        else:
            self.sampler = torch.utils.data.RandomSampler(train_dataset)
            
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=self.sampler,
            collate_fn=self.collator
        )
        
        # 3. Build Optimizer
        # Extract params via get_trainable_fsdp_params(model) if FSDP
        if self.is_fsdp:
            trainable_params = get_trainable_fsdp_params(model)
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            
        if is_main_process(self.rank):
            print(f"[sft] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
            
        if config.use_paged_optimizer:
            self.optimizer = PagedAdamW(
                trainable_params, 
                lr=config.lr, 
                weight_decay=config.weight_decay,
                page_size_mb=config.page_size_mb
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
            
        # 4. Scheduler
        total_steps = (len(self.train_dataloader) // config.grad_accum) * config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
    def _check_fsdp(self, model) -> bool:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return isinstance(model, FSDP)
        except ImportError:
            return False

    def train(self) -> Dict[str, Any]:
        print_on_main(f"\n[sft] Starting SFT training for {self.config.num_epochs} epochs...", self.rank)
        
        self.model.train()
        device = next(self.model.parameters()).device
        global_step = 0
        total_loss = 0
        start_time = time.time()
        peak_vram = 0
        
        for epoch in range(self.config.num_epochs):
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(epoch)
                
            for step, batch in enumerate(self.train_dataloader):
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.grad_accum
                loss.backward()
                
                total_loss += loss.item() * self.config.grad_accum
                
                if (step + 1) % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.config.log_every == 0:
                        avg_loss = total_loss / (step + 1)
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        print_on_main(
                            f"Epoch {epoch} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s",
                            self.rank
                        )
                        
                    if global_step % 50 == 0:
                        stats = get_memory_stats()
                        peak_vram = max(peak_vram, stats['allocated_gb'])
                        if is_main_process(self.rank):
                            print(f"  VRAM: {stats['allocated_gb']:.2f}GB alloc")

                    if self.config.save_every > 0 and global_step % self.config.save_every == 0:
                        checkpoint_path = os.path.join(self.config.output_dir, f"lora_step_{global_step}.pt")
                        self.save_lora(checkpoint_path)
        
        wall_time = time.time() - start_time
        final_loss = total_loss / (len(self.train_dataloader) * self.config.num_epochs)
        
        return {
            "final_loss": final_loss,
            "total_steps": global_step,
            "wall_time_seconds": wall_time,
            "peak_vram_gb": peak_vram
        }

    def save_lora(self, path: str):
        """Saves LoRA weights, handling FSDP gathering if necessary."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        barrier(self.rank)
        
        if self.is_fsdp:
            save_fsdp_lora_weights(self.model, path, self.rank)
        else:
            if is_main_process(self.rank):
                lora_state = {
                    name: p.data.clone() 
                    for name, p in self.model.named_parameters() 
                    if p.requires_grad
                }
                torch.save(lora_state, path)
        
        if is_main_process(self.rank):
            config_path = path.replace(".pt", "_config.json")
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2)

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}
            
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=self.collator)
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }
