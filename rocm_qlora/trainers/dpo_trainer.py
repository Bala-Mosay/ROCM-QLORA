"""
Direct Preference Optimization trainer for rocm-qlora.

DPO trains a policy model to prefer chosen responses over rejected ones
using a closed-form loss that bypasses explicit reward modeling.

ROCm memory optimization:
  Reference model is kept QUANTIZED (INT8/NF4), not FP16.
  Both models being quantized keeps the VRAM profile low.
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# rocm-qlora imports
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.utils.rocm_utils import get_memory_stats
from rocm_qlora.distributed import (
    get_trainable_fsdp_params, save_fsdp_lora_weights,
    is_main_process, print_on_main, barrier
)

@dataclass
class DPOConfig:
    beta: float = 0.1
    output_dir: str = "./outputs"
    num_epochs: int = 1
    batch_size: int = 1        # DPO needs 2x forward passes per sample
    grad_accum: int = 8
    lr: float = 5e-5           # lower than SFT
    max_seq_length: int = 512
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    use_paged_optimizer: bool = True
    log_every: int = 10
    rank: int = 0
    world_size: int = 1

class ROCmDPOTrainer:
    def __init__(
        self, 
        policy_model: nn.Module, 
        tokenizer, 
        train_dataset: List[Dict[str, str]], 
        config: DPOConfig, 
        ref_model: Optional[nn.Module] = None
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Detect FSDP
        self.is_fsdp = self._check_fsdp(policy_model)
        
        # 1. Reference Model Setup
        if ref_model is None:
            # For ROCm-optimized DPO, we expect the user to provide a frozen quantized model.
            # If not provided, we warn. Auto-cloning a quantized model is complex via deepcopy.
            # We'll use the policy_model itself but skip LoRA for ref forward if possible,
            # or expect the user to pass a separate instance.
            print_on_main("[dpo] WARNING: No ref_model provided. DPO requires a reference model.", self.rank)
            self.ref_model = policy_model # THIS IS TEMPORARY - In real use, pass a frozen clone.
        else:
            self.ref_model = ref_model
            
        # Freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        # 2. Build Dataloader
        # For DPO, we tokenize on-the-fly to handle prompt/chosen/rejected pairs.
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
            collate_fn=lambda x: x # pass raw dicts to training loop for manual tokenization
        )
        
        # 3. Build Optimizer
        if self.is_fsdp:
            trainable_params = get_trainable_fsdp_params(policy_model)
        else:
            trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
            
        if config.use_paged_optimizer:
            self.optimizer = PagedAdamW(trainable_params, lr=config.lr)
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
            
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

    def _tokenize_preference_pair(self, prompt: str, chosen: str, rejected: str) -> Dict[str, torch.Tensor]:
        """Tokenizes prompt + chosen and prompt + rejected. Masks prompt in labels."""
        def tokenize_one(response):
            full_text = prompt + response
            tokens = self.tokenizer(
                full_text, 
                max_length=self.config.max_seq_length, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            prompt_tokens = self.tokenizer(
                prompt, 
                max_length=self.config.max_seq_length, 
                truncation=True, 
                return_tensors="pt"
            )
            prompt_len = prompt_tokens.input_ids.shape[1]
            
            labels = tokens.input_ids.clone()
            labels[:, :prompt_len] = -100 # Mask prompt tokens
            return tokens.input_ids, tokens.attention_mask, labels

        c_ids, c_mask, c_labels = tokenize_one(chosen)
        r_ids, r_mask, r_labels = tokenize_one(rejected)
        
        return {
            "chosen_ids": c_ids, "chosen_mask": c_mask, "chosen_labels": c_labels,
            "rejected_ids": r_ids, "rejected_mask": r_mask, "rejected_labels": r_labels
        }

    def compute_log_probs(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes total response log prob."""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs for the actual label tokens
        # labels: [batch, seq], log_probs: [batch, seq, vocab]
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        
        # Mask out prompt and padding
        mask = (labels != -100)
        return (per_token_logps * mask).sum(-1)

    def compute_dpo_loss(
        self, 
        policy_chosen_lp: torch.Tensor, 
        policy_rejected_lp: torch.Tensor, 
        ref_chosen_lp: torch.Tensor, 
        ref_rejected_lp: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard DPO loss calculation."""
        chosen_rewards = self.config.beta * (policy_chosen_lp - ref_chosen_lp)
        rejected_rewards = self.config.beta * (policy_rejected_lp - ref_rejected_lp)
        
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        margin = (chosen_rewards - rejected_rewards).mean().item()
        
        return loss, {
            "loss": loss.item(),
            "accuracy": accuracy,
            "reward_margin": margin,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item()
        }

    def train(self) -> Dict[str, Any]:
        print_on_main(f"\n[dpo] Starting DPO training...", self.rank)
        self.policy_model.train()
        device = next(self.policy_model.parameters()).device
        global_step = 0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(epoch)
                
            for step, batch_raw in enumerate(self.train_dataloader):
                # batch_raw is list of dicts: [{prompt, chosen, rejected}]
                # We process one at a time for simplicity in this trainer
                for sample in batch_raw:
                    data = self._tokenize_preference_pair(sample['prompt'], sample['chosen'], sample['rejected'])
                    data = {k: v.to(device) for k, v in data.items()}
                    
                    # Policy forward
                    policy_chosen_lp = self.compute_log_probs(
                        self.policy_model, data['chosen_ids'], data['chosen_mask'], data['chosen_labels']
                    )
                    policy_rejected_lp = self.compute_log_probs(
                        self.policy_model, data['rejected_ids'], data['rejected_mask'], data['rejected_labels']
                    )
                    
                    # Reference forward
                    with torch.no_grad():
                        ref_chosen_lp = self.compute_log_probs(
                            self.ref_model, data['chosen_ids'], data['chosen_mask'], data['chosen_labels']
                        )
                        ref_rejected_lp = self.compute_log_probs(
                            self.ref_model, data['rejected_ids'], data['rejected_mask'], data['rejected_labels']
                        )
                    
                    loss, metrics = self.compute_dpo_loss(
                        policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp
                    )
                    
                    (loss / self.config.grad_accum).backward()
                
                if (step + 1) % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.config.log_every == 0:
                        print_on_main(
                            f"Step {global_step} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.2f} | Margin: {metrics['reward_margin']:.3f}",
                            self.rank
                        )

        return {
            "final_loss": metrics['loss'],
            "final_accuracy": metrics['accuracy'],
            "total_steps": global_step,
            "wall_time_seconds": time.time() - start_time
        }

    def save_lora(self, path: str):
        """Saves policy LoRA weights."""
        barrier(self.rank)
        if self.is_fsdp:
            save_fsdp_lora_weights(self.policy_model, path, self.rank)
        else:
            if is_main_process(self.rank):
                lora_state = {name: p.data.clone() for name, p in self.policy_model.named_parameters() if p.requires_grad}
                torch.save(lora_state, path)
