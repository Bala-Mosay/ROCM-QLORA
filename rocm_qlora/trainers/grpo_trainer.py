"""
Group Relative Policy Optimization trainer for rocm-qlora.

GRPO is an RL algorithm that removes the need for a separate value model 
by using group-relative normalization of rewards.
Optimized for LoRA-based reasoning models on ROCm.
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from torch.utils.data import DataLoader

# rocm-qlora imports
from rocm_qlora.optim import PagedAdamW
from rocm_qlora.utils.rocm_utils import get_memory_stats
from rocm_qlora.distributed import (
    get_trainable_fsdp_params, save_fsdp_lora_weights,
    is_main_process, print_on_main, barrier
)

@dataclass
class GRPOConfig:
    G: int = 8                    # completions per prompt
    beta: float = 0.04            # KL penalty weight
    lr: float = 1e-5              # RL is sensitive, use lower LR
    output_dir: str = "./outputs"
    num_steps: int = 500          # Online RL uses steps
    grad_accum: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    max_prompt_length: int = 256
    use_paged_optimizer: bool = True
    log_every: int = 10
    rank: int = 0
    world_size: int = 1

class ROCmGRPOTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        ref_model: nn.Module, 
        tokenizer, 
        reward_fn: Callable[[str, str], float], 
        config: GRPOConfig
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        
        # Detect FSDP
        self.is_fsdp = self._check_fsdp(model)
        
        # Freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        
        # Optimizer
        if self.is_fsdp:
            trainable_params = get_trainable_fsdp_params(model)
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            
        if config.use_paged_optimizer:
            self.optimizer = PagedAdamW(trainable_params, lr=config.lr)
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)

    def _check_fsdp(self, model) -> bool:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return isinstance(model, FSDP)
        except ImportError:
            return False

    def generate_completions(self, prompt: str, G: int) -> List[str]:
        """Generates G completions for a single prompt."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.config.max_prompt_length, 
            truncation=True
        ).to(device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate in batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                num_return_sequences=G,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Slice to get only completions
        completions = []
        for i in range(G):
            completion_tokens = outputs[i, prompt_len:]
            text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(text)
            
        return completions

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Normalizes rewards within the group (G)."""
        device = next(self.model.parameters()).device
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if len(rewards_t) <= 1:
            return torch.zeros_like(rewards_t)
            
        std = rewards_t.std()
        if std < 1e-8:
            return rewards_t - rewards_t.mean()
            
        return (rewards_t - rewards_t.mean()) / (std + 1e-8)

    def compute_log_probs(self, model: nn.Module, prompt: str, completions: List[str]) -> torch.Tensor:
        """Computes sum of log probabilities for each completion."""
        log_probs_list = []
        device = next(model.parameters()).device
        
        for completion in completions:
            full_text = prompt + completion
            tokens = self.tokenizer(full_text, return_tensors="pt").to(device)
            # tokenizer(prompt) returns dict, need length
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            
            outputs = model(**tokens)
            # [batch, seq, vocab]
            logits = outputs.logits[:, :-1, :]
            input_ids = tokens.input_ids[:, 1:]
            
            lp = F.log_softmax(logits, dim=-1)
            # Gather log probs for the actual tokens
            token_lp = torch.gather(lp, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)
            
            # Sum only over completion tokens
            completion_lp = token_lp[:, prompt_len-1:].sum()
            log_probs_list.append(completion_lp)
            
        return torch.stack(log_probs_list)

    def train(self, prompts: List[Dict[str, str]]) -> Dict[str, Any]:
        print_on_main(f"\n[grpo] Starting GRPO reasoning training...", self.rank)
        self.model.train()
        device = next(self.model.parameters()).device
        start_time = time.time()
        rewards = []
        step = 0
        
        for step in range(self.config.num_steps):
            # Cycle through prompts
            prompt_data = prompts[step % len(prompts)]
            prompt = prompt_data["prompt"]
            
            # 1. Generate completions
            completions = self.generate_completions(prompt, self.config.G)
            
            # 2. Score completions
            rewards = [self.reward_fn(prompt, c) for c in completions]
            
            # 3. Advantages
            advantages = self.compute_advantages(rewards)
            
            # 4. Policy Gradient + KL
            # Policy log probs (trainable)
            policy_lps = self.compute_log_probs(self.model, prompt, completions)
            
            # Reference log probs (frozen)
            with torch.no_grad():
                ref_lps = self.compute_log_probs(self.ref_model, prompt, completions)
            
            # Loss = -mean(adv * log_pi) + beta * KL
            # First-order KL approx: policy_lp - ref_lp
            pg_loss = -(advantages.detach() * policy_lps).mean()
            kl_penalty = (policy_lps - ref_lps).mean()
            
            loss = pg_loss + self.config.beta * kl_penalty
            
            # 5. Step
            loss.backward()
            if (step + 1) % self.config.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            if step % self.config.log_every == 0:
                mean_r = sum(rewards) / len(rewards)
                print_on_main(
                    f"Step {step} | Loss: {loss.item():.4f} | Mean Reward: {mean_r:.2f} | KL: {kl_penalty.item():.4f}",
                    self.rank
                )
        
        return {
            "final_mean_reward": sum(rewards)/len(rewards),
            "total_steps": step + 1,
            "wall_time_seconds": time.time() - start_time
        }

    @staticmethod
    def built_in_reward_fns() -> Dict[str, Callable[[str, str], float]]:
        """Returns standard reward functions for reasoning models."""
        
        def format_reward(prompt: str, completion: str) -> float:
            """Checks for <think>...</think><answer>...</answer> structure."""
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            if re.search(pattern, completion, re.DOTALL):
                return 1.0
            return 0.0

        def length_reward(prompt: str, completion: str) -> float:
            """Rewards responses between 50 and 500 tokens."""
            # Heuristic: 1 token approx 4 chars
            length = len(completion) / 4
            if 50 <= length <= 500:
                return 1.0
            elif length < 50:
                return length / 50.0
            else:
                return max(0.0, 1.0 - (length - 500) / 500.0)

        def math_reward(prompt: str, completion: str) -> float:
            """Extracts answer from <answer> tags and compares with prompt ground truth."""
            # Expect prompt to have "Answer: X"
            truth_match = re.search(r"Answer:\s*(\d+\.?\d*)", prompt)
            if not truth_match:
                return 0.0
            truth = truth_match.group(1)
            
            # Expect completion to have <answer>X</answer>
            pred_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if not pred_match:
                return 0.0
            pred = pred_match.group(1).strip()
            
            return 1.0 if pred == truth else 0.0

        return {
            "format": format_reward,
            "length": length_reward,
            "math": math_reward
        }
