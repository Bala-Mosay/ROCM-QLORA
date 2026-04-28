import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import unittest.mock as mock
from dataclasses import asdict

from rocm_qlora.trainers import (
    ROCmSFTTrainer, SFTConfig,
    ROCmDPOTrainer, DPOConfig,
    ROCmGRPOTrainer, GRPOConfig
)
from rocm_qlora import quantize_model

# --- Mocks ---

class FakeOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits

class FakeTransformerForTrainer(nn.Module):
    """Mimics HF CausalLM interface for trainer tests."""
    def __init__(self):
        super().__init__()
        # Use 128x128 to pass the min_size check in quantize_model
        self.q_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)
        self.config = type('Config', (), {'model_type': 'llama'})()

    def forward(self, input_ids=None, attention_mask=None,
                labels=None, position_ids=None, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else 1
        seq = input_ids.shape[1] if input_ids is not None else 8
        logits = torch.randn(batch, seq, 100, requires_grad=True)
        loss = torch.tensor(2.0, requires_grad=True)
        return FakeOutput(loss=loss, logits=logits)

    def generate(self, **kwargs):
        return torch.randint(0, 100, (kwargs.get('num_return_sequences', 1), 32))

    def gradient_checkpointing_enable(self): pass
    def eval(self): return self
    def train(self, mode=True): return self

class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = torch.randint(2, 100, (1, 16))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    def decode(self, ids, **kwargs): return "fake completion text"

@pytest.fixture
def quantized_fake_model():
    model = FakeTransformerForTrainer()
    model = quantize_model(model, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])
    return model

@pytest.fixture
def fake_sft_dataset():
    return [
        {"input_ids": torch.randint(0, 100, (20,)).tolist(), "attention_mask": [1]*20, "labels": torch.randint(0, 100, (20,)).tolist()}
        for _ in range(10)
    ]

@pytest.fixture
def fake_dpo_dataset():
    return [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
        for _ in range(4)
    ]

@pytest.fixture
def fake_grpo_prompts():
    return [{"prompt": "What is 2+2? Answer: 4"} for _ in range(4)]

# --- SFT Tests ---

def test_sft_config_defaults():
    config = SFTConfig()
    assert config.num_epochs == 3
    assert config.use_paged_optimizer is True

def test_sft_trainer_init_no_error(quantized_fake_model, fake_sft_dataset):
    config = SFTConfig(num_epochs=1)
    with mock.patch('rocm_qlora.trainers.sft_trainer.enable_all_kernels', return_value=0):
        trainer = ROCmSFTTrainer(quantized_fake_model, FakeTokenizer(), fake_sft_dataset, config)
    assert trainer is not None

def test_sft_trainer_detects_fsdp_false(quantized_fake_model, fake_sft_dataset):
    config = SFTConfig()
    with mock.patch('rocm_qlora.trainers.sft_trainer.enable_all_kernels', return_value=0):
        trainer = ROCmSFTTrainer(quantized_fake_model, FakeTokenizer(), fake_sft_dataset, config)
    assert trainer.is_fsdp is False

def test_sft_save_creates_pt_file(quantized_fake_model, fake_sft_dataset, tmp_path):
    config = SFTConfig()
    with mock.patch('rocm_qlora.trainers.sft_trainer.enable_all_kernels', return_value=0):
        trainer = ROCmSFTTrainer(quantized_fake_model, FakeTokenizer(), fake_sft_dataset, config)
    path = str(tmp_path / "lora.pt")
    with mock.patch('rocm_qlora.trainers.sft_trainer.barrier'):
        trainer.save_lora(path)
    assert os.path.exists(path)
    assert os.path.exists(path.replace(".pt", "_config.json"))

def test_sft_evaluate_returns_dict(quantized_fake_model, fake_sft_dataset):
    config = SFTConfig()
    with mock.patch('rocm_qlora.trainers.sft_trainer.enable_all_kernels', return_value=0):
        trainer = ROCmSFTTrainer(quantized_fake_model, FakeTokenizer(), fake_sft_dataset, config)
    res = trainer.evaluate(fake_sft_dataset)
    assert "eval_loss" in res
    assert "perplexity" in res

def test_sft_perplexity_equals_exp_loss(quantized_fake_model, fake_sft_dataset):
    config = SFTConfig()
    with mock.patch('rocm_qlora.trainers.sft_trainer.enable_all_kernels', return_value=0):
        trainer = ROCmSFTTrainer(quantized_fake_model, FakeTokenizer(), fake_sft_dataset, config)
    res = trainer.evaluate(fake_sft_dataset)
    assert abs(res["perplexity"] - math.exp(res["eval_loss"])) < 1e-3

# --- DPO Tests ---

def test_dpo_config_defaults():
    config = DPOConfig()
    assert config.beta == 0.1
    assert config.lr == 5e-5

def test_dpo_trainer_init_no_error(quantized_fake_model, fake_dpo_dataset):
    config = DPOConfig()
    ref = FakeTransformerForTrainer()
    ref = quantize_model(ref, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])
    trainer = ROCmDPOTrainer(quantized_fake_model, FakeTokenizer(), fake_dpo_dataset, config, ref_model=ref)
    assert trainer is not None

def test_dpo_loss_formula_exact():
    beta = 0.1
    policy_c = torch.tensor([-1.0])
    policy_rejected = torch.tensor([-2.0])
    ref_c = torch.tensor([-1.0])
    ref_rejected = torch.tensor([-1.5])
    
    c_rew = beta * (policy_c - ref_c)
    r_rew = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(c_rew - r_rew).mean()
    
    expected = -math.log(1.0 / (1.0 + math.exp(-0.05)))
    assert abs(loss.item() - expected) < 1e-5

def test_dpo_ref_model_frozen(quantized_fake_model, fake_dpo_dataset):
    config = DPOConfig()
    ref = FakeTransformerForTrainer()
    trainer = ROCmDPOTrainer(quantized_fake_model, FakeTokenizer(), fake_dpo_dataset, config, ref_model=ref)
    for p in trainer.ref_model.parameters():
        assert p.requires_grad is False

# --- GRPO Tests ---

def test_grpo_config_defaults():
    config = GRPOConfig()
    assert config.G == 8
    assert config.beta == 0.04

def test_grpo_trainer_init_no_error(quantized_fake_model, fake_grpo_prompts):
    config = GRPOConfig()
    ref = FakeTransformerForTrainer()
    ref = quantize_model(ref, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])
    trainer = ROCmGRPOTrainer(quantized_fake_model, ref, FakeTokenizer(), lambda p, c: 1.0, config)
    assert trainer is not None

def test_grpo_advantages_normalized(quantized_fake_model):
    ref = FakeTransformerForTrainer()
    ref = quantize_model(ref, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])
    trainer = ROCmGRPOTrainer(quantized_fake_model, ref, FakeTokenizer(), lambda p, c: 1.0, GRPOConfig())
    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = trainer.compute_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-2

def test_grpo_advantages_single_value(quantized_fake_model):
    ref = FakeTransformerForTrainer()
    ref = quantize_model(ref, bits=8, lora_r=4, target_modules=["q_proj", "v_proj"])
    trainer = ROCmGRPOTrainer(quantized_fake_model, ref, FakeTokenizer(), lambda p, c: 1.0, GRPOConfig())
    rewards = [5.0]
    adv = trainer.compute_advantages(rewards)
    assert adv.shape == (1,)
    assert adv.item() == 0.0

def test_grpo_format_reward_detects_structure():
    fns = ROCmGRPOTrainer.built_in_reward_fns()
    fmt = fns["format"]
    assert fmt("", "<think>step 1</think><answer>42</answer>") == 1.0
    assert fmt("", "plain text") == 0.0

def test_grpo_math_reward_correct():
    fns = ROCmGRPOTrainer.built_in_reward_fns()
    math_fn = fns["math"]
    assert math_fn("Answer: 4", "<answer>4</answer>") == 1.0
    assert math_fn("Answer: 4", "<answer>5</answer>") == 0.0

def test_grpo_ref_model_frozen(quantized_fake_model):
    ref = FakeTransformerForTrainer()
    trainer = ROCmGRPOTrainer(quantized_fake_model, ref, FakeTokenizer(), lambda p, c: 1.0, GRPOConfig())
    for p in trainer.ref_model.parameters():
        assert p.requires_grad is False
