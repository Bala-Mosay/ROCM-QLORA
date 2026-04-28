import pytest
import torch
import torch.nn as nn
import unittest.mock as mock
import os
from rocm_qlora.distributed import (
    get_qlora_fsdp_policy,
    prepare_model_for_fsdp,
    get_trainable_fsdp_params,
    save_fsdp_lora_weights,
    get_transformer_layer_class,
    init_distributed,
    is_main_process,
    print_on_main,
    get_distributed_info
)
from rocm_qlora.model.quantize_model import quantize_model

def mock_distributed_initialized():
    return mock.patch('torch.distributed.is_initialized', return_value=True)

def mock_distributed_not_initialized():
    return mock.patch('torch.distributed.is_initialized', return_value=False)

def test_get_policy_returns_callable():
    policy = get_qlora_fsdp_policy(nn.Linear)
    assert callable(policy)

def test_prepare_raises_if_not_initialized():
    model = nn.Linear(4, 4)
    with mock_distributed_not_initialized():
        with pytest.raises(RuntimeError) as excinfo:
            prepare_model_for_fsdp(model, nn.Linear)
        assert "torchrun" in str(excinfo.value)

def test_get_trainable_fsdp_params_lora_only():
    # Use a real model structure but keep it small
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'model_type': 'llama'})()
            self.layers = nn.ModuleList([nn.Linear(128, 128)])
            
    model = FakeTransformer()
    model = quantize_model(model, bits=8, lora_r=4, target_modules=["layers"])
    
    # Even if not wrapped in FSDP, the utility should filter params
    trainable = get_trainable_fsdp_params(model)
    for p in trainable:
        assert p.requires_grad is True
        
    # Check that frozen weights are NOT included
    all_params = list(model.parameters())
    assert len(trainable) < len(all_params)

def test_get_trainable_params_nonzero():
    model = nn.Linear(4, 4)
    model.weight.requires_grad = True
    trainable = get_trainable_fsdp_params(model)
    assert len(trainable) > 0

def test_is_main_process_rank0():
    assert is_main_process(0) is True

def test_is_main_process_rank1():
    assert is_main_process(1) is False

def test_print_on_main_rank0_prints():
    with mock.patch('builtins.print') as mock_print:
        print_on_main("test message", 0)
        mock_print.assert_called_once_with("test message")

def test_print_on_main_rank1_silent():
    with mock.patch('builtins.print') as mock_print:
        print_on_main("test message", 1)
        mock_print.assert_not_called()

def test_init_distributed_raises_without_env():
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError) as excinfo:
            init_distributed()
        assert "MASTER_ADDR" in str(excinfo.value)

def test_save_fsdp_skips_non_zero_rank():
    model = nn.Linear(4, 4)
    # Mock FSDP.summon_full_params context manager
    with mock.patch('torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params') as mock_summon:
        mock_summon.return_value.__enter__.return_value = None
        with mock.patch('torch.save') as mock_save:
            save_fsdp_lora_weights(model, "dummy.pt", rank=1)
            mock_save.assert_not_called()

def test_get_transformer_layer_class_returns_type():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    # Add dummy config for auto-detection
    model.config = type('Config', (), {'model_type': 'llama'})()
    cls = get_transformer_layer_class(model)
    assert isinstance(cls, type)

def test_get_distributed_info_keys():
    info = get_distributed_info()
    required = {"rank", "local_rank", "world_size", "backend", "nccl_available", "rccl_available"}
    assert all(k in info for k in required)
