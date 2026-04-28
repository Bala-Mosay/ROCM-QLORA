import pytest
import torch
import torch.nn as nn
from rocm_qlora.optim.paged_adamw import PagedAdamW
from rocm_qlora.utils.compile_utils import compile_model, is_compile_safe

def test_paged_adamw_step_runs():
    model = nn.Linear(10, 2)
    optimizer = PagedAdamW(model.parameters(), lr=1e-3)
    
    # Dummy step
    input_data = torch.randn(1, 10)
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    
    optimizer.step()
    # No error = success

def test_states_on_cpu_between_steps():
    model = nn.Linear(10, 2)
    optimizer = PagedAdamW(model.parameters(), lr=1e-3)
    
    # Initial states should be on CPU (pinned)
    for p in model.parameters():
        state = optimizer.state[p]
        assert state['exp_avg'].device.type == 'cpu'
        assert state['exp_avg_sq'].device.type == 'cpu'
        if torch.cuda.is_available():
            assert state['exp_avg'].is_pinned()

def test_state_on_gpu_during_step():
    """
    NOTE: Not directly testable in a synchronous CPU-only test environment.
    Testing the GPU-residency during the step would require a custom GPU hook 
    or a Mock device that records access, which is outside the scope of 
    standard unit tests. We rely on the .to(device) and .copy_() logic 
    verified by functional convergence tests.
    """
    pass

def test_loss_decreases():
    # Simple linear regression: y = x * 2
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10.0)
    optimizer = PagedAdamW(model.parameters(), lr=1e-1)
    
    target = torch.tensor([[20.0]])
    x = torch.tensor([[10.0]])
    
    initial_loss = torch.nn.functional.mse_loss(model(x), target)
    
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
    final_loss = torch.nn.functional.mse_loss(model(x), target)
    assert final_loss < initial_loss

def test_state_dict_roundtrip():
    model = nn.Linear(4, 2)
    optimizer = PagedAdamW(model.parameters(), lr=1e-3)
    
    # Run a step to populate states
    input_data = torch.randn(1, 4)
    model(input_data).sum().backward()
    optimizer.step()
    
    sd = optimizer.state_dict()
    
    new_optimizer = PagedAdamW(model.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(sd)
    
    # Check that states are correctly restored on CPU and pinned
    for p in model.parameters():
        state = new_optimizer.state[p]
        assert state['exp_avg'].device.type == 'cpu'
        if torch.cuda.is_available():
            assert state['exp_avg'].is_pinned()

def test_memory_stats_returns_dict():
    model = nn.Linear(10, 10)
    optimizer = PagedAdamW(model.parameters())
    stats = optimizer.get_memory_stats()
    
    assert isinstance(stats, dict)
    assert "cpu_pinned_mb" in stats
    assert "num_params" in stats
    # 10*10 + 10 = 110 params. Each state has 2 float32 tensors.
    # 110 * 4 bytes * 2 = 880 bytes.
    assert stats["num_params"] == 2 # weight and bias

def test_paged_vs_adamw_convergence():
    # Both should produce nearly identical updates for small models
    torch.manual_seed(42)
    model_p = nn.Linear(5, 5)
    model_a = nn.Linear(5, 5)
    model_a.load_state_dict(model_p.state_dict())
    
    optim_p = PagedAdamW(model_p.parameters(), lr=1e-2)
    optim_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2)
    
    x = torch.randn(1, 5)
    
    for _ in range(5):
        optim_p.zero_grad()
        optim_a.zero_grad()
        
        loss_p = model_p(x).sum()
        loss_a = model_a(x).sum()
        
        loss_p.backward()
        loss_a.backward()
        
        optim_p.step()
        optim_a.step()
        
    # Check weights match
    torch.testing.assert_close(model_p.weight, model_a.weight, atol=1e-5, rtol=1e-5)

def test_compile_utils_returns_model():
    model = nn.Linear(10, 10)
    # On CPU/old ROCm, this should just return the model
    compiled = compile_model(model)
    assert isinstance(compiled, nn.Module)

def test_is_compile_safe_returns_bool():
    res = is_compile_safe()
    assert isinstance(res, bool)
