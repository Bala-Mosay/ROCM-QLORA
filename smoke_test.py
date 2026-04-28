"""
smoke_test.py — rocm-qlora integration test
Run: python smoke_test.py
Exit 0 = PASS, Exit 1 = FAIL
No GPU required. No external downloads.
"""

import sys
import torch
import torch.nn as nn
from typing import List

# 1. Import test
try:
    from rocm_qlora import quantize_model, QuantLinear, LoRALinear, check_rocm
    from rocm_qlora.utils.rocm_utils import get_memory_stats, print_model_summary
except ImportError as e:
    print(f"[smoke_test] [ FAIL ] Import test: {e}")
    sys.exit(1)

# Mock model for cold tests (no HuggingFace)
class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)

class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeAttention() for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.q_proj(x)
        return x

def run_smoke_test():
    print(f"[smoke_test] Running 12 checks...")
    results = []
    
    def check(name, condition, error_msg=""):
        if condition:
            print(f"[  OK  ] {name}")
            results.append(True)
        else:
            print(f"[ FAIL ] {name} — {error_msg}")
            results.append(False)

    # 1. Import check
    check("Import test", True)

    # 2. Build fake model
    try:
        model = FakeTransformer()
        check("Build fake model", True)
    except Exception as e:
        check("Build fake model", False, str(e))

    # 3. Quantize
    try:
        model = quantize_model(
            model, 
            bits=8, 
            lora_r=4, 
            target_modules=["q_proj", "v_proj"]
        )
        check("Quantize model", True)
    except Exception as e:
        check("Quantize model", False, str(e))

    # 4. Structure check
    all_replaced = True
    for i in range(2):
        if not isinstance(model.layers[i].q_proj, LoRALinear) or \
           not isinstance(model.layers[i].v_proj, LoRALinear):
            all_replaced = False
    check("Structure check", all_replaced)

    # 5. Forward pass
    try:
        x = torch.randn(1, 128)
        out = model(x)
        check("Forward pass", out.shape == (1, 128))
    except Exception as e:
        check("Forward pass", False, str(e))

    # 6. Backward pass
    try:
        out.sum().backward()
        check("Backward pass", True)
    except Exception as e:
        check("Backward pass", False, str(e))

    # 7. Gradient check
    grad_params = [n for n, p in model.named_parameters() if p.grad is not None]
    only_lora = all("lora_A" in n or "lora_B" in n for n in grad_params) and len(grad_params) > 0
    check("Gradient check", only_lora, f"Found grads in: {grad_params}")

    # 8. Frozen base check
    base_frozen = True
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            for p in module.parameters():
                if p.requires_grad:
                    base_frozen = False
    check("Frozen base check", base_frozen)

    # 9. Trainable param count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("Trainable param count", trainable > 0, f"Found {trainable} params")

    # 10. Zero init check
    fresh_linear = nn.Linear(64, 64)
    fresh_quant = QuantLinear.from_linear(fresh_linear)
    fresh_lora = LoRALinear(fresh_quant, r=4)
    x_64 = torch.randn(1, 64)
    with torch.no_grad():
        base_out = fresh_quant(x_64)
        lora_out = fresh_lora(x_64)
    check("Zero init check", torch.allclose(base_out, lora_out))

    # 11. Memory check
    check("Memory check", fresh_quant.weight_quant.dtype == torch.int8)

    # 12. check_rocm returns dict
    info = check_rocm()
    check("check_rocm returns dict", isinstance(info, dict) and "available" in info)

    # Final summary
    passed = sum(results)
    total = len(results)
    print("=" * 30)
    print(f"RESULT: {passed}/{total} passed")
    if passed < total:
        failed_names = [f"Check {i+1}" for i, r in enumerate(results) if not r]
        print(f"FAILED: {failed_names}")
        print("=" * 30)
        sys.exit(1)
    else:
        print("=" * 30)
        sys.exit(0)

if __name__ == "__main__":
    run_smoke_test()
