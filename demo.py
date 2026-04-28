"""
Interactive demo and inference script for rocm-qlora.
Reconstructs a quantized model, reloads LoRA weights, and provides base vs. fine-tuned comparisons.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rocm_qlora import quantize_model, LoRALinear, check_rocm
from rocm_qlora.utils.rocm_utils import get_memory_stats

def generate(model, tokenizer, prompt, device, max_new_tokens=150):
    """Greedy deterministic generation helper."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic for comparison
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # decode only the generated tokens, not the prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    # 1. Config loading block
    config_path = "./outputs/training_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("training_config.json not found. Run train.py first.")
        
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"[demo] Reconstructing model from config: {config['model_id']}")
    
    # 2. Model reconstruction (CPU first, then quantize, then move to GPU)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = quantize_model(
        model,
        bits=config["bits"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        block_size=config["block_size"],
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 3. Load LoRA weights
    lora_path = "./outputs/lora_weights.pt"
    if not os.path.exists(lora_path):
        raise FileNotFoundError("lora_weights.pt not found. Run train.py first.")
        
    print(f"[demo] Loading LoRA weights from {lora_path}")
    lora_weights = torch.load(lora_path, map_location=device, weights_only=True)
    
    # NOTE: strict=False is required because the state_dict only contains LoRA parameters.
    missing, unexpected = model.load_state_dict(lora_weights, strict=False)
    print(f"[demo] Loaded LoRA weights: {len(lora_weights)} tensors")
    print(f"[demo] Missing keys: {len(missing)} (Base weights - expected)")
    print(f"[demo] Unexpected keys: {len(unexpected)}")
    
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
        
    # 4. Inference Comparison
    TEST_PROMPTS = [
        "### Instruction:\nExplain what a large language model is in simple terms.\n\n### Response:\n",
        "### Instruction:\nWrite a short poem about the mountains.\n\n### Response:\n",
        "### Instruction:\nGive me a step-by-step recipe for making scrambled eggs.\n\n### Response:\n",
    ]
    
    print("\n" + "=" * 70)
    print(" BASE VS FINE-TUNED COMPARISON ".center(70, "="))
    print("=" * 70)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        # Run 1: Base model (Zero out LoRA temporarily)
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data.zero_()
                module.lora_B.data.zero_()
        
        base_response = generate(model, tokenizer, prompt, device)
        
        # Run 2: Fine-tuned model (Restore LoRA weights)
        model.load_state_dict(lora_weights, strict=False)
        finetuned_response = generate(model, tokenizer, prompt, device)
        
        print("\n" + "=" * 70)
        print(f"PROMPT {i+1}: {prompt.split('Instruction:')[1].split('Response:')[0].strip()[:60]}...")
        print("-" * 35 + " BASE " + "-" * 29)
        print(base_response.strip())
        print("-" * 32 + " FINE-TUNED " + "-" * 26)
        print(finetuned_response.strip())
        print("=" * 70)
        
    # 5. Final Merge and Save
    # NOTE: Wrap in eval() and no_grad() to prevent autograd tracking errors during buffer writes on ROCm.
    print("\n[demo] Merging LoRA weights into base model for deployment...")
    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and not module.merged:
                module.merge_lora()
    print("[demo] Merge complete.")
    
    os.makedirs("./outputs/merged_model", exist_ok=True)
    torch.save(model.state_dict(), "./outputs/merged_model/model_state.pt")
    print("[demo] Merged model saved to ./outputs/merged_model/model_state.pt")
    
    stats = get_memory_stats()
    print(f"[demo] Final VRAM: {stats['allocated_gb']:.2f}GB allocated")

if __name__ == "__main__":
    main()
