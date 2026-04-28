"""
ROCm-native profiling utilities for rocm-qlora.
"""
import torch
import torch.nn as nn
import os
import time
from typing import Dict, Any, List, Optional, Tuple

from rocm_qlora.fp8 import detect_fp8_support

class ROCmProfiler:
    """
    Handles training step profiling and model benchmarking.
    """
    def __init__(
        self, 
        output_dir: str = "./profiles", 
        profile_memory: bool = True, 
        record_shapes: bool = True, 
        with_stack: bool = False
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        
        # Detect rocprofiler-sdk
        try:
            import rocprofiler_sdk
            self.rocprofiler_available = True
        except ImportError:
            self.rocprofiler_available = False
            
        self.fp8_active = detect_fp8_support()['fp8_supported']

    def profile_training_step(self, model: nn.Module, batch: Dict[str, torch.Tensor], device: str, label: str = "step") -> Dict[str, Any]:
        """
        Profiles a single training step (forward + backward).
        """
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        trace_path = os.path.join(self.output_dir, f"{label}_trace.json")
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir)
        ) as prof:
            # 1. Forward
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs.mean()
            # 2. Backward
            loss.backward()

        # Export Chrome trace
        prof.export_chrome_trace(trace_path)
        
        # Analyze results
        stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        
        # Extract top ops
        top_ops = []
        fp8_detected = False
        for entry in prof.key_averages():
            if entry.cuda_time_total > 0:
                name = entry.key
                top_ops.append({
                    "name": name,
                    "cuda_time_ms": entry.cuda_time_total / 1000,
                    "cpu_time_ms": entry.cpu_time_total / 1000,
                    "count": entry.count
                })
                # Check for FP8 indicators in op names
                if any(x in name.lower() for x in ["fp8", "cast", "te_linear"]):
                    fp8_detected = True
        
        top_ops = sorted(top_ops, key=lambda x: x["cuda_time_ms"], reverse=True)[:5]
        
        # Get memory stats
        vram_stats = torch.cuda.memory_stats(device) if torch.cuda.is_available() else {}
        peak_vram = vram_stats.get("allocated_bytes.all.peak", 0) / (1024**3)

        return {
            "total_step_ms": sum(x["cuda_time_ms"] for x in top_ops), # rough estimate
            "peak_vram_gb": peak_vram,
            "top_5_ops": top_ops,
            "trace_path": trace_path,
            "fp8_kernels_detected": fp8_detected
        }

    def profile_model_summary(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 512)) -> Dict[str, Any]:
        """Returns a structural and memory summary of the model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count QuantLinear buffers
        quant_mb = 0
        from rocm_qlora.quantization.quant_linear import QuantLinear
        for m in model.modules():
            if isinstance(m, QuantLinear):
                for b in m.buffers():
                    quant_mb += b.numel() * b.element_size()
        
        quant_mb = quant_mb / (1024**2)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "quant_buffers_mb": quant_mb,
            "lora_params_mb": (trainable_params * 2) / (1024**2), # assume FP16
            "estimated_activation_mb": (torch.prod(torch.tensor(input_shape)).item() * 4096 * 2) / (1024**2), # rough
            "flops_per_forward_estimate": total_params * 2
        }

    def benchmark_forward_pass(self, model: nn.Module, batch: Dict[str, torch.Tensor], device: str, n_runs: int = 20, warmup_runs: int = 5) -> Dict[str, Any]:
        """Measures mean latency and tokens/sec."""
        model.eval()
        device = torch.device(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(**batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
        mean_ms = sum(times) / n_runs
        
        # Tokens per second calculation
        batch_size = batch["input_ids"].shape[0] if "input_ids" in batch else 1
        seq_len = batch["input_ids"].shape[1] if "input_ids" in batch else 512
        tps = (batch_size * seq_len) / (mean_ms / 1000)
        
        return {
            "mean_ms": mean_ms,
            "std_ms": torch.tensor(times).std().item() if n_runs > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "n_runs": n_runs,
            "tokens_per_second": tps
        }

    def compare_configurations(self, model_configs: Dict[str, nn.Module], batch: Dict[str, torch.Tensor], device: str, n_runs: int = 20) -> Dict[str, Any]:
        """Compares multiple model versions and prints a speedup table."""
        results = {}
        baseline_name = list(model_configs.keys())[0]
        
        print(f"\n{'Configuration':<25} | {'Mean MS':<10} | {'Speedup':<10} | {'Tokens/s':<10}")
        print("-" * 65)
        
        baseline_ms = 1.0
        for name, model in model_configs.items():
            res = self.benchmark_forward_pass(model, batch, device, n_runs=n_runs)
            if name == baseline_name:
                baseline_ms = res["mean_ms"]
                speedup = 1.0
            else:
                speedup = baseline_ms / res["mean_ms"]
                
            results[name] = {
                "mean_ms": res["mean_ms"],
                "speedup_vs_baseline": speedup,
                "tokens_per_second": res["tokens_per_second"]
            }
            
            print(f"{name:<25} | {res['mean_ms']:>10.2f} | {speedup:>10.2f}x | {res['tokens_per_second']:>10.1f}")
            
        return results

    def get_rocprofiler_status(self) -> Dict[str, Any]:
        """Returns rocprofiler-sdk installation status."""
        return {
            "available": self.rocprofiler_available,
            "version": "6.1+" if self.rocprofiler_available else None,
            "install_command": "pip install rocprofiler-sdk"
        }
