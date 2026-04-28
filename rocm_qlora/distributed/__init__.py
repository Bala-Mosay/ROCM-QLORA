"""
Multi-GPU distributed training utilities for rocm-qlora.
Uses PyTorch FSDP with ROCm RCCL backend (exposed as 'nccl').
"""
from rocm_qlora.distributed.fsdp_policy import (
    get_qlora_fsdp_policy,
    prepare_model_for_fsdp,
    get_trainable_fsdp_params,
    save_fsdp_lora_weights,
    get_transformer_layer_class,
)
from rocm_qlora.distributed.launch_utils import (
    init_distributed,
    setup_device,
    cleanup_distributed,
    is_main_process,
    print_on_main,
    barrier,
    get_distributed_info,
)

__all__ = [
    "get_qlora_fsdp_policy", 
    "prepare_model_for_fsdp",
    "get_trainable_fsdp_params", 
    "save_fsdp_lora_weights",
    "get_transformer_layer_class",
    "init_distributed", 
    "setup_device", 
    "cleanup_distributed",
    "is_main_process", 
    "print_on_main", 
    "barrier", 
    "get_distributed_info",
]
