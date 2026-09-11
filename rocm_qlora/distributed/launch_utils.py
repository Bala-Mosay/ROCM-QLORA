"""
Multi-GPU launch utilities for ROCm.

ROCm distributed training uses RCCL (ROCm Collective Communications Library)
which is AMD's equivalent of NCCL. PyTorch on ROCm automatically maps
the "nccl" backend string to RCCL — always use "nccl", never "rccl".

Launch with torchrun:
  torchrun --nproc_per_node=NUM_GPUS train_v3_fsdp.py [args]
"""
import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Tuple, Dict, Any

def init_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> Tuple[int, int, int]:
    """
    Initializes the distributed process group for ROCm (RCCL).
    Returns: (rank, local_rank, world_size)
    """
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "MASTER_ADDR/MASTER_PORT not set. "
            "Please launch with: torchrun --nproc_per_node=N script.py"
        )
        
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # NOTE: PyTorch on ROCm transparently maps nccl -> RCCL.
    dist.init_process_group(
        backend=backend, 
        timeout=timedelta(minutes=timeout_minutes)
    )
    
    return rank, local_rank, world_size

def setup_device(local_rank: int) -> torch.device:
    """
    Sets the current CUDA device and returns the device object.
    Falls back to CPU if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")

def cleanup_distributed():
    """
    Cleans up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    """Returns True if the current process is the main process (Rank 0)."""
    return rank == 0

def print_on_main(message: str, rank: int):
    """Prints a message only on the main process."""
    if rank == 0:
        print(message)

def barrier(rank: int):
    """Synchronizes all processes in the group."""
    if dist.is_initialized():
        # print_on_main("[distributed] Barrier sync...", rank)
        dist.barrier()

def get_distributed_info() -> Dict[str, Any]:
    """Returns telemetry about the distributed environment."""
    return {
        "rank": int(os.environ.get("RANK", -1)),
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "world_size": int(os.environ.get("WORLD_SIZE", -1)),
        "backend": dist.get_backend() if dist.is_initialized() else "none",
        "nccl_available": torch.cuda.is_available() and dist.is_nccl_available(),
        "rccl_available": dist.is_nccl_available() # On ROCm, nccl_available implies RCCL
    }
