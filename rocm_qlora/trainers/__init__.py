"""
Post-training alignment trainers for rocm-qlora.

Training progression:
  SFT  -> teaches format and instruction following (start here)
  DPO  -> aligns outputs with human preferences (after SFT)
  GRPO -> develops reasoning capabilities (after SFT, no labels needed)

All trainers:
  - Accept both FSDP and non-FSDP models
  - Default to PagedAdamW (0 VRAM optimizer states)
  - Support sequence packing via PackedSequenceCollator
  - Train only LoRA params — base quantized weights always frozen
"""
from rocm_qlora.trainers.sft_trainer import ROCmSFTTrainer, SFTConfig
from rocm_qlora.trainers.dpo_trainer import ROCmDPOTrainer, DPOConfig
from rocm_qlora.trainers.grpo_trainer import ROCmGRPOTrainer, GRPOConfig

__all__ = [
    "ROCmSFTTrainer", "SFTConfig",
    "ROCmDPOTrainer", "DPOConfig",
    "ROCmGRPOTrainer", "GRPOConfig",
]
