"""
Efficient data pipeline for rocm-qlora.
Sequence packing eliminates padding waste — the largest free throughput gain.
"""
from rocm_qlora.data.packing import (
    pack_sequences, 
    sort_by_length,
    compute_packing_efficiency, 
    build_packed_dataset,
)
from rocm_qlora.data.collator import (
    PackedSequenceCollator, 
    build_position_ids, 
    build_block_attention_mask,
)

__all__ = [
    "pack_sequences", 
    "sort_by_length",
    "compute_packing_efficiency", 
    "build_packed_dataset",
    "PackedSequenceCollator", 
    "build_position_ids", 
    "build_block_attention_mask",
]
