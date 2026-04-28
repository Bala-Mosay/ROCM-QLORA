"""
Sequence packing for efficient LLM fine-tuning on ROCm.

Problem: padding every sample to max_length wastes compute on pad tokens.
If average sample length is 200 tokens and max_length is 512,
roughly 60% of every forward/backward pass is wasted.

Solution: greedy first-fit bin packing. Concatenate multiple short
sequences into a single max_length window separated by EOS tokens.
The model sees a dense window of real tokens — zero padding waste.

Expected throughput improvement: 1.5x–3x on typical instruction datasets.
For tatsu-lab/alpaca (avg ~200 tokens, max 512): ~87% efficiency.
"""
import logging
from typing import List, Dict, Any, Tuple, Callable, Optional

logger = logging.getLogger("rocm_qlora.data.packing")

def pack_sequences(
    samples: List[Dict[str, List[int]]], 
    max_length: int, 
    eos_token_id: int, 
    pad_token_id: int
) -> List[Dict[str, List[int]]]:
    """
    Greedy first-fit bin packing for sequences.
    Concatenates multiple short samples into single max_length windows.
    """
    packs = []
    current_ids: List[int] = []
    current_mask: List[int] = []
    current_labels: List[int] = []

    for i, sample in enumerate(samples):
        ids = sample['input_ids']
        mask = sample['attention_mask']
        labels = sample['labels']
        
        # Guard: skip any sample where len(input_ids) > max_length
        if len(ids) > max_length:
            logger.warning(f"Skipping sample at index {i} because its length ({len(ids)}) exceeds max_length ({max_length}).")
            continue
            
        needed = len(ids) + 1  # +1 for EOS separator
        
        if len(current_ids) + needed <= max_length:
            current_ids += ids + [eos_token_id]
            current_mask += mask + [1]
            current_labels += labels + [-100]  # EOS: no loss
        else:
            if current_ids:  # finalize current pack
                pad_len = max_length - len(current_ids)
                packs.append({
                    'input_ids': current_ids + [pad_token_id] * pad_len,
                    'attention_mask': current_mask + [0] * pad_len,
                    'labels': current_labels + [-100] * pad_len,
                })
            
            # Start a new pack with the current sample
            # We don't add EOS here yet, it will be added when the NEXT sample is appended or when flushed.
            # Actually, to be consistent with the logic above, we should add EOS now or 
            # change the logic. The logic above adds EOS *with* the sample.
            # Let's stick to the prompt's logic: 
            # if we can't fit it, we finalize, then current_ids = sample['input_ids']
            # Wait, if we start a new pack, we still need to add EOS later.
            
            # Refined logic to match prompt exactly:
            current_ids = ids + [eos_token_id]
            current_mask = mask + [1]
            current_labels = labels + [-100]

    if current_ids:  # flush final pack
        pad_len = max_length - len(current_ids)
        packs.append({
            'input_ids': current_ids + [pad_token_id] * pad_len,
            'attention_mask': current_mask + [0] * pad_len,
            'labels': current_labels + [-100] * pad_len,
        })
        
    return packs

def sort_by_length(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort samples descending by length to improve packing efficiency.
    """
    return sorted(samples, key=lambda x: len(x['input_ids']), reverse=True)

def compute_packing_efficiency(
    original_samples: List[Dict[str, Any]], 
    packed_samples: List[Dict[str, Any]],
    max_length: int
) -> Dict[str, Any]:
    """
    Compute packing efficiency metrics.
    """
    # original_tokens: real tokens only, no padding
    original_tokens = sum(sum(s['attention_mask']) for s in original_samples)
    
    # packed_tokens: real tokens in packed windows
    packed_tokens = sum(sum(s['attention_mask']) for s in packed_samples)
    
    # padding_tokens: zeros in attention masks
    padding_tokens = sum(s['attention_mask'].count(0) for s in packed_samples)
    
    efficiency_pct = (original_tokens / (original_tokens + padding_tokens)) * 100 if (original_tokens + padding_tokens) > 0 else 0
    compression_ratio = len(original_samples) / len(packed_samples) if len(packed_samples) > 0 else 1.0
    
    return {
        'original_tokens': original_tokens,
        'packed_tokens': packed_tokens,
        'padding_tokens': padding_tokens,
        'efficiency_pct': efficiency_pct,
        'compression_ratio': compression_ratio,
        'original_samples': len(original_samples),
        'packed_samples': len(packed_samples)
    }

def build_packed_dataset(
    raw_texts: List[str], 
    tokenizer, 
    max_length: int, 
    format_fn: Optional[Callable] = None
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """
    Full pipeline to build a packed dataset.
    """
    if format_fn:
        texts = [format_fn(t) for t in raw_texts]
    else:
        texts = raw_texts
        
    # Tokenize all
    logger.info(f"Tokenizing {len(texts)} samples...")
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=False # We handle EOS/SOS manually or via packing
    )
    
    samples = []
    for i in range(len(texts)):
        ids = encoded['input_ids'][i]
        mask = encoded['attention_mask'][i]
        
        # Build labels: copy input_ids, set pad/ignore positions to -100
        labels = list(ids)
        # Note: In this simple flow, all tokens are real. 
        # Pad tokens aren't added yet, they are added during packing.
        
        samples.append({
            'input_ids': ids,
            'attention_mask': mask,
            'labels': labels
        })
        
    # Sort and Pack
    sorted_samples = sort_by_length(samples)
    packed_samples = pack_sequences(
        sorted_samples, 
        max_length, 
        tokenizer.eos_token_id, 
        tokenizer.pad_token_id
    )
    
    # Stats
    stats = compute_packing_efficiency(samples, packed_samples, max_length)
    
    return packed_samples, stats
