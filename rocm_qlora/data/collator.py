"""
Custom data collator for packed sequences.
Handles position_ids reset and optional block-diagonal attention masks.
"""
import torch
from typing import List, Dict, Any

def build_position_ids(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Builds position IDs that reset after every EOS token.
    Input: [batch_size, seq_len]
    Output: [batch_size, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    position_ids = torch.zeros_like(input_ids, dtype=torch.long)
    
    for b in range(batch_size):
        current_pos = 0
        for s in range(seq_len):
            position_ids[b, s] = current_pos
            if input_ids[b, s] == eos_token_id:
                current_pos = 0
            else:
                current_pos += 1
                
    return position_ids

def build_block_attention_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Builds a 4D block-diagonal causal attention mask.
    Prevents tokens in one sequence from attending to tokens in another sequence.
    Input: [batch_size, seq_len]
    Returns: [batch_size, 1, seq_len, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    # Initialize with causal mask (lower triangular)
    # mask[i, j] = 1 means i can attend to j
    mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool))
    
    for b in range(batch_size):
        # Identify sequence boundaries
        # seq_boundaries = (input_ids[b] == eos_token_id).nonzero().flatten()
        # Instead of scanning once, we can build a 'segment_id' for each token
        segment_id = torch.zeros(seq_len, dtype=torch.long)
        curr_segment = 0
        for s in range(seq_len):
            segment_id[s] = curr_segment
            if input_ids[b, s] == eos_token_id:
                curr_segment += 1
        
        # tokens in different segments cannot attend to each other
        # segment_id[i] != segment_id[j] => mask[b, 0, i, j] = 0
        seg_i = segment_id.view(seq_len, 1)
        seg_j = segment_id.view(1, seq_len)
        same_segment = (seg_i == seg_j)
        
        mask[b, 0] = mask[b, 0] & same_segment
        
    return mask.to(torch.float32)

class PackedSequenceCollator:
    """
    Collator for packed sequences that ensures position_ids are correctly set.
    """
    def __init__(
        self, 
        pad_token_id: int, 
        eos_token_id: int, 
        use_block_attention: bool = False
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.use_block_attention = use_block_attention

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.
        """
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        # Build position IDs
        position_ids = build_position_ids(input_ids, self.eos_token_id)
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'position_ids': position_ids
        }
        
        if self.use_block_attention:
            # Note: HuggingFace models usually expect the attention_mask to be 4D
            # if it's a custom mask, or 2D for the standard case.
            # We override the attention_mask with our 4D block mask.
            batch['attention_mask'] = build_block_attention_mask(input_ids, self.eos_token_id)
            
        return batch
