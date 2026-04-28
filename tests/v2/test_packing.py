import pytest
import torch
import logging
from rocm_qlora.data.packing import (
    pack_sequences, sort_by_length, compute_packing_efficiency
)
from rocm_qlora.data.collator import (
    PackedSequenceCollator, build_position_ids, build_block_attention_mask
)

@pytest.fixture
def fake_samples():
    """Create fake tokenized samples of varying lengths."""
    return [
        {'input_ids': [100+i for i in range(50)],  'attention_mask': [1]*50,  'labels': [100+i for i in range(50)]},
        {'input_ids': [200+i for i in range(120)], 'attention_mask': [1]*120, 'labels': [200+i for i in range(120)]},
        {'input_ids': [300+i for i in range(80)],  'attention_mask': [1]*80,  'labels': [300+i for i in range(80)]},
        {'input_ids': [400+i for i in range(200)], 'attention_mask': [1]*200, 'labels': [400+i for i in range(200)]},
        {'input_ids': [500+i for i in range(30)],  'attention_mask': [1]*30,  'labels': [500+i for i in range(30)]},
    ]

def test_no_pack_exceeds_max_length(fake_samples):
    max_len = 512
    packed = pack_sequences(fake_samples, max_len, eos_token_id=1, pad_token_id=0)
    for p in packed:
        assert len(p['input_ids']) == max_len

def test_eos_separator_present(fake_samples):
    max_len = 512
    eos_id = 1
    # Sample 0 (50) + Sample 1 (120) = 170. 170 + 2 EOS = 172.
    packed = pack_sequences(fake_samples, max_len, eos_token_id=eos_id, pad_token_id=0)
    # Check if EOS is present at expected boundaries
    # The packing logic adds EOS after each sample.
    ids = packed[0]['input_ids']
    # 50 tokens + 1 EOS + 120 tokens + 1 EOS + ...
    assert ids[50] == eos_id
    assert ids[50 + 1 + 120] == eos_id

def test_eos_label_is_minus100(fake_samples):
    max_len = 512
    eos_id = 1
    packed = pack_sequences(fake_samples, max_len, eos_token_id=eos_id, pad_token_id=0)
    for p in packed:
        for i, val in enumerate(p['input_ids']):
            if val == eos_id:
                assert p['labels'][i] == -100

def test_no_sample_skipped(fake_samples):
    max_len = 512
    # Total real tokens in original
    original_total = sum(len(s['input_ids']) for s in fake_samples)
    
    packed = pack_sequences(fake_samples, max_len, eos_token_id=1, pad_token_id=0)
    # Total real tokens in packed (excluding EOS added by packing and padding)
    # Wait, the efficiency calculation considers EOS as real tokens in some contexts?
    # No, attention_mask = 1 for EOS in packing logic.
    packed_total_with_eos = sum(sum(p['attention_mask']) for p in packed)
    
    # original_total + len(fake_samples) (one EOS per sample)
    assert packed_total_with_eos == original_total + len(fake_samples)

def test_sort_by_length_descending(fake_samples):
    sorted_samples = sort_by_length(fake_samples)
    lengths = [len(s['input_ids']) for s in sorted_samples]
    assert lengths == sorted(lengths, reverse=True)

def test_efficiency_improves_with_sort(fake_samples):
    max_len = 300
    # Unsorted
    packed_u = pack_sequences(fake_samples, max_len, 1, 0)
    eff_u = compute_packing_efficiency(fake_samples, packed_u, max_len)['efficiency_pct']
    
    # Sorted
    sorted_s = sort_by_length(fake_samples)
    packed_s = pack_sequences(sorted_s, max_len, 1, 0)
    eff_s = compute_packing_efficiency(fake_samples, packed_s, max_len)['efficiency_pct']
    
    # Heuristic: sorting usually improves or keeps efficiency same
    assert eff_s >= eff_u

def test_efficiency_calculation_formula():
    # 1 sample of 100 real tokens + 1 EOS = 101 real tokens in window of 512.
    # efficiency = 100 / (100 + padding) ? No, prompt says:
    # original_tokens = sum(s['attention_mask']) -> 100
    # padding_tokens = packed['attention_mask'].count(0) -> 512 - 101 = 411
    # efficiency = (100 / (100 + 411)) * 100 = (100 / 511) * 100 approx 19.5%
    # Wait, 100 + 411 = 511. 100 / 511 = 0.1956. Correct.
    
    orig = [{'attention_mask': [1]*100}]
    packed = [{'attention_mask': [1]*101 + [0]*411}] # 100 real + 1 EOS + 411 pad
    res = compute_packing_efficiency(orig, packed, 512)
    assert pytest.approx(res['efficiency_pct'], 0.1) == 19.56

def test_compression_ratio_positive(fake_samples):
    packed = pack_sequences(fake_samples, 512, 1, 0)
    res = compute_packing_efficiency(fake_samples, packed, 512)
    assert res['compression_ratio'] >= 1.0

def test_oversized_sample_skipped_with_warning(caplog):
    samples = [{'input_ids': [1]*10, 'attention_mask': [1]*10, 'labels': [1]*10}]
    with caplog.at_level(logging.WARNING):
        packed = pack_sequences(samples, max_length=5, eos_token_id=1, pad_token_id=0)
    assert len(packed) == 0
    assert "exceeds max_length" in caplog.text

def test_zero_waste_perfect_packing():
    # 10 samples of 50 tokens. Each needs 50 + 1 (EOS) = 51.
    # 10 * 51 = 510. Max length 512. 2 pads.
    samples = [{'input_ids': [1]*50, 'attention_mask': [1]*50, 'labels': [1]*50} for _ in range(10)]
    packed = pack_sequences(samples, 512, 1, 0)
    assert len(packed) == 1
    eff = compute_packing_efficiency(samples, packed, 512)
    # original_tokens = 500. padding = 2. eff = 500 / 502 * 100 = 99.6%
    assert eff['efficiency_pct'] > 99.0

def test_build_position_ids_resets_at_eos():
    # [10, 20, 1, 30, 40, 50, 1, 60] with EOS=1
    ids = torch.tensor([[10, 20, 1, 30, 40, 50, 1, 60]])
    pos = build_position_ids(ids, eos_token_id=1)
    expected = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0]])
    torch.testing.assert_close(pos, expected)

def test_position_ids_shape():
    ids = torch.randn(2, 16).long()
    pos = build_position_ids(ids, 1)
    assert pos.shape == ids.shape

def test_collator_returns_required_keys():
    collator = PackedSequenceCollator(0, 1)
    features = [{'input_ids': [1]*10, 'attention_mask': [1]*10, 'labels': [1]*10}]
    batch = collator(features)
    assert set(batch.keys()) == {'input_ids', 'attention_mask', 'labels', 'position_ids'}

def test_collator_output_is_tensors():
    collator = PackedSequenceCollator(0, 1)
    features = [{'input_ids': [1]*10, 'attention_mask': [1]*10, 'labels': [1]*10}]
    batch = collator(features)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)

def test_collator_position_ids_dtype_long():
    collator = PackedSequenceCollator(0, 1)
    features = [{'input_ids': [1]*10, 'attention_mask': [1]*10, 'labels': [1]*10}]
    batch = collator(features)
    assert batch['position_ids'].dtype == torch.long

def test_block_attention_mask_shape():
    ids = torch.tensor([[10, 20, 1, 30, 40, 50, 1, 60]])
    mask = build_block_attention_mask(ids, 1)
    assert mask.shape == (1, 1, 8, 8)

def test_block_attention_mask_no_cross_attention():
    # [10, 20, 1, | 30, 40, 50, 1, | 60]
    # Indices: 0, 1, 2 | 3, 4, 5, 6 | 7
    ids = torch.tensor([[10, 20, 1, 30, 40, 50, 1, 60]])
    mask = build_block_attention_mask(ids, 1)
    # Token 3 (start of seq 2) should NOT attend to token 0 (seq 1)
    assert mask[0, 0, 3, 0] == 0
    # Token 1 should attend to token 0 (causal, same seq)
    assert mask[0, 0, 1, 0] == 1
    # Token 0 should NOT attend to token 1 (causal)
    assert mask[0, 0, 0, 1] == 0
