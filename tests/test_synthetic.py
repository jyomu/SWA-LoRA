import random

from transformers import AutoTokenizer

from swa_lora.synthetic import build_relay_training_block, make_relay_gap

TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"


def _tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def test_relay_training_block_has_exact_length():
    tokenizer = _tokenizer()
    ids, _ = build_relay_training_block(tokenizer, seq_length=512, gap=200, key="482913")
    assert len(ids) == 512


def test_relay_training_block_contains_key_near_the_start():
    tokenizer = _tokenizer()
    key = "482913"
    ids, _ = build_relay_training_block(tokenizer, seq_length=1024, gap=400, key=key)
    head_text = tokenizer.decode(ids[:60])
    assert key in head_text


def test_relay_training_block_code_span_matches_the_key():
    tokenizer = _tokenizer()
    key = "482913"
    ids, (start, end) = build_relay_training_block(tokenizer, seq_length=1024, gap=400, key=key)
    span_text = tokenizer.decode(ids[start:end])
    assert key in span_text
    # outside the span shouldn't already contain the code (sanity check the
    # span isn't trivially the whole sequence)
    assert end < len(ids)


def test_relay_training_block_handles_gap_larger_than_seq_length():
    tokenizer = _tokenizer()
    ids, (start, end) = build_relay_training_block(tokenizer, seq_length=256, gap=4096, key="111111")
    assert len(ids) == 256
    assert 0 <= start <= end <= 256


def test_make_relay_gap_always_exceeds_window():
    rng = random.Random(0)
    for window in (2048, 1024, 512, 256, 128):
        for _ in range(20):
            gap = make_relay_gap(window, rng)
            assert gap > window
