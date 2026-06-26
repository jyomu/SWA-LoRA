import random

# Deliberately a *different* template from swa_lora/eval.py's passkey task --
# training on this exact wording must not leak into what the held-out
# passkey eval is measuring, otherwise the eval stops testing generalization
# and starts testing memorization of the training template.
SYNTH_PREFIX = "Listen closely, this is a memory exercise. "
SYNTH_INFO = "The secret code is {key}. Hold onto the secret code {key}. "
SYNTH_FILLER = "Clouds drift overhead. Rivers flow to the sea. Mountains stand still. Time keeps moving forward. "


def build_relay_training_block(tokenizer, seq_length: int, gap: int, key: str) -> tuple[list[int], tuple[int, int]]:
    """A synthetic document for relay training: a secret code near the start,
    then `gap` tokens of filler before the block ends. The hidden states for
    that trailing span can only match the (full-attention) teacher's if
    information about the code survived the SWA window -- this is the same
    underlying signal `hidden_state_loss` already optimizes on natural text,
    just made dense/guaranteed instead of incidental.

    Returns the token ids plus `(start, end)`, the half-open span of the
    *first* occurrence of the code within those ids -- callers wanting a
    CE loss focused on "did the model decode the code correctly" (rather
    than the trivial, already-near-perfect filler tokens) should mask
    labels to -100 outside this span.
    """
    prefix_ids = tokenizer(SYNTH_PREFIX)["input_ids"]
    info_ids = tokenizer(SYNTH_INFO.format(key=key))["input_ids"]
    filler_ids = tokenizer(SYNTH_FILLER)["input_ids"]

    code_start = len(prefix_ids) + len(filler_ids)
    code_end = code_start + len(info_ids)

    after_reps = max(round(gap / len(filler_ids)), 0)
    ids = prefix_ids + filler_ids + info_ids + filler_ids * after_reps
    if len(ids) < seq_length:
        ids = ids + filler_ids * ((seq_length - len(ids)) // len(filler_ids) + 1)
    ids = ids[:seq_length]
    return ids, (min(code_start, seq_length), min(code_end, seq_length))


def make_relay_gap(window: int, rng: random.Random, ratios: tuple[float, ...] = (1.5, 2.0, 3.0)) -> int:
    """Pick a gap that exceeds the current window -- forces genuine relay,
    mirroring `distances_for_window`'s ratio>1 (relay-required) regime but
    for training instead of eval."""
    return int(window * rng.choice(ratios))
