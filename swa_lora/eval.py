import math
import random

import torch

PASSKEY_PREFIX = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about the important information there. "
)
PASSKEY_INFO = "The pass key is {key}. Remember it. {key} is the pass key. "
PASSKEY_SUFFIX = "What is the pass key? The pass key is"
FILLER_UNIT = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)


def _build_passkey_text(tokenizer, distance: int, key: str, leading_filler_reps: int = 3) -> str:
    """`distance` is the token gap between the end of the passkey sentence and
    the point where generation starts (i.e. roughly how far back the model
    must reach to answer). This is what actually determines whether a single
    attention window can reach the passkey directly -- NOT the passkey's
    position relative to the total prompt length."""
    prefix = PASSKEY_PREFIX
    info = PASSKEY_INFO.format(key=key)
    suffix = " " + PASSKEY_SUFFIX

    suffix_len = len(tokenizer(suffix)["input_ids"])
    filler_unit_len = len(tokenizer(FILLER_UNIT)["input_ids"])

    after_tokens = max(distance - suffix_len, 0)
    after_reps = max(round(after_tokens / filler_unit_len), 0)

    return prefix + FILLER_UNIT * leading_filler_reps + info + FILLER_UNIT * after_reps + suffix


@torch.no_grad()
def passkey_retrieval_eval(
    model,
    tokenizer,
    device,
    distances: list[int],
    num_samples: int = 5,
    max_new_tokens: int = 6,
    seed: int = 0,
) -> dict[int, float]:
    """Needle-in-a-haystack passkey retrieval accuracy, keyed by the token
    distance between the passkey and the query.

    Distance is the quantity that actually matters for a windowed model: if
    distance <= window, a single attention hop reaches the passkey directly
    at every layer, so success there proves nothing about multi-layer relay.
    Pick distances that exceed the window under test (e.g. 2x, 4x) to test
    the capability the architecture is actually supposed to provide.
    """
    rng = random.Random(seed)
    was_training = model.training
    model.eval()
    results: dict[int, float] = {}
    for distance in distances:
        correct = 0
        for _ in range(num_samples):
            key = str(rng.randint(10000, 99999))
            text = _build_passkey_text(tokenizer, distance, key)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(out_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            if key in gen:
                correct += 1
        results[distance] = correct / num_samples
    if was_training:
        model.train()
    return results


def distances_for_window(
    window: int, ratios: list[float] = (0.5, 1.0, 2.0, 4.0), max_distance: int = 4096
) -> list[int]:
    """Distances split into trivially-reachable (ratio<=1) and relay-required
    (ratio>1, no single attention hop spans the gap) regimes for a given
    window. Capped at max_distance so a wide window doesn't blow up eval
    prompt length far past what was ever trained on."""
    return sorted({min(max(int(window * r), 1), max_distance) for r in ratios})


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list[str], device, max_length: int = 2048) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids=input_ids, labels=input_ids)
        losses.append(out.loss.item())
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)
