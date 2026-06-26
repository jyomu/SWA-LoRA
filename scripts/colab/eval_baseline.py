import json
import sys

sys.path.insert(0, ".")

from datasets import load_dataset  # noqa: E402  (must precede torch import on Windows; harmless here too)
import torch  # noqa: E402

from swa_lora.eval import compute_perplexity, distances_for_window, passkey_retrieval_eval  # noqa: E402
from swa_lora.pretrained import build_pretrained_setup  # noqa: E402

SEQUENCE_LENGTH = 2048
NUM_EVAL_DOCS = 8
SLIDING_WINDOW = 256

setup = build_pretrained_setup(
    model_name="Qwen/Qwen3-0.6B-Base",
    num_full_top_layers=1,
    sliding_window=SLIDING_WINDOW,
    device="cuda",
)
# Distances >1x window require genuine multi-layer relay; <=1x window is
# trivially reachable by a single attention hop and mostly a sanity check.
PASSKEY_DISTANCES = distances_for_window(SLIDING_WINDOW)

ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
eval_texts = []
for example in ds:
    text = example["TEXT"]
    if len(text.strip()) >= 5000:
        eval_texts.append(text)
    if len(eval_texts) >= NUM_EVAL_DOCS:
        break

ppl_teacher = compute_perplexity(setup.teacher, setup.tokenizer, eval_texts, "cuda", max_length=SEQUENCE_LENGTH)
ppl_hybrid_pre = compute_perplexity(setup.student, setup.tokenizer, eval_texts, "cuda", max_length=SEQUENCE_LENGTH)

passkey_pre = passkey_retrieval_eval(
    setup.student, setup.tokenizer, "cuda", distances=PASSKEY_DISTANCES, num_samples=5
)
# Run on the teacher too -- without this control, a 0% score is ambiguous
# between "SWA can't retrieve at this distance" and "this small base model
# can't follow the passkey instruction at all".
passkey_teacher = passkey_retrieval_eval(
    setup.teacher, setup.tokenizer, "cuda", distances=PASSKEY_DISTANCES, num_samples=5
)

results = {
    "A_full_teacher_ppl": ppl_teacher,
    "A_full_teacher_passkey": {str(k): v for k, v in passkey_teacher.items()},
    "B_hybrid_no_train_ppl": ppl_hybrid_pre,
    "B_hybrid_no_train_passkey": {str(k): v for k, v in passkey_pre.items()},
}

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("RESULTS_JSON", json.dumps(results))
