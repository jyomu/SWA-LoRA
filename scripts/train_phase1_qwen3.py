"""Phase 1 (plan section 17, week-1 minimum experiment):
Qwen/Qwen3-0.6B-Base, lower layers -> SWA, top layer(s) -> full attention,
LoRA on the SWA layers, final-hidden-state distillation from the frozen
full-attention teacher. Compares: full teacher / hybrid no-train / hybrid+LoRA.
"""

import argparse
import json
from pathlib import Path

# `datasets` (pyarrow) must be imported before `torch` -- on Windows, loading
# torch's CUDA/MKL libs first and then pyarrow's bundled Arrow runtime causes
# a DLL conflict that segfaults the process with no traceback.
from datasets import load_dataset

import torch

from swa_lora.eval import compute_perplexity, distances_for_window, passkey_retrieval_eval
from swa_lora.lora_setup import apply_lora
from swa_lora.pretrained import build_pretrained_setup
from swa_lora.trainer import Trainer, TrainerConfig


def iter_long_documents(dataset_name, dataset_config, split, text_column, min_chars):
    # streaming=True: these long-document datasets (e.g. full Gutenberg books)
    # are tens of GB; we only ever need a few thousand tokens' worth.
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    for example in ds:
        text = example[text_column]
        if len(text.strip()) >= min_chars:
            yield text


def make_packed_batches(tokenizer, doc_iter, seq_length, num_batches, batch_size, device):
    buffer: list[int] = []
    blocks: list[list[int]] = []
    target_blocks = num_batches * batch_size
    for text in doc_iter:
        buffer.extend(tokenizer(text)["input_ids"])
        while len(buffer) >= seq_length and len(blocks) < target_blocks:
            blocks.append(buffer[:seq_length])
            buffer = buffer[seq_length:]
        if len(blocks) >= target_blocks:
            break
    blocks = blocks[:target_blocks]
    input_ids = torch.tensor(blocks, dtype=torch.long)
    return input_ids.view(num_batches, batch_size, seq_length).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--num-full-top-layers", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--include-mlp-lora", action="store_true")
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0)
    parser.add_argument(
        "--lambda-local",
        type=float,
        default=0.0,
        help="plan section 6.3 per-layer local cutoff loss weight; 0 disables",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    # Full Project Gutenberg books -- long enough for real long-range
    # dependencies (plan section 11). "wikitext"/"pg19" on the Hub are
    # script-based datasets and unsupported by current `datasets` versions.
    parser.add_argument("--dataset-name", default="sedthh/gutenberg_english")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="TEXT")
    parser.add_argument("--min-doc-chars", type=int, default=5000)
    parser.add_argument("--num-eval-docs", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables periodic eval during training")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="0 disables periodic checkpointing")
    parser.add_argument("--metrics-out", default="metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="checkpoints/phase1")
    parser.add_argument("--skip-passkey-eval", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    history = {"config": vars(args), "train_loss": [], "periodic_eval": []}

    def dump_history():
        with open(args.metrics_out, "w") as f:
            json.dump(history, f, indent=2)

    setup = build_pretrained_setup(
        model_name=args.model_name,
        num_full_top_layers=args.num_full_top_layers,
        sliding_window=args.sliding_window,
        device=args.device,
    )
    print("layer_types:", setup.policy.layer_types)

    if args.gradient_checkpointing:
        setup.student.gradient_checkpointing_enable()
        setup.student.enable_input_require_grads()

    print("Loading eval texts...")
    doc_iter = iter_long_documents(
        args.dataset_name, args.dataset_config, args.dataset_split, args.text_column, args.min_doc_chars
    )
    eval_texts = [next(doc_iter) for _ in range(args.num_eval_docs)]

    print("Condition A (full teacher) / B (hybrid, no training) perplexity:")
    ppl_teacher = compute_perplexity(
        setup.teacher, setup.tokenizer, eval_texts, args.device, max_length=args.sequence_length
    )
    ppl_hybrid_pre = compute_perplexity(
        setup.student, setup.tokenizer, eval_texts, args.device, max_length=args.sequence_length
    )
    print(f"  A full teacher        ppl={ppl_teacher:.3f}")
    print(f"  B hybrid, no training ppl={ppl_hybrid_pre:.3f}")

    # Distances relative to the SWA window: ratios <=1 are trivially reachable
    # by a single attention hop at every layer (a 0% there would mean the eval
    # itself is broken, not that retrieval failed); ratios >1 require genuine
    # multi-layer relay since no single layer's window spans the gap.
    passkey_distances = distances_for_window(args.sliding_window)

    passkey_pre = None
    passkey_teacher = None
    if not args.skip_passkey_eval:
        # Run the teacher too -- without this control, a 0% score is
        # ambiguous between "SWA can't retrieve at this distance" and "this
        # small base model can't follow the passkey instruction at all".
        passkey_teacher = passkey_retrieval_eval(
            setup.teacher, setup.tokenizer, args.device, distances=passkey_distances, num_samples=5
        )
        print("  A passkey retrieval (full teacher):", passkey_teacher)
        passkey_pre = passkey_retrieval_eval(
            setup.student, setup.tokenizer, args.device, distances=passkey_distances, num_samples=5
        )
        print("  B passkey retrieval (no training):", passkey_pre)

    history["baseline"] = {
        "A_full_teacher_ppl": ppl_teacher,
        "A_full_teacher_passkey": {str(k): v for k, v in (passkey_teacher or {}).items()},
        "B_hybrid_no_train_ppl": ppl_hybrid_pre,
        "B_hybrid_no_train_passkey": {str(k): v for k, v in (passkey_pre or {}).items()},
    }
    dump_history()

    print("Building train batches...")
    train_batches = make_packed_batches(
        setup.tokenizer,
        doc_iter,  # continues past the eval docs pulled above -- no overlap
        args.sequence_length,
        args.max_steps,
        args.batch_size,
        args.device,
    )

    student = apply_lora(
        setup.student,
        setup.adapter,
        setup.policy,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        include_mlp=args.include_mlp_lora,
    )
    student.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    trainer = Trainer(
        setup.teacher,
        student,
        setup.adapter,
        optimizer,
        TrainerConfig(
            grad_accum_steps=args.grad_accum_steps,
            lambda_cos=args.lambda_cos,
            lambda_ce=args.lambda_ce,
            lambda_local=args.lambda_local,
            amp_dtype=torch.bfloat16,
        ),
        swa_layer_indices=setup.policy.swa_layer_indices,
    )

    out_dir = Path(args.output_dir)

    def run_eval_snapshot(step):
        ppl_now = compute_perplexity(
            student, setup.tokenizer, eval_texts, args.device, max_length=args.sequence_length
        )
        passkey_now = None
        if not args.skip_passkey_eval:
            passkey_now = passkey_retrieval_eval(
                student, setup.tokenizer, args.device, distances=passkey_distances, num_samples=5
            )
        entry = {"step": step, "ppl": ppl_now, "passkey": {str(k): v for k, v in (passkey_now or {}).items()}}
        print(f"  [eval@{step}] ppl={ppl_now:.3f} passkey={entry['passkey']}")
        return entry

    print(f"Training for {args.max_steps} steps...")
    for step, batch in enumerate(train_batches):
        labels = batch if args.lambda_ce > 0 else None
        metrics = trainer.train_step(input_ids=batch, labels=labels)
        scalars = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
        history["train_loss"].append({"step": step, **scalars})
        if step % args.log_every == 0:
            print(step, scalars)

        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            trainer.save_checkpoint(out_dir)
            dump_history()

        if args.eval_every and (step + 1) % args.eval_every == 0:
            history["periodic_eval"].append(run_eval_snapshot(step + 1))
            dump_history()

    print("Condition C (hybrid + LoRA, hidden loss) final eval:")
    history["final"] = run_eval_snapshot(args.max_steps)

    trainer.save_checkpoint(out_dir)
    dump_history()
    print(f"Saved LoRA checkpoint to {out_dir}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
