"""Adaptive, success-gated SWA-window curriculum.

Train at the current window; periodically check whether passkey retrieval
succeeds at the relay-required distance (window * --relay-ratio). Only once
it succeeds do we shrink the window (by --window-decay) and continue -- if
it never succeeds within --max-steps-per-window, we stop there, since that
window is the boundary where relay capability breaks down. This replaces an
earlier fixed (window, steps) schedule with one that advances only as fast
as the model actually demonstrates the capability we care about, and with
much finer (continuous) window steps since we're no longer pre-committing a
fixed step budget per window.
"""

import argparse
import json
import random
from pathlib import Path

# `datasets` (pyarrow) must be imported before `torch` -- on Windows, loading
# torch's CUDA/MKL libs first and then pyarrow's bundled Arrow runtime causes
# a DLL conflict that segfaults the process with no traceback.
from datasets import load_dataset

import torch
import wandb

from swa_lora.eval import compute_perplexity, distances_for_window, passkey_retrieval_eval
from swa_lora.lora_setup import apply_lora
from swa_lora.pretrained import build_pretrained_setup
from swa_lora.synthetic import build_relay_training_block, make_relay_gap
from swa_lora.trainer import Trainer, TrainerConfig


def iter_long_documents(dataset_name, dataset_config, split, text_column, min_chars):
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    for example in ds:
        text = example[text_column]
        if len(text.strip()) >= min_chars:
            yield text


def planned_window_schedule(start: int, window_min: int, decay: float) -> list[int]:
    """The window sizes a fully-successful run would pass through -- used only
    to size the teacher baseline eval, since the actual run may stop early."""
    windows = [start]
    while windows[-1] > window_min:
        nxt = max(window_min, round(windows[-1] * decay))
        if nxt == windows[-1]:
            break
        windows.append(nxt)
    return windows


def make_mixed_batches(
    tokenizer,
    doc_iter,
    seq_length,
    num_batches,
    batch_size,
    device,
    window,
    synthetic_ratio=0.0,
    synth_rng: random.Random | None = None,
    build_labels: bool = False,
):
    """Pack `num_batches * batch_size` blocks of `seq_length` tokens. Each
    block is, with probability `synthetic_ratio`, a synthetic relay-training
    document (see swa_lora/synthetic.py) instead of natural text -- this
    makes "passkey survives past the window" training signal dense/guaranteed
    rather than incidental, without touching the loss/training code at all.

    If `build_labels`, also returns a same-shape labels tensor for the CE
    loss (lambda_ce): natural blocks get full next-token labels (the plan's
    6.2 general-language-ability regularizer); synthetic blocks are masked
    to -100 everywhere except the code span, so CE there is focused purely
    on "did the model decode the relay-critical code correctly" rather than
    the (trivial, uninformative) repeated filler tokens.
    """
    target_blocks = num_batches * batch_size
    if synthetic_ratio > 0:
        is_synthetic = [synth_rng.random() < synthetic_ratio for _ in range(target_blocks)]
    else:
        is_synthetic = [False] * target_blocks
    num_natural = is_synthetic.count(False)

    buffer: list[int] = []
    natural_blocks: list[list[int]] = []
    for text in doc_iter:
        buffer.extend(tokenizer(text)["input_ids"])
        while len(buffer) >= seq_length and len(natural_blocks) < num_natural:
            natural_blocks.append(buffer[:seq_length])
            buffer = buffer[seq_length:]
        if len(natural_blocks) >= num_natural:
            break

    blocks = []
    label_blocks = [] if build_labels else None
    nat_idx = 0
    for synth in is_synthetic:
        if synth:
            key = str(synth_rng.randint(100000, 999999))
            gap = make_relay_gap(window, synth_rng)
            ids, (start, end) = build_relay_training_block(tokenizer, seq_length, gap, key)
            blocks.append(ids)
            if build_labels:
                label = [-100] * seq_length
                label[start:end] = ids[start:end]
                label_blocks.append(label)
        else:
            ids = natural_blocks[nat_idx]
            nat_idx += 1
            blocks.append(ids)
            if build_labels:
                label_blocks.append(list(ids))

    input_ids = torch.tensor(blocks, dtype=torch.long).view(num_batches, batch_size, seq_length).to(device)
    if build_labels:
        labels = torch.tensor(label_blocks, dtype=torch.long).view(num_batches, batch_size, seq_length).to(device)
        return input_ids, labels
    return input_ids, None


def next_training_batch(tokenizer, doc_iter, seq_length, batch_size, device, window, synthetic_ratio, synth_rng, build_labels):
    batch, labels = make_mixed_batches(
        tokenizer,
        doc_iter,
        seq_length,
        num_batches=1,
        batch_size=batch_size,
        device=device,
        window=window,
        synthetic_ratio=synthetic_ratio,
        synth_rng=synth_rng,
        build_labels=build_labels,
    )
    return batch[0], (labels[0] if labels is not None else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-full-top-layers", type=int, default=1)
    parser.add_argument("--window-start", type=int, default=2048)
    parser.add_argument("--window-min", type=int, default=64)
    parser.add_argument("--window-decay", type=float, default=0.85, help="window *= this once the current window succeeds")
    parser.add_argument(
        "--relay-ratio",
        type=float,
        default=2.0,
        help="success is measured at distance = window * relay_ratio (must be > 1, i.e. genuinely relay-required)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.6,
        help="passkey accuracy at the relay-required distance needed to advance to the next (smaller) window",
    )
    parser.add_argument("--eval-every", type=int, default=50, help="check for success every N steps within a window")
    parser.add_argument(
        "--max-steps-per-window",
        type=int,
        default=400,
        help="give up on this window (and stop the whole curriculum -- this is the boundary) if no success within this many steps",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--include-mlp-lora", action="store_true")
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    parser.add_argument(
        "--lambda-local",
        type=float,
        default=0.0,
        help="plan section 6.3 per-layer local cutoff loss weight; 0 disables",
    )
    parser.add_argument(
        "--lambda-ce",
        type=float,
        default=0.0,
        help="frozen-LM-head cross-entropy loss weight (plan 6.2); on synthetic blocks this is masked to "
        "just the code span, on natural blocks it's the full sequence; 0 disables",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--dataset-name", default="sedthh/gutenberg_english")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="TEXT")
    parser.add_argument("--min-doc-chars", type=int, default=5000)
    parser.add_argument("--num-eval-docs", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="checkpoints/phase1_curriculum")
    parser.add_argument("--metrics-out", default="metrics_curriculum.json")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.0,
        help="fraction of training blocks replaced with synthetic relay-training docs (distance > current window); 0 disables",
    )
    parser.add_argument("--synthetic-seed", type=int, default=12345)
    parser.add_argument("--wandb-project", default="swa-lora")
    parser.add_argument("--wandb-entity", default="jyomu-none")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args)
        )

    history = {"config": vars(args), "stages": []}

    def dump_history():
        with open(args.metrics_out, "w") as f:
            json.dump(history, f, indent=2)

    setup = build_pretrained_setup(
        model_name=args.model_name,
        num_full_top_layers=args.num_full_top_layers,
        sliding_window=args.window_start,
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

    # Evaluate the teacher once at every distance a fully-successful run could
    # reach, so each checkpoint's passkey result has a directly comparable
    # reference even though the real run may stop earlier.
    planned_windows = planned_window_schedule(args.window_start, args.window_min, args.window_decay)
    all_distances = sorted({d for window in planned_windows for d in distances_for_window(window)})

    ppl_teacher = compute_perplexity(
        setup.teacher, setup.tokenizer, eval_texts, args.device, max_length=args.sequence_length
    )
    passkey_teacher = passkey_retrieval_eval(
        setup.teacher, setup.tokenizer, args.device, distances=all_distances, num_samples=5
    )
    history["teacher_baseline"] = {
        "ppl": ppl_teacher,
        "passkey": {str(k): v for k, v in passkey_teacher.items()},
    }
    print(f"A full teacher ppl={ppl_teacher:.3f} passkey={history['teacher_baseline']['passkey']}")
    dump_history()
    if use_wandb:
        wandb.log(
            {
                "teacher/ppl": ppl_teacher,
                **{f"teacher/passkey_dist_{k}": v for k, v in passkey_teacher.items()},
            },
            step=0,
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
            lambda_local=args.lambda_local,
            lambda_ce=args.lambda_ce,
            amp_dtype=torch.bfloat16,
        ),
        swa_layer_indices=setup.policy.swa_layer_indices,
    )

    out_dir = Path(args.output_dir)
    global_step = 0
    synth_rng = random.Random(args.synthetic_seed)
    build_labels = args.lambda_ce > 0
    window = args.window_start

    while True:
        print(f"=== Window={window} (starting at global_step={global_step}) ===")
        setup.adapter.set_sliding_window(student, window)
        relay_distance = round(window * args.relay_ratio)
        step_in_window = 0
        stage_loss = []
        checks = []
        outcome = None

        while True:
            batch, label_batch = next_training_batch(
                setup.tokenizer,
                doc_iter,
                args.sequence_length,
                args.batch_size,
                args.device,
                window,
                args.synthetic_ratio,
                synth_rng,
                build_labels,
            )
            metrics = trainer.train_step(input_ids=batch, labels=label_batch)
            scalars = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
            stage_loss.append({"global_step": global_step, **scalars})
            global_step += 1
            step_in_window += 1
            if use_wandb:
                wandb.log({"train/window": window, **{f"train/{k}": v for k, v in scalars.items()}}, step=global_step)
            if global_step % args.log_every == 0:
                print(global_step, scalars)

            if step_in_window % args.eval_every == 0:
                ppl_now = compute_perplexity(
                    student, setup.tokenizer, eval_texts, args.device, max_length=args.sequence_length
                )
                check_distances = distances_for_window(window)
                passkey_now = passkey_retrieval_eval(
                    student, setup.tokenizer, args.device, distances=check_distances, num_samples=5
                )
                success_acc = passkey_now.get(relay_distance)
                check = {
                    "global_step": global_step,
                    "step_in_window": step_in_window,
                    "ppl": ppl_now,
                    "passkey": {str(k): v for k, v in passkey_now.items()},
                    "relay_distance": relay_distance,
                    "relay_accuracy": success_acc,
                }
                checks.append(check)
                print(f"  [check] window={window} step_in_window={step_in_window} ppl={ppl_now:.3f} "
                      f"relay@{relay_distance}={success_acc} passkey={check['passkey']}")
                if use_wandb:
                    wandb.log(
                        {
                            "check/window": window,
                            "check/ppl": ppl_now,
                            "check/relay_accuracy": success_acc,
                            **{f"check/passkey_dist_{d}": v for d, v in passkey_now.items()},
                        },
                        step=global_step,
                    )

                if success_acc is not None and success_acc >= args.success_threshold:
                    outcome = "advanced"
                    break
                if step_in_window >= args.max_steps_per_window:
                    outcome = "gave_up"
                    break

        stage_distances = distances_for_window(window)
        stage_result = {
            "window": window,
            "global_step": global_step,
            "outcome": outcome,
            "steps_taken": step_in_window,
            "relay_distance": relay_distance,
            "checks": checks,
            "passkey_teacher_ref": {str(d): history["teacher_baseline"]["passkey"].get(str(d)) for d in stage_distances},
            "train_loss": stage_loss,
        }
        history["stages"].append(stage_result)
        print(f"  window={window} outcome={outcome} steps_taken={step_in_window}")
        if use_wandb:
            wandb.log({"stage/window": window, "stage/outcome": outcome, "stage/steps_taken": step_in_window}, step=global_step)

        trainer.save_checkpoint(out_dir / f"window_{window}")
        dump_history()

        if outcome == "gave_up":
            print(f"Stopping: window={window} never reached success_threshold within max_steps_per_window. "
                  f"This is the relay-capability boundary.")
            break

        next_window = max(args.window_min, round(window * args.window_decay))
        if next_window == window:
            print("Stopping: window already at window_min.")
            break
        window = next_window

    print(f"Curriculum complete. Metrics in {args.metrics_out}, checkpoints under {out_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
