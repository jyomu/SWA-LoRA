import sys
import time

sys.path.insert(0, ".")
sys.argv = [
    "train_phase1_qwen3.py",
    "--sequence-length", "2048",
    "--sliding-window", "256",
    "--num-full-top-layers", "1",
    "--max-steps", "1000",
    "--grad-accum-steps", "4",
    "--log-every", "20",
    "--eval-every", "100",
    "--checkpoint-every", "100",
    "--num-eval-docs", "8",
    "--gradient-checkpointing",
    "--metrics-out", "metrics_w256_g1_long.json",
    "--output-dir", "checkpoints/phase1_w256_g1_long",
]

t0 = time.time()
exec(open("scripts/train_phase1_qwen3.py").read())
print("TOTAL_WALLTIME_SEC", time.time() - t0)
