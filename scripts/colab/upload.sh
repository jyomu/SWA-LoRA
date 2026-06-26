#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
cd /mnt/c/Users/jyomu/Downloads/SWA-LoRA

files=(
  swa_lora/__init__.py
  swa_lora/policy.py
  swa_lora/freeze.py
  swa_lora/losses.py
  swa_lora/lora_setup.py
  swa_lora/trainer.py
  swa_lora/pretrained.py
  swa_lora/toy.py
  swa_lora/eval.py
  swa_lora/adapters/__init__.py
  swa_lora/adapters/base.py
  swa_lora/adapters/qwen3.py
  scripts/train_phase1_qwen3.py
)

for f in "${files[@]}"; do
  echo "uploading $f"
  colab upload -s swa-train "$f" "content/$f"
done
