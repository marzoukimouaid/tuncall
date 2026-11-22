#!/usr/bin/env bash
# Laptop run of the D-15e baselines on test-domain and telephone (D-15g).
# One model at a time, smallest first: batch 4, falling back to 2 then 1 if the GPU runs out of
# memory. Each model's weights are deleted after it finishes (disk space), except turbo, the
# Phase 3 base model. Usage: nohup scripts/overnight_baselines.sh > outputs/phase1/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

uv sync --locked --extra gpu || exit 1  # CUDA torch (about 4.5 GB the first time)
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'" || exit 1

for pair in whisper-small:openai/whisper-small whisper-large-v3-turbo:openai/whisper-large-v3-turbo \
            whisper-medium:openai/whisper-medium whisper-large-v3:openai/whisper-large-v3 \
            farukstt:medyas/FarukSTT; do
  key="${pair%%:*}" repo="${pair#*:}"
  for bs in 4 2 1; do
    echo "=== $(date +%T) $key batch $bs"
    uv run python scripts/benchmark_asr.py --config configs/phase1/baselines.yaml \
      --test-sets test-domain,telephone --models "$key" --batch-size "$bs" && break
  done
  [ "$key" != whisper-large-v3-turbo ] && rm -rf "$HUB/models--${repo//\//--}"
  df -h . | tail -1
done
echo "=== $(date +%T) done"
uv run python scripts/benchmark_asr.py --config configs/phase1/baselines.yaml
