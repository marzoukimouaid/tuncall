#!/usr/bin/env bash
# Laptop run of the D-15e baselines on test-domain and telephone (D-15g).
# One model at a time, smallest first: batch 4, falling back to 2 then 1 if the GPU runs out of
# memory. Models download to HF_HOME (set in .env, e.g. on an external drive).
# Usage: nohup scripts/overnight_baselines.sh > outputs/phase1/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
# A missing HF_HOME (drive not mounted, or dropped off USB mid-run) must stop the run, not
# download to the internal disk or burn through the batch-size retries.
drive() { [ -n "${HF_HOME:-}" ] && [ -d "$HF_HOME" ] || { echo "HF_HOME '${HF_HOME:-}' not found: mount the drive"; exit 1; }; }
drive

uv sync --locked --extra gpu || exit 1  # CUDA torch (about 4.5 GB the first time)
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'" || exit 1

for key in whisper-small whisper-large-v3-turbo whisper-medium whisper-large-v3 farukstt; do
  for bs in 4 2 1; do
    drive
    echo "=== $(date +%T) $key batch $bs"
    uv run python scripts/benchmark_asr.py --config configs/phase1/baselines.yaml \
      --test-sets test-domain,telephone --models "$key" --batch-size "$bs" && break
  done
done
echo "=== $(date +%T) done"
uv run python scripts/benchmark_asr.py --config configs/phase1/baselines.yaml
