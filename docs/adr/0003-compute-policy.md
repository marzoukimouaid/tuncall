# 0003. Compute policy: laptop first, cloud as overflow

- **Status:** accepted
- **Date:** 2026-11-20
- **Decision log ID:** B-01c (refines B-01, B-01b)

## Context

Available compute: an RTX 4070 Laptop GPU (8 GB VRAM) and free Colab (T4, 16 GB) and Kaggle (about 30 GPU h/week). No paid GPUs. The home upload speed is very slow (N-01).

## Options

1. **Cloud-first:** Colab/Kaggle for all training. More memory, but sessions time out, GPUs aren't guaranteed, and every dataset must be moved there.
2. **Laptop-first, cloud as overflow:** the laptop is faster to iterate on and always available. The cloud covers only what doesn't fit.
3. **Laptop only:** simplest, but some jobs can't fit in 8 GB.

## Decision

Option 2. All training and evaluation run on the laptop. Colab/Kaggle are used only for (a) jobs needing more than 8 GB of GPU memory and (b) jobs needing data-center bandwidth (processing large public datasets, merging and publishing full models).

## Consequences

- Training code runs unchanged on the laptop, Colab and Kaggle; only config and environment differ.
- Scripts checkpoint often and resume cleanly, because cloud sessions time out.
- Model size choices are benchmarked against the 8 GB limit (see 0004).
