# 0002. Experiment tracking: MLflow hosted on DagsHub

- **Status:** accepted (supersedes self-hosted MLflow, D-80 / D-80a / D-80b)
- **Date:** 2026-11-20
- **Decision log ID:** D-80v2 (with D-80c, D-80d)

## Context

Runs happen on the laptop and sometimes on Colab/Kaggle. The home connection uploads at about 0.68 Mbps with 450–1,000 ms latency (N-01). Every run must be recorded in one place with its commit, config, data version and machine.

## Options

1. **Self-hosted MLflow + Postgres + artifact store on the laptop:** shows infrastructure skill. No off-site backup, Colab runs need a separate import step, and it's more to maintain.
2. **MLflow on DagsHub:** the standard MLflow client logs from every machine to one hosted server. Off-site backup, public experiment links, nothing to run. Free-tier limits could change.
3. **Weights & Biases:** polished UI, but a proprietary client and lock-in.

## Decision

Option 2. Logging uses the standard MLflow client, so moving to self-hosted MLflow later only changes the tracking URI.

## Consequences

- Only small artifacts (configs, plots, eval reports, LoRA adapters ≤ ~50 MB) are uploaded. Checkpoints and data stay local (D-80c).
- Logging is async with a local buffer, so the slow connection never blocks or crashes training (D-80d).
- No tracking containers on the laptop. Docker is used later for the serving stack.
