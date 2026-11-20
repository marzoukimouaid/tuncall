# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

`PROJECT.md` is the authoritative spec: vision, architecture (§3), the `CallRecord` contract (§4), phases (§5), the dev/MLOps workflow (§5b), layout (§6), the open-questions backlog (§8), working agreements (§8b) and the Decision Log (§9). Read the relevant sections before starting non-trivial work. If code and PROJECT.md disagree, update one of them so they match.

## Working agreements (binding)

- **Ask, don't assume.** Any choice not already in the §9 Decision Log (a library, tool, file layout, threshold, label scheme) must be asked about before it is implemented. Open questions are listed in §8 by ID (D-xx, B-xx, P-xx).
- **Log decisions.** When a decision is made, add a row to §9 (`ID | Date | Decision | Reason | Alternatives rejected`). Superseded entries are struck through, never deleted. Major decisions also get an ADR in `docs/adr/NNNN-title.md` (template: `docs/adr/0000-template.md`).
- **Tools do tool work (W-01).** Never format or fix lint by hand. Run `uv run ruff check --fix` and `uv run ruff format`, then fix only what Ruff can't.
- **Numbers need scripts.** Every metric quoted anywhere must be reproducible from a committed script and config.
- **Frozen test sets.** `test-domain` and `test-general` are never used for training or tuning; their file hashes are recorded in the repo.

## Commands

The project uses uv (Python 3.12, `.venv` in the repo). The uv binary is in `~/.local/bin`.

```bash
uv sync                                   # install / update the venv from uv.lock
uv add <pkg>                              # runtime dependency
uv add --dev <pkg>                        # dev dependency
uv run ruff check --fix && uv run ruff format
uv run pyright
uv run pytest                             # all tests
uv run pytest tests/test_import.py::test_package_imports   # a single test
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`, job name `checks`) runs `uv sync --locked`, `ruff check`, `ruff format --check`, `pyright` and `pytest`. A stale `uv.lock` fails CI, so commit lockfile changes. A CPU smoke test (tiny train + eval) is planned for Phase 1 (D-83).

Pre-commit runs Ruff and pyright through `uv run` (versions come from `uv.lock`) and blocks files over 500 KB.

## Git and GitHub workflow

- `main` is protected: it requires a PR and a passing `checks` job, the branch must be up to date, and the rules apply to admins too. Never push to `main`; work on short-lived branches.
- Squash merge only. The PR title becomes the commit message, so PR titles and commits follow Conventional Commits (`feat(asr): …`, `fix(eval): …`, `ci: …`, `docs(adr): …`).
- Work is tracked in GitHub Issues: labels `area:*` / `type:*` / `prio:*`, one milestone per phase ("Phase N – …"), and the Projects board at users/marzoukimouaid/projects/1. PRs close issues with `Closes #N`.
- `astral-sh/setup-uv` publishes no floating major tags, so pin it to an exact release.

## Architecture (planned; most of it is not built yet)

Pipeline: ingestion → speech layer (ASR + alignment + diarization) → language layer (normalization, intent/sentiment/NER, summary) → knowledge layer (search, RAG, analytics) → API/UI. One `CallRecord` object flows through every stage, and each output records the model version that produced it.

Code is a single package `src/tuncall/`. Add a subpackage per component (`data/`, `asr/`, `eval/`, …) only when its first code arrives; don't create empty placeholders. Planned shared modules:
- `config.py`: plain YAML configs validated with Pydantic. Unknown keys and wrong types are errors (D-87). The full resolved config is logged with every run.
- `tracking.py`: MLflow on DagsHub, logging async and batched with a disk buffer. A logging failure must never crash or slow down training (D-80d).

Run configs go in `configs/`, entry-point scripts in `scripts/`.

## Constraints that shape code

- **Compute:** the laptop RTX 4070 (8 GB VRAM) is the main machine. Colab and Kaggle are overflow only. Training code must run unchanged in all three places (only config and environment differ) and must checkpoint and resume cleanly.
- **Network:** upload speed is about 0.68 Mbps. Nothing over ~50 MB is uploaded from home. Full checkpoints, datasets and base models stay local or go to the HF Hub through the cloud. Only small artifacts (configs, plots, eval reports, LoRA adapters) go to MLflow.
- **Run metadata:** every run records its git commit, config, dataset version and machine.
- **Transcription convention (D-10):** everything is in Arabic script, including French/English words. Report CER next to WER, and normalize Arabic orthography before scoring.
- **Secrets:** read tokens only from environment variables or the git-ignored `.env` (names are in `.env.example`).
- **Git contents:** git holds code and configs only. Data, checkpoints, audio and weights are git-ignored.
