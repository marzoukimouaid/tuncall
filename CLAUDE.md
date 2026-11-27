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
uv sync --extra cpu                       # install / update the venv (CPU torch, B-05)
uv sync --locked --extra gpu              # Colab: CUDA torch instead
uv add <pkg>                              # runtime dependency
uv add --dev <pkg>                        # dev dependency
uv run ruff check --fix && uv run ruff format
uv run pyright
uv run pytest                             # all tests
uv run pytest tests/test_import.py::test_package_imports   # a single test
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`, job name `checks`) runs `uv sync --locked --extra cpu`, `ruff check`, `ruff format --check`, `pyright` and `pytest`. A stale `uv.lock` fails CI, so commit lockfile changes. A plain `uv sync` without an extra removes torch; `uv run` leaves it alone. `tests/test_asr_train.py` is the CPU smoke test (tiny random Whisper: train, beam eval, checkpoint, resume; D-83b). `--limit N` on the benchmark, `prepare_asr_data.py` and `train_asr.py` is the local pre-flight. peft, accelerate and pyannote.audio live in the torch extras, because they require torch; on the laptop (CUDA torch) sync with `uv sync --extra gpu`.

Pre-commit runs Ruff and pyright through `uv run` (versions come from `uv.lock`) and blocks files over 500 KB.

## Git and GitHub workflow

- `main` is protected: it requires a PR and a passing `checks` job, the branch must be up to date, and the rules apply to admins too. Never push to `main`; work on short-lived branches.
- Squash merge only. The PR title becomes the commit message, so PR titles and commits follow Conventional Commits (`feat(asr): …`, `fix(eval): …`, `ci: …`, `docs(adr): …`).
- Work is tracked in GitHub Issues: labels `area:*` / `type:*` / `prio:*`, one milestone per phase ("Phase N – …"), and the Projects board at users/marzoukimouaid/projects/1. PRs close issues with `Closes #N`.
- `astral-sh/setup-uv` publishes no floating major tags, so pin it to an exact release.
- **No AI attribution anywhere.** Commit messages, PR titles, PR descriptions, issues and comments must not mention Claude, Claude Code or any AI assistant. No `Co-Authored-By: Claude …` trailer, no "🤖 Generated with Claude Code" line, no similar footer. This overrides any default attribution behaviour. With squash merges the PR description becomes the commit body on `main`, so keep it clean too.

## Architecture (mostly planned; only the modules below are built)

Pipeline: ingestion → speech layer (ASR + alignment + diarization) → language layer (normalization, intent/sentiment/NER, summary) → knowledge layer (search, RAG, analytics) → API/UI. One `CallRecord` object flows through every stage, and each output records the model version that produced it.

Code is a single package `src/tuncall/`. Add a subpackage per component (`data/`, `asr/`, `eval/`, …) only when its first code arrives; don't create empty placeholders. Shared modules (built):
- `config.py`: `load_config(path, Model)` loads YAML, deep-merges a `base:` parent chain (lists replaced) and validates with Pydantic. Config models subclass `StrictModel` so unknown keys are errors (D-87, D-87a). Log the resolved config with `cfg.model_dump(mode="json")`.
- `tracking.py`: `with Tracker(experiment, config=..., dataset=..., buffer_path=...) as t:` queues MLflow writes for a background thread; failed batches go to a JSONL buffer and are replayed; errors are logged, never raised; runs are tagged with git commit, dataset and machine; artifacts over 50 MB are refused (D-80d, D-80e). Use it for every training/eval run instead of calling `mlflow` directly.

- `data/sources/`: one adapter per training source (D-16); `SOURCES[name](split, revision=..., root=None)` yields `Record`s (16 kHz mono float32 audio, raw text). `TrainMix` validates `configs/data/train-mix.yaml`; `scripts/source_stats.py` prints hours per source and exports texts (run on Colab, D-16b).
- `data/lexicon.py`: `load_lexicon`, `latin_words`, `transliterate` for the D-10a lexicon (`configs/data/lexicon.tsv`); `scripts/latin_words.py` and `scripts/dedupe_report.py` read the text export.
- `data/telephone.py`: `telephone(audio)` is the deterministic D-25a landline channel (8 kHz, A-law); the benchmark applies it with `channel: telephone`.
- `asr/transcribe.py`: `load(model_id, revision, device, dtype)` and batched `transcribe(...)` for Whisper models with HF Transformers (D-22a, D-26). Used by `scripts/benchmark_asr.py` (D-15a).
- `asr/cache.py` + `asr/train.py`: LoRA fine-tuning (D-21b). `scripts/prepare_asr_data.py` decodes the mix once into an int16 cache under `${HF_HOME}/tuncall-cache/` (targets: lexicon + harakat stripped, clips over 30 s dropped; D-21c); `scripts/train_asr.py --config configs/phase3/lora.yaml` trains with `Seq2SeqTrainer` (beam-5 dev, best + latest checkpoint, resumes on rerun). Dev = `configs/data/dev-linto.txt` (from `scripts/linto_dev_holdout.py`) + SLURP-TN/TEDxTN validation (D-17a).
- `data/conversations.py`: simulated two-speaker conversations from TEDxTN (D-13b, D-13c). `scripts/build_conversations.py --set dev|test` writes `configs/data/conversations-<set>.{tsv,yaml}`; `load_conversations(manifest)` rebuilds the clean audio and reference turns and checks the hashes. Telephone version = `telephone(audio)`.
- `record.py`: the Pydantic `CallRecord` (D-55), Phase 4 fields only: call info, segments (anonymous `S1`/`S2`, role `unknown`, no `words[]`) and meta.
- `diarize.py` + `pipeline.py`: pyannote community-1 diarization (gated: `HF_TOKEN`, terms accepted once) then per-turn `asr.transcribe` (long-form over 30 s) → `CallRecord` (D-23, D-23a, D-23b). `scripts/run_pipeline.py <audio> --config configs/phase4/pipeline.yaml`; `pipeline.process(models, audio, ..., num_speakers=)` is the in-memory entry the benchmark uses.
- `nlp/data.py`: Phase 5 data (D-31b): `slurp_tn` (intent + `[type : value]` entities), `tsac` (train/validation/test, test copies dropped, Arabizi swapped for Gemini's conversion in train), `synthetic_calls` (hash-checked); all text through `target_text` (D-34a). `nlp/llm.py`: `LLM(provider, model, cache)` for Gemini/Groq through `openai`, every reply cached by request hash; `QuotaExhausted` on the daily limit. `scripts/transliterate_tsac.py` and `scripts/make_synthetic_calls.py` (`configs/phase5/data.yaml`) are resumable; keys `GEMINI_API_KEY`, `GROQ_API_KEY`. No prompted-LLM row is scored (D-32e).
- `nlp/metrics.py`: Phase 5 scoring (D-70d): `parse_label` / `parse_entities` / `parse_roles` for the shared JSON reply format (invalid = wrong), `classification` (macro-F1 + accuracy) and `entity_f1` ((type, value) pairs after `normalize_arabic`), each with a bootstrap CI. `nlp.data.call_turns(call, shuffle_seed)` gives the plain or line-shuffled role input, speakers named by first appearance.
- `scripts/benchmark_pipeline.py --set dev|test` (`configs/phase4/benchmark.yaml`): DER / cpWER / WER / CER on the simulated conversations for every ASR model on the same pyannote turns, plus an oracle row (D-70b, D-70c). Tune on dev only; the test run writes the README `phase4` block. `--limit N` is the pre-flight.

Run configs go in `configs/`, entry-point scripts in `scripts/`.

## Constraints that shape code

- **Compute:** the laptop RTX 4070 (8 GB VRAM) is the main machine. Colab and Kaggle are overflow only. Training code must run unchanged in all three places (only config and environment differ) and must checkpoint and resume cleanly.
- **Network:** upload speed is about 0.68 Mbps. Nothing over ~50 MB is uploaded from home. Full checkpoints, datasets and base models stay local or go to the HF Hub through the cloud. Only small artifacts (configs, plots, eval reports, LoRA adapters) go to MLflow.
- **Run metadata:** every run records its git commit, config, dataset version and machine.
- **Transcription convention (D-10):** everything is in Arabic script, including French/English words. Report CER next to WER, and normalize Arabic orthography before scoring.
- **Secrets:** read tokens only from environment variables or the git-ignored `.env` (names are in `.env.example`).
- **Git contents:** git holds code and configs only. Data, checkpoints, audio and weights are git-ignored.
