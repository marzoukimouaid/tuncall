# 0005. Phase 1 benchmark protocol

- **Status:** accepted
- **Date:** 2026-11-21
- **Decision log ID:** D-13, D-14, D-22a, D-26, D-70a / D-73 (with D-12, D-20)

## Context

Phase 1 must produce baseline WER/CER for five existing models (D-20) that anyone can reproduce with one command, and that later fine-tuned models are compared against. The only test set that exists before Phase 2 is the LinTO test split (test-general, ~1.3 h, D-12). It is small, so point estimates alone would mislead. Transcripts are all Arabic script with no standard spelling for loanwords (0001), so raw WER counts spelling variants as errors. Uploads from home are limited to ~50 MB (N-01).

## Options

1. **Test sets:** freeze test-general now and add test-domain in Phase 2, or record a pilot domain set early.
2. **Freezing:** pin LinTO's Hub commit and commit file + utterance hashes, or mirror the split into our own HF repo.
3. **Inference:** HF Transformers (loads every candidate as-is, same stack as Phase 3 training), or faster-whisper (faster, needs a CTranslate2 conversion per model, output can differ slightly).
4. **Decoding:** beam 5 or greedy. In both cases Arabic is forced and there is no temperature fallback.
5. **Metrics:** jiwer plus our own normalizer, or HF `evaluate` plus an Arabic text library.

## Decision

- Freeze **test-general only** in Phase 1 (D-13). test-domain is added in Phase 2 as a new column of the same table.
- Freeze by **pinning LinTO commit `99796bf…`** and committing the sha256 of each test parquet plus a per-utterance digest in `configs/data/test-general.yaml`. Evaluation refuses to run on a mismatch (D-14).
- Run baselines with **HF Transformers** in fp16 on the laptop GPU (D-22a).
- Decode with `language="ar"`, `task="transcribe"`, `num_beams=5` and temperature 0 only (D-26).
- Score WER and CER with **jiwer** after our own D-10 normalizer. Report **95% bootstrap CIs** (1000 utterance resamples) and **paired** bootstrap CIs for deltas between models. Raw WER is shown for reference (D-70a, D-73).

## Consequences

- The baseline table is deterministic. Rerunning a model gives the same numbers, and any change to the data is detected.
- If LinTO rewrites or deletes that commit, the test set must be re-obtained from an archive. The hashes still prove whether a copy is identical.
- FarukSTT may have been trained on LinTO test audio, so its test-general number carries a contamination note. The uncontaminated comparison is test-domain (Phase 2).
- The normalizer and metric code live in `tuncall.eval` and are reused unchanged for Phase 3 evaluation.
- D-22, the serving inference engine, is still open; faster-whisper can be adopted later with a parity check against these numbers.
