# 0004. ASR base model: Whisper large-v3-turbo

- **Status:** accepted
- **Date:** 2026-11-21
- **Decision log ID:** D-21a (with D-20)

## Context

The project must show our own model training, not reuse someone else's fine-tune. FarukSTT (Whisper large-v3 fine-tuned on Darija) is the strongest public model and the baseline to beat. Fine-tuning runs on an 8 GB laptop GPU (0003). Speech is code-switched Darija/French/English, transcribed in Arabic script (0001).

## Options

1. **Whisper large-v3-turbo (809M, MIT):** large-v3 encoder with a 4-layer decoder. Multilingual decoder suits code-switching. LoRA fits in 8 GB. Fast serving via faster-whisper. Less decoder capacity than large-v3.
2. **Whisper large-v3 (1.55B, MIT):** highest ceiling and the same base as FarukSTT. LoRA in 8 GB is borderline, so most runs would move to Colab.
3. **w2v-BERT 2.0 + CTC (~600M, MIT):** full fine-tune fits 8 GB, no hallucination, fast. Needs an external language model for good WER and has a weaker ecosystem for diarization.
4. **Continue from FarukSTT:** likely the best numbers quickly, but it's someone else's model and its training data has no stated license.

## Decision

Option 1. FarukSTT is a zero-shot baseline only, never a starting checkpoint. Full fine-tune vs. LoRA is still open (D-21).

## Consequences

- Phase 1 benchmarks turbo zero-shot alongside large-v3, medium, small and FarukSTT (D-20).
- Seq2seq can hallucinate on silence and noise, so the pipeline needs VAD and output checks.
- The base model's MIT license allows publishing our fine-tuned model (B-04c).
