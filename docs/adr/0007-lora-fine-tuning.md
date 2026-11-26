# 0007. LoRA fine-tuning of Whisper large-v3-turbo on the laptop

- **Status:** accepted
- **Date:** 2026-11-23
- **Decision log ID:** D-21b (with D-21c, D-17a, D-83b)

## Context

Phase 3 fine-tunes Whisper large-v3-turbo (809M parameters, D-21a) on ~128 h of Tunisian
speech. The main machine is a laptop RTX 4070 with 8 GB of GPU memory (B-01c); Colab's T4
(16 GB, fp16 only, sessions die) is overflow. Updating all 809M weights with Adam needs about
13 GB for weights, gradients and optimizer state before activations.

## Options

1. **Full fine-tune on Colab:** every weight learns; every run moves to a T4, with slower
   iteration and session limits.
2. **Freeze the encoder, fully train the 4-layer decoder:** fits 8 GB; the audio encoder
   can't adapt to Tunisian acoustics or telephone audio.
3. **LoRA on encoder and decoder:** the base stays frozen and small rank-32 matrices are
   trained beside every attention and feed-forward layer (~28M parameters); fits 8 GB with
   bf16 and gradient checkpointing.

## Decision

LoRA (option 3) on the laptop, through the Hugging Face `Seq2SeqTrainer` with `peft`. The
dev set (LinTO hold-out + SLURP-TN and TEDxTN validation) is decoded with beam 5 during
training, so dev numbers use the benchmark's decoding.

## Consequences

- All runs stay on the laptop; Colab is not needed for Phase 3's first run.
- The saved adapter (~28M parameters in fp32) is 111 MB, over the 50 MB artifact limit
  (N-01), so it stays local like checkpoints; MLflow gets metrics and configs.
- Measured on the laptop with synthetic 30 s clips: peak GPU memory 4.8 GiB, ~10 s per
  optimizer step (32 clips), so the 3-epoch first run (~9,000 steps) takes about 26 h of
  training plus the dev passes, spread over several nights with resume.
- Scoring a fine-tuned model in `scripts/benchmark_asr.py` needs it to load a LoRA adapter
  (or a merged model), which is follow-up work.
- A full fine-tune on Colab remains possible as a later ablation.
