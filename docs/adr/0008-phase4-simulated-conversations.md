# 0008. Phase 4: simulated conversations and a diarize-first pipeline

- **Status:** accepted
- **Date:** 2026-11-26
- **Decision log ID:** D-13b, D-23, D-23a, D-24, D-55, D-70b

## Context

Phase 4 turns a recording with several speakers into a transcript split by speaker
("who said what, when"). Its exit criterion was a pipeline evaluation on test calls, but
no Tunisian call audio is available (D-08v2). Scoring diarization needs recordings where
the words, the speakers and the timings are all known. Among the held-out sources, LinTO
and SLURP-TN carry no speaker ids, and Omnilingual has ids but is in the training mix.
TEDxTN's clip file names carry the talk title and speaker, and each talk has one speaker.
Its validation and test splits were never trained on.

## Options

1. **Simulated conversations from TEDxTN clips.** Exact labels at no cost, real Tunisian
   speech, fully reproducible. But they are optimistic: each speaker keeps their own
   microphone and room, there are no backchannels, and the speech is on stage, not in a
   dialogue.
2. **Hand-label 5–10 real talk shows or podcasts.** Realistic, but it takes hours of the
   owner's labelling time and the licensing is unclear.
3. **A public diarization benchmark (e.g. VoxConverse).** Comparable with papers, but not
   Tunisian, and its WER means nothing for our model.

For the pipeline order: diarize first and transcribe each turn, or transcribe the whole
recording and assign words to speakers by their timestamps (WhisperX style).

## Decision

Score Phase 4 on **simulated two-speaker conversations** built from TEDxTN, with each talk
treated as one speaker. Validation talks give the dev set and test talks give the frozen
test set, about 40 conversations of about 2 minutes each, in a clean and a telephone
(D-25a) version. A committed manifest is frozen, and the audio is rebuilt from code.
The pipeline is batch only. It diarizes first with pyannote
`speaker-diarization-community-1`, used off the shelf, and then transcribes each turn with
`asr.transcribe`. It outputs a Pydantic `CallRecord` with anonymous speakers and
turn-level times. Results report DER, cpWER and WER/CER with bootstrap CIs for our LoRA,
base Whisper large-v3-turbo, FarukSTT and an oracle-turns row.

## Consequences

- Every Phase 4 number is labelled "simulated" and is expected to look better than it
  would on real calls. Options 2 and 3 can be added later as extra columns.
- The pipeline needs no word timestamps. `words[]` is left out until a later phase needs it.
- pyannote is a gated model: the owner accepts its terms on the Hub once.
- Agent/customer roles are not attempted until Phase 5.
- The §2 diarization targets are set from the first Phase 4 numbers.
