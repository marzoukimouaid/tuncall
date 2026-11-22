# 0006. Phase 2 data strategy: public sources, no recordings

- **Status:** accepted
- **Date:** 2026-11-22
- **Decision log ID:** D-08v2 (with D-08b, D-10a, D-12c, D-12d, D-13a, D-14b, D-16, D-25, P-05b, P-09)

## Context

D-08 planned to record role-played support calls with about 20 volunteers (~120 calls) and use them as the domain test set and main training set. The owner can't organise that. A search of the HF Hub, the LDC catalog and recent papers found no public Tunisian call-center audio. The closest real conversational sources need access we don't have: LDC2025S05 (210 h of Tunisian telephone conversations, LDC licence) and TARIC-SLU (railway-desk dialogues, download site down). Meanwhile several public Tunisian speech sets appeared that Phase 1 didn't use. The owner restated the goal as an ASR model of our own that understands Tunisian Darija in general (P-09).

## Options

1. **Record role-played calls (D-08):** a clean, publishable call-center test set, but not feasible for the owner.
2. **Synthetic calls:** LLM-written dialogues spoken by a Tunisian TTS (a pilot with Habibi-TTS sounded good enough). The text is exact, but the audio is fake; it can't serve as a test set, and without a call test set its training value can't be measured.
3. **Licensed telephone speech (LDC2025S05, TARIC):** real conversational audio, but needs paid or emailed access and can't be redistributed.
4. **A multi-source public mix:** every usable public Tunisian corpus through one adapter each, a task-oriented test set (SLURP-TN), and telephone robustness measured by simulating the phone channel on test-general.

## Decision

Option 4. No own recordings and no synthetic speech (D-08v2, D-08b).

- **Core training mix (D-12c):** LinTO, SLURP-TN train, Omnilingual ASR `aeb_Arab` only, TEDxTN (segmented mirror), senior_tunisian_voice. About 140 h.
- **Ablations (D-12d):** Deep Confessions and TunSwitch-weak have machine-made transcripts. Phase 3 trains with and without them. Both are shrunk on Colab into private derived repos.
- **test-domain = SLURP-TN test (D-13a),** frozen with the D-14 recipe on transliterated references (D-14b). Telephone-band test-general is a new column (D-25), and both get baselines before their targets are set (D-15e, P-05b).
- **One adapter per source (D-16)** reads a pinned upstream revision and yields a common record; Latin-script French/English is mapped to Arabic script through a lexicon reviewed by the owner (D-10a). A dedupe report runs before any training.

## Consequences

- No call-center claim: the domain result is "task-oriented requests" (SLURP-TN). A real call test set can be added later as a new column without touching the frozen sets.
- Several sources are non-commercial or no-derivatives (SLURP-TN, TEDxTN), and one has unclear provenance (the TEDxTN mirror). Licensing is set aside for now and reviewed before any release (P-04a).
- SLURP-TN has no speaker IDs, so test-domain may share speakers with SLURP-TN train. The score carries that caveat.
- Adding a source later (e.g. LDC if access appears) is one adapter and one config line.
- About 14 GB of core downloads plus Phase 3 checkpoints need more disk than is free; the owner frees space first (S-01c).
- New work for Phase 2: adapters, the transliteration lexicon, the dedupe report, the test-domain freeze and benchmark columns, Colab preprocessing, and `docs/datasheet.md` (P-10).
