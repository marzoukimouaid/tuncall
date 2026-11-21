# 0001. Transcription convention: all Arabic script

- **Status:** accepted
- **Date:** 2026-11-20
- **Decision log ID:** D-10

## Context

Darija speech switches between Tunisian Arabic, French and English mid-sentence. Public datasets disagree on how to write it: some use Arabic script for everything, some write French/English words in Latin script, some use Arabizi. ASR targets and WER are only meaningful if every dataset uses one convention.

## Options

1. **All Arabic script, French/English transliterated:** matches LinTO, our main public source. One script gives the model consistent targets. French loanwords have no standard Arabic spelling, which inflates WER.
2. **Arabic script for Darija, Latin script for French/English:** easier to read, closer to how bilingual speakers write. Datasets would need language tagging per word, and the model would have to switch scripts mid-sentence.
3. **Arabizi:** common on social media, but unstandardized. No major speech dataset uses it.

## Decision

Option 1: everything in Arabic script, including French and English words.

## Consequences

- Datasets with Latin-script words need a Latin → Arabic transliteration step (a C1 component with its own tests).
- An Arabic orthographic normalizer runs before scoring (alef/hamza, ta marbuta/ha, alef maqsura/ya, diacritics, tatweel, numerals). CER is always reported next to WER.
- Product and offer names appear in Arabic script, so NER needs an entity normalizer to canonical forms.
- An Arabizi/Latin display converter stays optional (D-34).
