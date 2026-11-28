# Model card: TunCall ASR, first LoRA (draft)

**Status:** draft for Phase 3. Not published; the release scope and licensing are reviewed
before any release (P-04, P-04a).

A speech recognition model for **Tunisian Arabic (Darija)**: Whisper large-v3-turbo with a
LoRA adapter trained on 128 hours of public Tunisian speech. It is the first stage of the
TunCall call-analysis pipeline. On the general Tunisian test set it reaches **26.6% WER and
10.3% CER**, about 10 points of WER better than FarukSTT, the strongest public Tunisian
model we tested.

## Model details

| | |
|---|---|
| Base model | OpenAI Whisper large-v3-turbo, 809M parameters, MIT license, pinned at Hub revision `41f01f3` (D-21a) |
| Method | LoRA, rank 32, alpha 64, dropout 0.05, on the attention (q, k, v, out) and feed-forward (fc1, fc2) layers of the encoder and decoder; about 28M trained parameters, the base weights are frozen (D-21b, ADR 0007) |
| Adapter | 111 MB (fp32), sha256 `4d1b9c99…a3db`; merged into the base weights before use, so the served model is a plain Whisper model (D-21d) |
| Input | Mono audio resampled to 16 kHz, **up to 30 seconds** per clip |
| Output | Text in **Arabic script only**: French and English words are written in Arabic letters, without diacritics (D-10) |
| Decoding | Beam search with 5 beams, language forced to Arabic, no temperature fallback (D-26) |
| Developed by | The TunCall project |

## Intended use

- Transcribing short segments of Tunisian Darija speech, including speech that switches into
  French or English, as the first step of call analysis (search, intent, sentiment, summaries).
- Research on Tunisian ASR: the training mix, test sets and scoring are all reproducible from
  the repository.

**Out of scope for this version:**

- Audio longer than 30 seconds without first cutting it into segments (the pipeline that does
  this is Phase 4).
- Telling speakers apart; the model outputs one text per clip, with no speaker labels.
- Real-time (streaming) transcription.
- Modern Standard Arabic or other Arabic dialects: they were not measured.
- Output in Latin script (Arabizi) or with diacritics.

## Training data

Four public corpora, 96,939 clips and 128 hours after cleaning (full details in the
datasheet):

| Source | What it is | Clips | Hours | License |
|---|---|---:|---:|---|
| LinTO Tunisian audio | Mixed Darija: YouTube talk shows, stories, radio, code-switching speech | 76,069 | 81.9 | CC BY 4.0 |
| TEDxTN | Tunisian TEDx talks | 15,322 | 22.1 | annotations CC BY-NC-ND 4.0 |
| Omnilingual ASR (Tunisian) | Spontaneous answers to prompts | 2,874 | 21.2 | CC BY 4.0 |
| SLURP-TN | Task-oriented requests to a voice assistant | 2,674 | 2.8 | CC BY-NC-ND 4.0 |

Cleaning before training (D-21c, D-16e):

- French and English words in Latin script were transliterated with the project's lexicon,
  and diacritics were removed from all transcripts.
- Clips longer than 30 seconds or left without text were dropped.
- 49 clips whose sentence also appears in a test or validation set were removed, so the test
  scores are free of repeated sentences.
- 500 LinTO clips, held out by whole recording, plus the SLURP-TN and TEDxTN validation splits
  (about 1,800 clips) form the development set used to pick the best checkpoint. None of them
  were trained on (D-17a).

Not used: two larger sources with machine-generated transcripts (Deep Confessions podcasts
and weakly labelled TunSwitch, about 333 hours). They are kept for a later with/without
comparison and join the mix only if they improve test WER (D-12d, deferred by P-11).

## Training procedure

- Hugging Face `Seq2SeqTrainer` with `peft`, on a single laptop RTX 4070 (8 GB).
- Learning rate 1e-4 with 500 warm-up steps and linear decay; effective batch of 32 clips;
  bf16 with gradient checkpointing; seed 0.
- 3 epochs, 9,033 optimizer steps. The development set was decoded with the same beam search
  as the benchmark every 500 steps, and the checkpoint with the lowest development WER was
  kept. That was the last one: **27.4% WER, 11.1% CER** on the development set, still falling
  at the end of training (more epochs are an open option, D-27).
- Every run logs its git commit, configuration, dataset version and machine to MLflow.

## Evaluation

### Test sets

All three are frozen: never used for training or tuning, with their file hashes recorded in
the repository.

| Test set | Utterances | What it measures |
|---|---:|---|
| General | 1,013 | The LinTO test split (~1.3 h): everyday Tunisian Darija from varied sources |
| Task-oriented | 893 | The SLURP-TN test split (~1 h): short requests, closest to support-call wording (D-13a) |
| Telephone band | 1,013 | The general set passed through a simulated landline: 8 kHz, 300–3,400 Hz, A-law compression (D-25a) |

### Results

WER and CER in %, with 95% bootstrap confidence intervals over utterances. ΔWER is a paired
bootstrap against FarukSTT; a negative value means fewer errors than FarukSTT.

| Test set | WER [95% CI] | CER [95% CI] | ΔWER vs FarukSTT [95% CI] | Looping |
|---|---|---|---|---:|
| General | 26.6 [24.8, 28.5] | 10.3 [8.9, 12.3] | −10.6 [−13.1, −8.5] | 1 |
| Task-oriented | 31.1 [29.4, 32.7] | 8.9 [8.3, 9.5] | −7.1 [−8.7, −5.7] | 0 |
| Telephone band | 35.2 [31.9, 39.1] | 15.7 [12.2, 20.0] | −5.6 [−8.8, −1.9] | 3 |

For comparison on the general set: FarukSTT scores 37.2% WER / 13.2% CER, and the base
Whisper large-v3-turbo without the adapter scores 143.7% / 112.0%, because it gets stuck
repeating itself on 61 clips. "Looping" counts those clips (D-15d).

These numbers are copied from the "Fine-tuned ASR" table in the project README, which the
benchmark script writes. To reproduce them:

```bash
uv run python scripts/benchmark_asr.py --config configs/phase3/benchmark.yaml
```

**Scoring:** before scoring, Latin-script words are transliterated with the same lexicon and
Arabic spelling is normalized (hamza and alef variants, ta marbuta, alef maqsura, diacritics,
tatweel, digits and punctuation), so the scores
measure words, not spelling conventions (D-10). Unnormalized ("raw") WER is reported next to
it in the README.

### Caveats on the comparison

- **The task-oriented set may share speakers with training.** SLURP-TN publishes no speaker
  IDs, so overlap between its train and test speakers can't be ruled out. This model was
  trained on SLURP-TN train and FarukSTT was not, so this set may favour our model (D-13a).
- **FarukSTT may have seen about 3% of the general test sentences** in its own training data
  (D-12b). That would favour FarukSTT.
- **The telephone set is a simulation.** It models a clean landline only: no background
  noise, mobile or VoIP compression, or bad connections. Real calls are likely harder.

## Limitations and risks

- **Telephone audio is the weakest case.** The model never heard telephone-quality audio in
  training; its telephone CER (15.7%) is slightly behind FarukSTT's (14.5%), with overlapping
  intervals. Training with telephone-band augmentation is the planned fix.
- **About one word in four is still wrong** on general speech. Many errors are small (spacing
  around «و», a single letter), but names, numbers and rare words can be misheard, so
  transcripts should not be treated as exact records.
- **Domain gap.** No real call-center audio exists publicly, so the model was never trained or
  tested on actual support calls.
- **Coverage.** The training data leans towards media speech (YouTube, TV, radio, TEDx).
  Regional accents, older or younger speakers and noisy environments may be under-represented,
  and accuracy across those groups was not measured.
- **Spelling convention.** Darija has no standard spelling. The output follows the training
  transcripts' conventions, which vary between sources.

## Language models (Phase 5)

After transcription, the pipeline labels each turn of a call with an **intent**, its
**entities** and its **sentiment**, and guesses whether each speaker is the **agent** or the
**customer**. Two kinds of models were fine-tuned per task and compared on validation data.
The better one ships, or the faster one when they are within one point:

| Task | Shipped model | Test score |
|---|---|---|
| Intent | Qwen3 4B Instruct, 4-bit, with a LoRA adapter | macro-F1 62.5% |
| Entities | Qwen3 4B Instruct, 4-bit, with a LoRA adapter | F1 67.3% on correct text, 40.6% on our transcripts |
| Sentiment | MARBERTv2, fine-tuned | macro-F1 97.4% |
| Speaker role | MARBERTv2, fine-tuned | 100% on synthetic calls |

The full comparison is in the README.

**Limits to know before using these labels:**

- **The intents are voice-assistant intents.** The only public Tunisian intent data,
  SLURP-TN, covers commands like setting an alarm or sending an email, not support-call
  reasons such as billing or a broken line. Swapping in call-centre labels later needs
  retraining only, with no pipeline changes.
- **Transcription errors hurt entities most.** Names and dates must be heard exactly, so
  entity F1 drops by about 25 points on our own transcripts. Intent drops by under 2.
- **Roles were tested on synthetic calls** written by an LLM, which make roles easy, so
  100% says little about real calls. Each speaker is judged on its own, so a call can get
  two customers.
- **Sentiment is positive or negative only**, learned from social-media comments, and its
  score may be inflated by near-duplicate comments in that dataset.
- Long turns are cut: the encoders read the first 128 tokens of each turn.

## Licensing and release

The adapter's license is **not decided**. Two training sources (SLURP-TN and TEDxTN) are
under non-commercial, no-derivatives licenses, and every source's license will be reviewed
before any release (P-04a). If released, the card must credit each source as its license
requires, including the CC BY 4.0 attribution for LinTO and the corpora it bundles. The base
model (MIT) allows redistribution.

## Credits

Base model: OpenAI Whisper large-v3-turbo. Training data: LinTO Tunisian audio (Linagora),
TEDxTN, Meta's Omnilingual ASR corpus (Tunisian part) and SLURP-TN (Elyadata). Baseline:
FarukSTT.
