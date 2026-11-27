# Phase 5 (NLP): implementation plan

Handoff spec from the Phase 5 grilling session (2026-11-27). Every choice below is decided
and logged in `PROJECT.md` §9 (P-14, D-30a, D-31a, D-32a, D-32b, D-33a, D-34a, D-55a,
D-70d) and summarized in ADR 0009. **Anything not covered here is still open: ask the
owner before implementing it** (working agreement 8b.1).

**Update (D-32c, 2026-11-27):** Llama 3.3 70B and Gemini 2.5 Flash no longer answer. Read
`openai/gpt-oss-120b` (Groq) and `gemini-3.5-flash` wherever this plan names them. Arabizi
conversion sends 25 comments per request.

**Update (D-31c, D-32d, D-32e, 2026-11-27):** Claude Sonnet wrote the Phase 5 data, and there
is **no prompted-LLM row** any more: Phase 5 compares the encoder and the Qwen3 4B LoRA only.
Step 2 is now the JSON output parsers, the metrics with CIs and the role inputs.

## Goal and exit criterion

Add intent, entities, sentiment and speaker roles to the pipeline, and publish a
comparison of four contenders on quality, speed and cost. Phase 5 is done when:

1. The README has a `phase5` block (between `<!-- phase5:start -->` / `<!-- phase5:end -->`
   markers, like `phase4`) with the comparison table from `scripts/benchmark_nlp.py --set test`.
2. `scripts/run_pipeline.py` fills the new `CallRecord` fields with the shipped models.
3. §2's NLP row has targets set from the test numbers (P-05 rule, as P-13 did), and §5 marks
   Phase 5 done.

Summaries are **not** in Phase 5 (D-33a, moved to Phase 7).

## Before writing code: check these facts

- Gemini and Groq free-tier rate limits and the exact model ids (Gemini 2.5 Flash, Llama 3.3
  70B Versatile on Groq). Both must work through the `openai` client with their
  OpenAI-compatible base URLs. If a model id is gone, ask the owner.
- Whether TunBERT loads in Transformers. Its only Hub copy (`tunis-ai/TunBERT`) needs
  `trust_remote_code` and is already fine-tuned for TSAC. Drop it after about an hour of
  fiddling (D-32a).
- The exact Qwen3 4B Instruct repo id and revision to pin.

## Tasks, data and labels (D-30a, D-31a, D-34a)

| Task | Data | Labels | Unit scored |
|---|---|---|---|
| Intent | SLURP-TN (`Elyadata/SLURP-TN`, revision from `configs/data/train-mix.yaml`), column `intent` | SLURP-TN's own intents, as is (21 in train) | utterance |
| Entities | SLURP-TN, column `tun_slu_annoation` (sic), markup `[type : value]` | SLURP-TN's own entity types (33 in train) | (type, value) pair |
| Sentiment | TSAC (`fbougares/tsac`, LGPL-3.0, train 13,669 / test 3,400) | positive / negative | comment |
| Role | Synthetic support calls (below) | agent / customer | speaker in a call |

Splits: train on train, choose and tune on validation, run test **once**. SLURP-TN test
is the frozen test-domain set, so it's never used for training or tuning. TSAC has no
validation split, so carve one from TSAC train with a fixed seed and commit the id list.

**Text convention (D-34a).** All gold text, and every entity value, passes through
`tuncall.asr.cache.target_text` (D-10a lexicon + harakat stripped) before training and
scoring, so gold text looks like our ASR's output (`ابعث mail` becomes `ابعث مايل`).
Entity markup is parsed *before* transliteration, and each value is transliterated on
its own.

**TSAC Arabizi half (D-31a).** About 47% of TSAC is in Latin letters (Arabizi). Gemini
(free tier) converts each Arabizi **train** comment to Arabic script, with replies cached.
The owner spot-checks 50 conversions. Converted comments are used for training only. TSAC
test is scored **only on comments that were in Arabic script originally**. Commit the
converted comments to `configs/data/` (small TSV) with a sha256.
`scripts/transliterate_tsac.py` does this and is resumable across free-tier daily limits.

**Synthetic support calls (D-31a).** `scripts/make_synthetic_calls.py` asks Gemini (free
tier) for **about 200 short Darija support calls**: telecom, bank and e-commerce topics,
6–20 lines each, all in Arabic script (D-10), every line tagged agent or customer by the
writer. Roles are true by construction because the writer decides them, not a model
guessing. The prompt and seed are fixed, and every reply is cached and committed.
Split **100 train / 30 validation / 70 test** with a fixed seed. The test files are
frozen, and their sha256 hashes are recorded like the other test sets. The owner
spot-checks about 20 calls for realistic Darija before freezing. This is synthetic
*text*. D-08v2 rejected synthetic *speech* only.

Role scoring uses two inputs: the call as written, and the same call with **lines
shuffled** (fixed seed). The shuffled input stops "the first speaker is the agent" from
scoring 100% without understanding anything.

## Contenders (D-32a, D-32b)

| Row | Model | How | Ships in pipeline? |
|---|---|---|---|
| Prompted rival | Llama 3.3 70B via Groq, free tier | prompt only | no |
| Prompted ceiling | Gemini 2.5 Flash, free tier | prompt only | no |
| Encoder | best of MARBERTv2 / CAMeLBERT-DA / TunBERT on validation | fine-tune, one per task | candidate |
| Small LLM | Qwen3 4B Instruct, 4-bit, LoRA | one LoRA per task | candidate |

The owner decides later which LLM rows appear in public write-ups. The benchmark always
computes both.

**Prompting protocol.** One fixed prompt per task, containing the label list, **one example
per label taken from train**, and a demand for JSON output. It's written once, checked on
validation, and never edited after looking at test. Invalid JSON counts as a wrong answer,
with no retry. Model ids are pinned in config, and every raw reply is cached to disk with
a key (model, prompt hash, input), so scores can be recomputed offline.

**Output format.** The prompted LLMs and the Qwen LoRA answer in the same JSON format, so
one parser handles both.
- Intent: `{"intent": ...}`
- Entities: `{"entities": [{"type": ..., "value": ...}]}`
- Sentiment: `{"sentiment": ...}`
- Role: `{"S1": "agent", ...}`

**Encoder entities.** These are token tags (BIO) built from the gold markup; predicted
spans are turned back into (type, value) pairs.

## Metrics and table (D-70d)

- Intent: **macro-F1** plus accuracy. The table notes that some test intents have only
  1–3 examples.
- Entities: **F1 over (type, value) pairs** per utterance, exact match after
  `tuncall.eval.normalize_arabic`, with positions ignored.
- Sentiment: macro-F1. Role: accuracy (plain and shuffled).
- Every score gets a 95% bootstrap CI over the scored unit (D-73).
- SLURP-TN tasks are scored on **two inputs**: gold text, and our LoRA ASR's transcript of
  the same test audio. That transcript already exists as `hyp` in
  `outputs/phase1/test-domain/turbo-lora-r32.jsonl`, reproducible by `benchmark_asr.py`.
  The gap between the two is the cost of ASR errors. Models are trained on gold text only.
- Extra columns:
  - milliseconds per input on the laptop GPU, over the test set (local models)
  - measured wall-clock time per input for the APIs, noting it includes network time and
    rate limits
  - parameter count
  - cost: $0 for every row, plus a note giving the paid-tier price per 1,000 inputs, read
    from each provider's pricing page on the run date
- Known bias to state under the table: Gemini wrote the synthetic calls, so its role score
  has a home advantage.

## Pipeline wiring (D-55a)

- **Which model ships, per task:** the local contender (encoder or Qwen LoRA) with the
  best **validation** score. If the two are within 1 point, the faster one ships. The API
  models never ship: they need the internet, have rate limits and can be retired.
- **New `CallRecord` fields** (in `src/tuncall/record.py`, which grows one phase at a time
  per D-55):
  - per segment: `intent`, `entities[] {type, value}` and `sentiment`
  - per speaker: a `role` guess, agent or customer, replacing the fixed `unknown`
  - `nlp_model_versions`
- **Model card:** the intent labels are voice-assistant intents (`email_sendemail`,
  `alarm_set`), so the card says this. The label set can be swapped later without pipeline
  changes.

## Code layout and dependencies

- `src/tuncall/nlp/`:
  - data loaders for SLURP-TN, TSAC and synthetic calls
  - the LLM client and its reply cache
  - encoder and LoRA training
  - metrics
- Scripts:
  - `scripts/make_synthetic_calls.py`
  - `scripts/transliterate_tsac.py`
  - `scripts/train_nlp.py --config ...`
  - `scripts/benchmark_nlp.py --set dev|test`
  - Each long script takes `--limit N` as a pre-flight, like the earlier phases.
- Configs go in `configs/phase5/` (YAML + `StrictModel`, D-87). Every training and
  evaluation run logs through `tuncall.tracking.Tracker`, MLflow experiment `phase5-nlp`.
- Frozen data goes in `configs/data/`: synthetic calls, the converted TSAC comments, and
  the id lists for splits made by us. Hashes are recorded.
- New dependencies:
  - `openai`, a runtime dependency: one client for both Gemini and Groq through their
    OpenAI-compatible endpoints.
  - `bitsandbytes` in the `cpu` and `gpu` extras next to peft.
- Keys `GEMINI_API_KEY` and `GROQ_API_KEY` go in `.env.example` (names only).
- Run `uv run ruff check --fix && uv run ruff format`, `uv run pyright`, `uv run pytest`.
  Add a CPU smoke test for each training path (tiny random model), like
  `tests/test_asr_train.py`.

## Order of work (P-14): one issue and one PR each, milestone "Phase 5 – NLP"

0. ADR 0009 and the §9 rows already exist (this session). Open the four issues.
1. **Data:** SLURP-TN and TSAC loaders with the text convention; TSAC validation split;
   `transliterate_tsac.py`; `make_synthetic_calls.py`; frozen files and hashes. Stop for
   the owner's spot-checks (50 TSAC conversions, about 20 calls) before freezing.
2. **Prompted rows:** LLM client and cache, prompts, JSON parser, metrics with CIs, and
   validation scores for both APIs.
3. **Training:** encoder fine-tunes (pick the best on validation) and Qwen 4-bit LoRA per
   task; checkpoint and resume as in Phase 3.
4. **Benchmark and pipeline:** `benchmark_nlp.py`, the README `phase5` block, the shipping
   choice per task, the `CallRecord` fields, the model card note, the §2 targets and the
   Phase 5 status in §5.

Git: short branch per PR off `main`, Conventional Commit titles, `Closes #N`, and no AI
attribution anywhere (see `CLAUDE.md`).

## Owner touchpoints

- Spot-check about 50 Arabizi conversions and about 20 synthetic calls (step 1).
- Approve the one-time test run (step 4).
- Decide which LLM rows go into public write-ups (any time after step 4).
