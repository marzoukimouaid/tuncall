# 0009. Phase 5: NLP on public labels, synthetic calls for roles, four-way comparison

- **Status:** accepted
- **Date:** 2026-11-27
- **Decision log ID:** P-14, D-30a, D-31a, D-32a, D-32b, D-33a, D-34a, D-55a, D-70d

## Context

Phase 5 adds intent, sentiment, entities and summaries to the pipeline and compares a
prompted large LLM, a fine-tuned encoder and a LoRA-tuned small LLM (D-32). There is no
call data (D-08v2), so there is nothing to label from real calls. A search of public
Tunisian resources (2026-11-27) found:
- SLURP-TN already carries intent labels (21) and entity markup (33 types) for every clip,
  with audio, and its test split is our frozen test-domain set.
- The only labelled Tunisian sentiment data is social-media comments (TSAC, about half in
  Arabizi).
- No public Tunisian NER or summarization set exists.
- Nothing has agent/customer roles.

The owner has no budget for paid APIs and prefers low-hassle options.

## Options

1. **SLURP-TN as the scored benchmark** for intent and entities: real speakers, no
   labelling, and audio to score on our ASR output too. But it's voice-assistant
   commands, not support calls.
2. **LLM-written support-call transcripts, labelled for everything.** This fits the call
   story, but everything is synthetic and checking it costs the owner hours.
3. **Roles:** leave them `unknown` (nothing to score); have an LLM guess roles without a
   score; or have an LLM *write* calls with roles marked, so the answers are true by
   construction.
4. **Prompted rival:** a strong model (Gemini), or a known-size open model that is weaker
   at Darija (Llama 3.3 70B), which supports a size claim ("matches X at 1/17 the size").

## Decision

- **Intent and entities:** scored on SLURP-TN with its own labels, on gold text and on our
  LoRA ASR's transcripts of the same audio.
- **Sentiment:** TSAC, positive/negative. Arabizi train comments are converted to Arabic
  script by the free LLM, and test uses only comments originally in Arabic script.
- **Roles:** about 200 Gemini-written Darija support calls with roles marked by
  construction (frozen 70-call test set), scored on plain and line-shuffled text.
- **Contenders:**
  - Llama 3.3 70B (Groq) and Gemini 2.5 Flash, both prompted on free tiers with replies
    cached.
  - The best of MARBERTv2 / CAMeLBERT-DA / TunBERT.
  - Qwen3 4B with 4-bit LoRA.
- **Pipeline:** the best local contender per task, on validation, ships into the pipeline.
- **Summaries:** moved to Phase 7.

## Consequences

- No labelling work for the owner, apart from spot-checks (about 50 conversions and about
  20 calls).
- Every Phase 5 number has a caveat:
  - intent and entities: voice-assistant commands
  - sentiment: social-media text
  - roles: synthetic text, and Gemini wrote it, so Gemini's role score has a home advantage
- The pipeline's intent field uses voice-assistant labels until real call intents exist.
  The model card says so.
- Free-tier models can be retired. Cached replies keep the numbers reproducible, but not
  re-runnable against the same model.
- Phase 7 inherits summaries, together with the LLM-as-judge question (D-72).
- Plan for the implementing session: `docs/phase5-plan.md`.

## Amendments

- D-32c (2026-11-27): the pinned API models were retired; replaced by `openai/gpt-oss-120b` and `gemini-3.5-flash`.
- D-31c (2026-11-27): Claude Sonnet wrote the synthetic calls and the Arabizi conversions instead of Gemini.
- D-32d (2026-11-27): the Gemini ceiling row is dropped.
- D-32e (2026-11-27): the Groq rival row is dropped too (200,000 tokens/day free limit); Phase 5 compares the two local contenders only.
