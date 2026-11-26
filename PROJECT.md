# Tunisian Darija Call Intelligence — Project Source of Truth

> **Codename:** `tuncall` (placeholder — P-01)
> **Owner:** Mouaid Marzouki
> **Doc version:** v0.29 — 2026-11-22
> **Status:** Phases 0–1 done; **Phase 2 (domain data) in progress**, plan in D-08v2. Any technology choice not logged in §9 is still OPEN until decided together.

This file is the single reference for the project. Every decision made gets written into the Decision Log (§9) with the date and reason. If code and this document disagree, update one so they match. Don't leave them drifting apart.

---

## 1. Vision

Build an **end-to-end speech and language intelligence system for Tunisian Darija**, the code-switched mix of Tunisian Arabic, French and English that people actually speak. It turns raw customer-support calls into structured, searchable and measurable information:

> audio in → *who said what, when* → *what they wanted, how they felt, which entities were mentioned* → *summaries, search, analytics, answers*

The project has three goals:

1. **Career:** show AI-engineering skill across the full model lifecycle (data → fine-tuning → evaluation → serving → monitoring), not just API calls.
2. **Contribution:** a Tunisian Darija ASR model of our own that is measurably better than the strongest public Darija baselines, with a reproducible benchmark. (We no longer record or publish our own domain dataset, P-09.)
3. **Commercial:** be something a Tunisian / North African call center, telecom, bank or BPO would pay for, either as a product or as a freelance/consulting engagement.

### 1.1 Why this project, and where the bar is

- Generic Whisper fine-tuning on Darija **already exists**. The best public model, FarukSTT (fine-tuned from Whisper large-v3), reports about 32% WER on its own validation set; base Whisper large-v3 is above 50%. Linagora/LinTO has released CC BY 4.0 Tunisian audio and text datasets, and TunSwitch, IWSLT Tunisian and others are also available.
- FarukSTT is used **only as a baseline to beat**. Our model is trained by us from a general-purpose base model (Whisper large-v3-turbo, D-21a), not continued from FarukSTT.
- So **the novelty can't be "I fine-tuned Whisper."** It has to come from:
  - **A stronger general Darija ASR model**, trained by us on a curated multi-source mix of public Tunisian speech (D-12c), measured on general, task-oriented (test-domain) and telephone-band test sets (P-09). No public customer-support call audio exists, so the original call-center domain target was dropped (D-08v2).
  - A **full system** around the model: diarization, NLP, retrieval, analytics.
  - **Engineering discipline:** evaluation harness, CI gates, reproducibility, serving performance, observability.

### 1.2 Non-goals (proposed, to confirm)

- Training a speech foundation model from scratch.
- Supporting all Arabic dialects (Tunisian first; others only if they come almost for free).
- Handling real customer recordings that contain PII before a privacy story exists (see D-07).

---

## 2. Success Criteria (set 2026-11-22 from the Phase 1 baselines — P-05a)

| Area | Metric | Target |
|---|---|---|
| ASR | ~~WER on our held-out **support-call test set** (test-domain, Phase 2): ≥ 20% lower WER than FarukSTT (relative), with the paired 95% CI of ΔWER entirely below 0~~ | **Superseded by P-05b** |
| ASR | WER / CER on **test-domain** (SLURP-TN test, D-13a) | TBD from its baseline in Phase 2 (P-05b) |
| ASR | WER / CER on **telephone-band test-general** (D-25) | TBD from its baseline in Phase 2 (P-05b) |
| ASR | WER / CER on the **general** Darija set (test-general) | **WER ≤ 40%, CER ≤ 17%** (set to match FarukSTT, then 38.8% / 16.3%; rescored with the D-10a lexicon (D-16c) it is 37.2% [35.1, 39.7] / 13.2%; it may have seen 3% of these clips, D-12b). **Met** by the first LoRA (2026-11-26): 26.6% [24.8, 28.5] / 10.3% |
| ASR | Looping utterances on test-general (D-15d) | **≤ 10 of 1013** (base Whisper large-v3-turbo: 61). **Met** by the first LoRA: 1 |
| Diarization | DER on test calls | TBD in Phase 4, after its baseline |
| NLP | Macro-F1 on intent / sentiment; entity-level F1 on NER | TBD in Phase 5, after its baseline vs. a prompted large LLM |
| System | End-to-end latency per minute of audio; cost per hour of audio | TBD in Phase 6 |
| Reproducibility | Someone else can reproduce headline numbers from the repo | Yes |
| Publication | Model card(s) + write-up | Scope TBD (P-04); training-source licenses reviewed before any release (P-04a) |

Rule: **every claim in the README must come with a number and the eval script that produced it.**

---

## 3. System Overview

```
                ┌────────────────────────────────────────────────────────────┐
  audio file /  │  INGESTION                                                 │
  live stream ─►│  format normalization · resampling · VAD · channel split   │
                └───────────────┬────────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────────┐
                │  SPEECH LAYER                                              │
                │  ASR (domain-adapted, code-switch aware)                   │
                │  + word timestamps / alignment                             │
                │  + speaker diarization (agent vs customer)                 │
                └───────────────┬────────────────────────────────────────────┘
                                ▼  diarized, timestamped transcript
                ┌────────────────────────────────────────────────────────────┐
                │  LANGUAGE LAYER                                            │
                │  text normalization (script / Arabizi handling)            │
                │  intent · sentiment · NER (phones, offers, products, ...)  │
                │  call summary · resolution status · QA checks              │
                └───────────────┬────────────────────────────────────────────┘
                                ▼  structured call record
                ┌────────────────────────────────────────────────────────────┐
                │  KNOWLEDGE & ANALYTICS LAYER                               │
                │  storage · hybrid search over calls · RAG Q&A              │
                │  ("why are customers calling about X this week?")          │
                │  dashboards: volumes, intents, sentiment trends            │
                └───────────────┬────────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────────┐
                │  INTERFACES                                                │
                │  REST/WebSocket API · dashboard UI · public demo           │
                └────────────────────────────────────────────────────────────┘

  Cross-cutting: evaluation harness · experiment tracking · data/model versioning
                 · CI/CD · observability · security & privacy
  Stretch:       real-time Darija voice agent (ASR → LLM → TTS)
```

### 3.1 Components in detail

**C1 — Data & Annotation**
- Gather public Tunisian audio/text datasets, then clean, dedupe and normalize them to one transcription convention (D-10).
- No own recordings (D-08v2): one adapter per public source (D-16) yields a common record; the training mix is a YAML list (D-12c, D-12d). French/English written in Latin script is transliterated with a reviewed lexicon (D-10a).
- Strictly separated splits: `train / dev / test-domain / test-general`. **The test sets are frozen once created** and never used for training or tuning. test-domain = SLURP-TN test (D-13a).
- Dedupe report: overlap between sources and against both test sets (D-16); train clips matching a test or validation split are excluded from the mix (D-16e).
- Datasheet (`docs/datasheet.md`, P-10): sources, licenses, hours, speakers, script convention, known gaps.

**C2 — ASR**
- First, **benchmark** candidate base models on our test sets with no training (D-20).
- Then fine-tune **Whisper large-v3-turbo** (D-21a) on domain data (method D-21). FarukSTT is a baseline to beat, never a starting checkpoint. Ablations: data mix, augmentation (noise, telephone-band 8 kHz simulation, speed), LoRA rank, etc.
- Metrics: WER, CER, and a **code-switch-aware breakdown** (errors on Arabic vs. French vs. English tokens).
- Export an optimized inference build (D-22).

**C3 — Alignment & Diarization**
- Word-level timestamps.
- Speaker diarization, or channel-based separation if the audio is stereo (D-23).

**C4 — NLP**
- Define the label set: intents, sentiment scale, entity types (D-30).
- Build labels: manual, or LLM-assisted with human review (D-31).
- Compare three approaches: **(a)** prompted large LLM, **(b)** fine-tuned encoder, **(c)** LoRA-fine-tuned small LLM, on quality, latency and cost (D-32). Ship the winner and publish the comparison.
- Summarization and QA checks ("did the agent confirm identity?").

**C5 — Knowledge & Analytics**
- Store structured call records.
- Hybrid search (keyword + vector) over transcripts, then RAG Q&A with citations to call segments (D-40…D-43).
- Aggregate analytics.

**C6 — Serving & Backend**
- API for submitting audio and getting results (sync for short clips, async job queue for long files).
- Model servers separate from the API so they can be scaled on their own (D-50…D-54).

**C7 — Interfaces**
- Dashboard UI and a public demo (D-60, D-61).

**C8 — Evaluation Harness** (first-class component, not an afterthought)
- One command reproduces every headline number.
- Golden sets for every layer; regression gates in CI (D-70…D-73).

**C9 — MLOps & Observability**
- Experiment tracking, model/data versioning, containerization, CI/CD, runtime metrics and traces (D-80…D-86).

**C10 — Stretch: Real-time Voice Agent**
- Streaming ASR → LLM with tools/RAG → Darija/French TTS, with latency budgets (D-90…D-92).

---

## 4. Data Flow & Contracts (draft)

A single **CallRecord** schema travels through the system. Exact fields and the serialization format are decided in D-55.

```
CallRecord
├── call_id, source, created_at, duration_s, sample_rate, channels
├── segments[]            # from C2/C3
│   ├── start, end, speaker (agent|customer|unknown)
│   ├── text_raw, text_normalized
│   ├── words[] {w, start, end, conf, lang_tag?}
│   └── asr_model_version
├── nlp                   # from C4
│   ├── intents[] {label, conf, segment_refs}
│   ├── sentiment {overall, per_segment[]}
│   ├── entities[] {type, value, normalized, segment_ref}
│   ├── summary, resolution_status, qa_checks{}
│   └── nlp_model_versions{}
└── meta {pipeline_version, processing_ms_by_stage{}, cost_estimate}
```

Every output records **which model version produced it**, so any result can be traced and reproduced.

---

## 5. Phases & Milestones (no fixed dates — P-03; tasks tracked in GitHub Issues — D-05)

| Phase | Goal | Exit criteria | Status | Milestone |
|---|---|---|---|---|
| **0. Foundations** | Repo, environment, compute, tracking, this doc finalized | All P-xx and core D-xx decisions logged | Done | [M0](https://github.com/marzoukimouaid/tuncall/milestone/1) |
| **1. Benchmarks first** | Frozen test sets and baseline numbers for existing models | Baseline table in README, reproducible with one command | Done (test-general only, D-13; targets P-05a) | [M1](https://github.com/marzoukimouaid/tuncall/milestone/2) |
| **2. Domain data** | Multi-source training mix and a frozen task-oriented test set from public Tunisian speech (D-08v2) | Adapters for every source + reviewed transliteration lexicon; dedupe report; test-domain frozen, with SLURP-TN and telephone-band test-general columns in the baseline table; `docs/datasheet.md` (P-10) | Done (2026-11-23; noisy sources as private derived repos, D-12h) | [M2](https://github.com/marzoukimouaid/tuncall/milestone/3) |
| **3. ASR fine-tuning** | Domain-adapted ASR beating baselines | ~~Ablation table~~ (deferred, P-11); model card draft (`docs/model-card.md`, D-21f) | In progress (first LoRA beats FarukSTT on all three test sets, README "Fine-tuned ASR") | [M3](https://github.com/marzoukimouaid/tuncall/milestone/4) |
| **4. Diarization + pipeline** | Audio → diarized transcript, end to end | Pipeline eval (WER + DER) on test calls | Not started | [M4](https://github.com/marzoukimouaid/tuncall/milestone/5) |
| **5. NLP** | Intent / sentiment / NER / summary | 3-way comparison table; chosen approach shipped | Not started | [M5](https://github.com/marzoukimouaid/tuncall/milestone/6) |
| **6. Serving & product** | API, job queue, storage, dashboard, demo | Deployed demo; load/latency numbers | Not started | [M6](https://github.com/marzoukimouaid/tuncall/milestone/7) |
| **7. Knowledge layer** | Search + RAG + analytics | RAG eval (retrieval + faithfulness) | Not started | [M7](https://github.com/marzoukimouaid/tuncall/milestone/8) |
| **8. Hardening & publish** | CI gates, monitoring, write-up, releases | Public release (scope per P-04) | Not started | [M8](https://github.com/marzoukimouaid/tuncall/milestone/9) |
| **9. Stretch** | Real-time voice agent | Latency budget met | Not started | [M9](https://github.com/marzoukimouaid/tuncall/milestone/10) |

**Release cut:** no deadline (P-03). Phases are quality gates: a phase is done when its exit criteria are met. Each phase has a GitHub milestone, and its status is updated here when it changes.

---

## 5b. Development & MLOps Workflow (decided 2026-11-20, see D-80v2, D-80c, D-80d, B-01c, N-01, S-01)

```
 YOUR LAPTOP (Ubuntu)                                     CLOUD
 ┌──────────────────────────────────────────┐          ┌──────────────────────────────────────┐
 │ Training & eval (RTX 4070, uv, scripts)  │ metrics  │ DagsHub = hosted MLflow              │
 │   │                                      │ + small  │  • run records, metrics, params      │
 │   ├─ logs metrics/params (async) ────────┼────────► │  • small artifacts (plots, preds,    │
 │   │                                      │ artifacts│    LoRA adapters ≤ ~50 MB)           │
 │   └─ full checkpoints stay LOCAL         │ (KB–MB)  │  • model registry                    │
 │                                          │          │                    ▲                 │
 │ Data cache (datasets, base models)       │          │ Colab (overflow) ──┘ logs live too   │
 │ Checkpoints (best + latest, auto-clean)  │          │                                      │
 │                                          │  code    │ GitHub + Actions (CI, image builds,  │
 │ docker compose (later): serving stack    │ ───────► │   release: merge LoRA → HF Hub)      │
 └──────────────────────────────────────────┘          │ HF Hub: datasets & released models   │
                                                       └──────────────────────────────────────┘
```

**Where each thing happens**

| Activity | Where | Notes |
|---|---|---|
| Writing code, unit tests, quick debug runs | Laptop | uv `.venv`; pre-commit (D-84d) |
| Training & evaluation | Laptop (RTX 4070, 8 GB) | Scripts + config files (YAML + Pydantic, D-87); checkpoint & resume built in |
| Jobs needing > 8 GB GPU memory | Colab / Kaggle | Same scripts and configs, run from a clone of the repo; checkpoint often |
| Heavy data processing of public datasets + pushing clean versions | Colab | Data-center bandwidth; result pulled to the laptop once |
| Experiment tracking & small artifacts | DagsHub (hosted MLflow) | Async, batched, buffered logging (D-80d) |
| Full checkpoints, data cache, base models | Laptop disk | Retention: best + latest per run (S-01) |
| CI (lint, types, tests, tiny-sample smoke test) | GitHub Actions | Checks per D-83; smoke test postponed to Phase 3 (D-83a) |
| Docker image builds | GitHub Actions → registry | Never uploaded from home |
| Publishing full models | GitHub Actions or Colab | Only the small LoRA adapter leaves home; merge + push happens in the cloud (runner capacity to verify) |
| Serving stack development (later phases) | Laptop, docker compose | Deployment target TBD (D-86) |

**Rules**
1. Every run must record its git commit, config, dataset version and machine (laptop / Colab / Kaggle) in MLflow.
2. Training code must run unchanged on the laptop and on Colab/Kaggle; only config and environment differ.
3. Nothing over ~50 MB is uploaded from home (N-01).
4. A logging failure must never crash or slow down training (D-80d).
5. Test sets are frozen; their files are hashed and the hashes recorded in this repo.

---

## 6. Repository Layout (decided 2026-11-20, D-03a)

One Python package (`src/tuncall/`), one `pyproject.toml`, one `uv.lock`. Subpackages are added when their first code arrives, not as empty placeholders. If serving later needs its own image and dependencies, split into a uv workspace.

```
tuncall/
├── PROJECT.md · README.md · LICENSE (MIT)
├── pyproject.toml · uv.lock · .python-version (3.12)
├── .pre-commit-config.yaml       # D-84d
├── .env.example                  # token names only; real .env is git-ignored
├── .github/
│   ├── workflows/ci.yml          # D-83
│   ├── ISSUE_TEMPLATE/           # feature / bug / experiment (D-05)
│   └── pull_request_template.md
├── src/tuncall/
│   ├── data/      # C1 (planned)
│   ├── asr/       # C2: Whisper inference (built, D-15a)
│   ├── eval/      # C8 (planned)
│   ├── config.py  # YAML + Pydantic loader, D-87 / D-87a
│   └── tracking.py# non-blocking MLflow tracker, D-80d / D-80e
├── configs/       # YAML run configs
├── scripts/       # entry points (train, eval, data prep)
├── tests/
└── docs/adr/      # ADRs (D-06); 0000-template.md
```

Never in git: `data/`, `checkpoints/`, `outputs/`, `mlruns/`, audio files, model weights, `.env` (B-04).

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No public customer-support call audio exists | No call-center benchmark; domain claim limited to task-oriented speech | Multi-source public mix (D-12c) and SLURP-TN as test-domain (D-13a); a call test set (TARIC, LDC, a company's calls) can be added later as a new column (D-08v2) |
| Noisy (AI-generated) transcripts in large sources | Model learns another system's errors | Used only as with/without ablations, kept only if real-test WER improves (D-12d) |
| Inconsistent transcription conventions across datasets | Noisy WER, misleading results | Fix one convention (D-10) and normalize everything before training or eval |
| Test-set contamination (public datasets overlap with model training data) | Inflated numbers | Dedupe report across sources and against both test sets (D-16); FarukSTT overlap measured for each test set (D-12b, D-15e) |
| GPU limits | Slow iteration, can't train large models | Pick model sizes to fit compute (B-01); LoRA; smart scheduling |
| Slow, high-latency home connection | Blocked network, slow logging, failed uploads | Network rules N-01; async buffered logging D-80d; heavy transfers in the cloud |
| DagsHub free-tier limits change (storage listed as 200 GB officially, 20 GB elsewhere) | Tracking or artifacts disrupted | Keep artifacts small; MLflow client is standard, so we can move to self-hosted MLflow by changing the URL |
| A token (HF / DagsHub) gets committed to the public repo | Accounts compromised | Tokens only in environment variables / an untracked `.env`; `.env` in `.gitignore`. Secret-detection hook was not selected (D-84d) and can be added later |
| Scope creep | Nothing finished by internship season | Hard v1 cut; stretch goals stay stretch |
| Privacy / consent | Can't publish data; legal exposure | No own recordings (D-08v2), so D-07 consent applies only if recording ever resumes; no real customer PII unless anonymized |
| Licensing | Can't release a derived dataset or model | Track every source's license in the datasheet |

---

## 8. Open Questions Backlog

Grouped by area. Each one becomes a Decision Log entry once answered. **Only rows marked ✅ are decided** (see §9 for the decision); everything else is open. "Notes" are options or considerations, not choices.

### P — Project & ways of working
| ID | Question | Notes |
|---|---|---|
| P-01 | Project name / codename? | ✅ decided → §9 |
| P-02 | Weekly time budget? | Drives scope; ✅ decided → §9 |
| P-03 | Target date for v1? | Internship applications typically open Jan–Mar; ✅ decided → §9 |
| P-04 | Publication ambition? | HF model + dataset cards / technical blog / workshop paper (e.g. Arabic NLP venues) / demo video; ✅ decided → §9 |
| P-05 | Confirm success metrics and targets (§2)? | ✅ decided 2026-11-22 → §9 (P-05a) |
| P-06 | End-state: portfolio only, open-source + consulting, or SaaS product? | Affects licensing, multi-tenancy, auth; ✅ decided → §9 |
| P-07 | Will anyone else collaborate (friends for recordings/annotation, a professor)? | ✅ decided → §9 |

### D — Repository & workflow
| ID | Question | Notes |
|---|---|---|
| D-01 | Use git at all? Hosted where? | GitHub / GitLab / none; ✅ decided → §9 |
| D-02 | Public from day one or private until v1? | ✅ decided → §9 |
| D-03 | Monorepo or multiple repos? | ✅ decided → §9 |
| D-04 | Branching / commit conventions? | trunk-based vs. GitFlow; Conventional Commits or not; ✅ decided → §9 |
| D-05 | Task tracking? | GitHub Projects / Issues / Notion / none; ✅ decided → §9 |
| D-06 | Architecture Decision Records (ADRs) alongside this file? | ✅ decided → §9 |
| D-07 | Privacy policy for audio (consent, PII redaction, retention)? | ✅ decided 2026-11-21 → §9 |
| D-08 | How do we get domain audio? | Record role-played calls with friends / partner with a company / public content (licensing!) / synthetic TTS / mix; ✅ decided → §9 (D-08 superseded by D-08v2) |
| D-09 | Annotation tool? | Label Studio / Argilla / custom / spreadsheets |
| D-10 | **Transcription convention for code-switching** | Arabic script for Darija + Latin for French/English? All Arabic script? Arabizi? Must match or map to public datasets; ✅ decided → §9 |
| D-11 | Size targets for domain train / test sets (hours)? | Obsolete: nothing is recorded (D-08v2); sizes follow from the chosen sources (D-12c) |
| D-12 | Which public datasets to include? | ✅ decided 2026-11-21 → §9 (LinTO for Phase 1–3); extended 2026-11-22 (D-12c, D-12d, D-12e) |
| D-17 | Which dev set does Phase 3 tune on? | ✅ decided 2026-11-23 → §9 (D-17a) |
| D-10b | Extend the transliteration lexicon to all ~10k Latin words (v1 has the 2,169 used ≥ 5 times, D-16c)? | Quality upgrade; the missing words are listed by `scripts/latin_words.py` in `outputs/lexicon/candidates.tsv`; decide after Phase 3's first results |
| D-10c | Strip Arabic diacritics (harakat) from training targets? | 64% of sampled Deep Confessions transcripts are fully diacritized, other sources mostly not; scoring already removes them (`normalize_arabic`), training targets don't yet. ✅ decided 2026-11-23 → §9 (D-21c: stripped) |

### B — Compute & environment
| ID | Question | Notes |
|---|---|---|
| B-01 | GPU source and monthly budget? | Kaggle/Colab free / Colab Pro / rented (RunPod, Vast, Lambda…) / university GPU / local GPU; ✅ decided → §9 |
| B-02 | Local machine specs (OS, RAM, GPU)? | ✅ decided → §9 |
| B-03 | Python environment tooling? | uv / poetry / conda / pip+venv; ✅ decided → §9 |
| B-04 | Where do large files live (audio, checkpoints)? | HF Hub / S3-compatible (R2, B2, MinIO) / Drive; ✅ decided → §9 |
| B-05 | How is torch installed (CPU vs CUDA)? | ✅ decided 2026-11-21 → §9 |

### ASR
| ID | Question | Notes |
|---|---|---|
| D-20 | Base model candidates to benchmark? | ✅ decided 2026-11-21 → §9 |
| D-21 | Fine-tuning method? | ✅ decided 2026-11-23 → §9 (D-21a base model, D-21b LoRA) |
| D-22 | Inference engine? | faster-whisper (CTranslate2) / whisper.cpp / HF Transformers / ONNX / TensorRT. Benchmarks ✅ decided (D-22a: HF Transformers); serving still open |
| D-23 | Diarization approach? | pyannote / NeMo / WhisperX pipeline / channel split |
| D-24 | Streaming ASR in v1 or batch only? | |
| D-25 | Telephone-quality (8 kHz) robustness as an explicit goal? | ✅ decided 2026-11-22 → §9 (codec: D-25a) |
| D-27 | Train the first LoRA (`configs/phase3/lora.yaml`) for more epochs? | Open, deferred 2026-11-26. Dev WER was still falling at the end of epoch 3 (27.4% / CER 11.1% at step 9,033, the best checkpoint), so more epochs may help. Its test WER (2026-11-26): test-general 26.6%, test-domain 31.1%, telephone 35.2%; ΔWER vs FarukSTT −10.6, −7.1 and −5.6 points, every paired 95% CI below 0; telephone CER 15.7% vs FarukSTT's 14.5% (CIs overlap). It can continue from where it stopped: copy `checkpoints/phase3/lora-r32/checkpoint-9033` (adapter + optimizer + scheduler + RNG + trainer state) to a new output dir, point a config that inherits `lora.yaml` at it with a higher `num_train_epochs`, and rerun `scripts/train_asr.py`. The linear schedule is recomputed for the new length, so the learning rate jumps from 0 back to part of its peak (~40% for 5 epochs): record the run as "3 + N epochs, continued", not as an N-epoch run. **Keep `checkpoint-9033`**: it is the only copy with optimizer state (`best/` holds the adapter only) |

### NLP
| ID | Question | Notes |
|---|---|---|
| D-30 | Label taxonomy (intents, sentiment scale, entity types)? | Ideally modelled on a realistic telecom/bank support domain |
| D-31 | Labelling strategy? | Manual / LLM-assisted pre-labels + human review / weak supervision |
| D-32 | Models to compare? | Prompted large LLM; encoder (e.g. TunBERT, MARBERT, XLM-R family); small LLM + LoRA |
| D-33 | Summarization: which model, which output language (Darija / French / English / Arabic)? | |
| D-34 | Text normalization / transliteration module (Arabizi ↔ Arabic)? | |

### Knowledge layer
| ID | Question | Notes |
|---|---|---|
| D-40 | Vector store? | pgvector / Qdrant / other |
| D-41 | Embedding model? | multilingual options; fine-tune for Darija or not |
| D-42 | Reranker? Fine-tuned or off-the-shelf? | |
| D-43 | LLM for RAG answers: hosted API or self-hosted? Which? | |

### Serving & backend
| ID | Question | Notes |
|---|---|---|
| D-50 | API framework? | FastAPI / NestJS (you know both) / other |
| D-51 | Async job processing? | Celery / RQ / Arq / Temporal + Redis/RabbitMQ |
| D-52 | Model serving? | Plain FastAPI workers / Triton / Ray Serve / BentoML / vLLM (for LLMs) |
| D-53 | Primary database? | PostgreSQL / other |
| D-54 | Auth & multi-tenancy in scope? | |
| D-55 | Schema format for CallRecord? | Pydantic / JSON Schema / Protobuf |

### Interfaces
| ID | Question | Notes |
|---|---|---|
| D-60 | Dashboard stack? | React (you know it) / Next.js / Streamlit / Gradio |
| D-61 | Public demo hosting? | HF Spaces / own VPS / none |

### Evaluation
| ID | Question | Notes |
|---|---|---|
| D-70 | Eval harness: custom or built on a framework? | ASR metrics ✅ decided (D-70a); rest open |
| D-71 | Run evals in CI? Which ones block a merge? | |
| D-72 | LLM-as-judge for summaries/RAG, calibrated against human ratings? | |
| D-73 | Statistical rigor: confidence intervals / bootstrap on WER deltas? | ✅ decided → §9 |

### MLOps & infra
| ID | Question | Notes |
|---|---|---|
| D-80 | Experiment tracking? | W&B / MLflow / none; ✅ decided → §9 |
| D-81 | Data & model versioning? | ✅ decided 2026-11-21 → §9 |
| D-82 | Containerization? | Docker + compose / k8s (kind/k3s) / none |
| D-83 | CI/CD platform? | GitHub Actions / GitLab CI / none; ✅ decided → §9 |
| D-87 | Config system for training? | Plain YAML / Hydra / other; ✅ decided → §9 |
| D-88 | Data versioning beyond HF Hub revisions? | ✅ decided 2026-11-21 → §9 (with D-81) |
| D-84 | Code quality tooling? | ruff, mypy/pyright, pre-commit, pytest coverage targets; ✅ decided → §9 |
| D-85 | Observability? | Prometheus + Grafana, OpenTelemetry, Langfuse (LLM traces) |
| D-86 | Deployment target? | VPS (Hetzner etc.) / cloud (AWS/GCP) / HF / local only |

### Stretch — voice agent
| ID | Question | Notes |
|---|---|---|
| D-90 | Include it at all? | |
| D-91 | TTS approach? | Fine-tune an open TTS on Darija / French-only TTS / none |
| D-92 | Real-time transport? | WebRTC (e.g. LiveKit) / WebSocket |

---

## 8b. Working Agreements (rules for anyone working on this repo, humans and AI assistants)

1. **Ask, don't assume.** Any choice not recorded in the Decision Log (§9) is asked about before it's implemented, even small ones (tools, libraries, file layout).
2. **Tools do tool work (W-01).** Formatting and lint fixes are done by Ruff, never by hand or by an AI rewriting code. The AI runs the tool and supervises the result.
3. **Every decision is logged** in §9 the moment it's made; superseded decisions are crossed out, not deleted.
4. **Numbers need scripts.** Any metric quoted anywhere must be reproducible from a committed script and config.

---

## 9. Decision Log

Format: `ID | Date | Decision | Reason | Alternatives rejected`

ADRs (`docs/adr/`): [0001 D-10 transcription convention](docs/adr/0001-transcription-convention.md) · [0002 D-80v2 tracking on DagsHub](docs/adr/0002-experiment-tracking-dagshub.md) · [0003 B-01c compute policy](docs/adr/0003-compute-policy.md) · [0004 D-21a ASR base model](docs/adr/0004-asr-base-model.md) · [0005 Phase 1 benchmark protocol](docs/adr/0005-phase1-benchmark-protocol.md) · [0006 D-08v2 Phase 2 data strategy](docs/adr/0006-phase2-data-strategy.md) · [0007 D-21b LoRA fine-tuning](docs/adr/0007-lora-fine-tuning.md)

| ID | Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|---|
| — | 2026-11-20 | Work on the speech project (Darija) now; improved document AI later | Resume already covers document AI; speech adds a new, distinctive area | Doc-AI flagship first |
| B-01 | 2026-11-20 | Compute = local RTX 4070 Laptop (8 GB VRAM) + Google Colab free (T4, 16 GB). No paid GPU for now | Available resources | Paid rented GPUs, university GPU |
| D-01, D-02 | 2026-11-20 | GitHub, public from day 1 | Visible build-in-public history for recruiters | Private until v1, GitLab, no repo |
| P-03 | 2026-11-20 | No fixed deadline; build it properly. Phases stay as the quality bar, not a timeline | Owner's choice: depth over speed | Internship-season v1 cut |
| P-04 | 2026-11-20 | Publish: HF model(s) + dataset, demo video, and a technical report (PDF / blog, possibly arXiv) linked from a LinkedIn post | Maximum credibility | — |

| P-02 | 2026-11-20 | Time budget: about 14–20 h/week normally, ≤8 h/week during exams. Quality matters more than speed | Owner is a student | — |
| P-06 | 2026-11-20 | End-state: portfolio + open source. No multi-tenancy or billing; a permissive license (exact license TBD) | Focus on credibility and reach | Open core, SaaS |
| ~~D-08~~ | 2026-11-20 | **SUPERSEDED by D-08v2.** ~~Domain data strategy (owner delegated this to Claude): **(1) Core:** scripted, role-played support calls recorded under signed consent; the only source for the published domain **test set** and the main domain train set. **(2) Train-only augmentation:** telephone-band simulation, noise, speed; synthetic TTS speech only as an ablation, kept only if it measurably helps. **(3) Opportunistic:** if a company shares real calls, use them as a *private* extra eval set and never publish them. **No scraped web audio** in the published dataset (licensing)~~ | Keeps the dataset fully publishable and the test set clean; real-call eval is a bonus, not a dependency | Web-scraped audio as a main source; synthetic-only data |

| D-10 | 2026-11-20 | Transcription convention: **all Arabic script**, including French/English words (transliterated) | Matches LinTO; one script gives the ASR model consistent targets | Arabic + Latin mixed; Arabizi |
| B-02 | 2026-11-20 | Dev machine: i7-13700H (20 threads), 32 GB RAM, RTX 4070 Laptop 8 GB, dual-boot Windows 11 / Ubuntu 24.04 LTS | Owner's hardware | — |
| B-03 | 2026-11-20 | Python tooling: **uv** with a project virtual environment (`.venv`) | Fast, lockfile-based, reproducible | poetry, conda, pip+venv |

| B-02b | 2026-11-20 | Development OS: **Ubuntu 24.04** (Windows not used for this project) | Fewer problems with CUDA, audio libraries and Docker | Windows native, WSL2 |
| B-04 | 2026-11-20 | Large files (audio, datasets, checkpoints) live on the **Hugging Face Hub**; git holds only code and configs | Free, versioned, and where we publish anyway | DVC, S3-compatible storage, Drive |

| ~~D-80~~ | 2026-11-20 | ~~Experiment tracking: MLflow, self-hosted~~ **SUPERSEDED by D-80v2** | Own the infrastructure; shows MLOps skill | W&B, none |
| B-04b | 2026-11-20 | Hugging Face dataset and model repos stay **private until checked** (consent forms, transcript quality, licenses), then made public | Voice data is hard to take back once published | Public from day 1 |

| B-01b | 2026-11-20 | Compute policy: **laptop RTX 4070 is the main machine** for everything; Colab free / Kaggle used **only as overflow** for jobs that don't fit in 8 GB (e.g. Whisper large-v3 fine-tuning). Training code must run unchanged in both places | Laptop is faster and always available; cloud only adds memory | Laptop only; deciding after benchmarks |

| ~~D-80a~~ | 2026-11-20 | **SUPERSEDED by D-80v2.** ~~Cloud (Colab/Kaggle) runs log to a local folder, synced via the HF Hub, then **imported into the laptop's MLflow** afterwards~~ | Cloud runs are rare; simplest setup | Tunnel, always-online server |
| ~~D-80b~~ | 2026-11-20 | **SUPERSEDED by D-80v2.** ~~MLflow run records stored in PostgreSQL running in Docker~~ | Production-style setup; Postgres is already familiar | SQLite |

| D-80v2 | 2026-11-20 | **Experiment tracking = MLflow hosted on DagsHub.** Laptop and Colab runs both log to the same DagsHub MLflow server using the standard MLflow client | One tracker for every machine with live logging, off-site backup, public experiment links, nothing to maintain; logging needs only KB of upload | Self-hosted MLflow + Postgres + S3 store (no off-site backup, Colab import step); W&B |
| D-80c | 2026-11-20 | **Artifact policy:** small artifacts (configs, plots, sample predictions, eval reports, LoRA adapters ≤ ~50 MB) go to DagsHub/MLflow. **Full checkpoints, datasets and base models stay local** and are never uploaded from home | Upload speed is 0.68 Mbps | Uploading checkpoints from home |
| D-80d | 2026-11-20 | **Logging resilience:** async + batched MLflow logging so training never waits on the network; retries with a local disk buffer that syncs when the connection returns. Training must never fail because logging failed | Latency 450–1,000 ms, unstable connection | Synchronous logging |
| B-01c | 2026-11-20 | **Compute policy (confirmed):** all training and evaluation on the laptop RTX 4070. **Colab/Kaggle only** for (a) jobs needing > 8 GB GPU memory, (b) jobs needing data-center bandwidth (e.g. process a 13 GB public dataset and push the clean version to the Hub; merge + publish full models) | Laptop is faster and always available; cloud makes up for the 8 GB limit and the slow home connection | Cloud-first (Colab + DagsHub for everything) |
| N-01 | 2026-11-20 | **Network rules:** home connection measured at ~17.6 Mbps down / **0.68 Mbps up**, latency 447 ms idle → 1,016 ms while uploading. (1) Never upload > ~50 MB from home; bigger transfers go through GitHub Actions or Colab. (2) Large downloads happen once, overnight, with a speed limit. (3) Own recordings are uploaded once (mobile data / university / overnight with a speed limit) | Large uploads make the connection unusable | — |
| S-01 | 2026-11-20 | **Disk:** owner will free up space for a local data cache, base models and checkpoints (target ~50–80 GB, to confirm). Keep only the best + latest checkpoint per run; deleted automatically | Training is local | Cloud-only storage |

| S-01b | 2026-11-20 | Disk space: not a constraint; owner manages it | Owner's call | — |
| N-02 | 2026-11-20 | Connection is **slow but not dropping**. Local log buffering (D-80d) stays as a safety net, not a main feature | Owner's report | — |
| B-04c | 2026-11-20 | **Models go public earlier** (once they beat a baseline on the frozen test set); the **dataset stays private until checked** (B-04b still applies to data) | Build-in-public visibility for models | Everything private until checked |
| P-01 | 2026-11-20 | Working codename: **`tuncall`** (placeholder, can be renamed later) | Owner asked for a placeholder | — |
| P-08 | 2026-11-20 | Code license: **MIT**. Dataset license decided separately (e.g. CC BY 4.0) | Permissive, matches P-06 | Apache-2.0 |

| D-87 | 2026-11-20 | Training config: **plain YAML files validated with Pydantic** (unknown or misspelled keys and wrong types are errors). The full resolved config of every run is logged to MLflow | Simple, explicit, easy to debug | Hydra, OmegaConf |

| D-03 | 2026-11-20 | **Monorepo:** data, training, eval, serving, app and docs in one GitHub repo | Changes that span components happen in one commit; one place to show the project | Multiple repos |
| D-04 | 2026-11-20 | **Trunk-based development:** `main` always works; short-lived branches → PR → CI green → merge. **Conventional Commits** for all commit messages (`feat(asr): …`, `fix(eval): …`) | Modern standard; readable history; changelogs can be automated | GitFlow; free-form messages |

| D-05 | 2026-11-20 | **Task tracking: GitHub Issues + Projects board.** Labels (`area:*`, `type:*`, optional `prio:*`), one milestone per phase, issue templates for Feature / Bug / Experiment, PRs close issues via `Closes #N`, board automation. **Phase-level plan stays in PROJECT.md §5** (status + milestone link); no separate ROADMAP.md | PR-linked, visible planning; one source for the phase plan | TASKS.md only; ROADMAP.md + Issues |

| D-84a | 2026-11-20 | **Ruff** for linting and formatting (replaces flake8/isort/black) | Fast, one tool | — |
| W-01 | 2026-11-20 | **Rule for AI assistants:** an AI assistant must never lint or format code by hand. It runs `ruff check --fix` / `ruff format`, reviews the output, and fixes only what Ruff can't fix automatically. Tools do tool work; the AI supervises | Don't waste effort on mechanical work | Manual formatting by the AI |

| D-83 | 2026-11-20 | **CI = GitHub Actions** running on every PR: (1) Ruff lint + format check, (2) pyright type check, (3) pytest unit tests, (4) CPU smoke test (train + eval a few steps on a tiny audio sample with a tiny model), (5) `uv.lock` consistency check. Merge to `main` only when all are green | Keeps `main` always working; runs in the cloud, so no upload cost | Fewer checks |
| D-84b | 2026-11-20 | Type checker: **pyright** | Faster and stricter | mypy, none |
| D-84c | 2026-11-20 | Tests: **pytest**. **No fixed coverage target for now**; revisit later | Owner's call | 80% enforced target |
| D-06 | 2026-11-20 | **ADR files** (`docs/adr/NNNN-title.md`: context, options, decision, consequences) only for **major** decisions (costly to reverse or needing justification in the report or an interview). PROJECT.md §9 keeps the one-line log for all decisions and links to ADRs. First ADRs to write: D-10 transcription convention, D-80v2 tracking on DagsHub, B-01c compute policy | Shows reasoning without paperwork for every small choice | PROJECT.md only; ADR for every decision |

| D-84d | 2026-11-20 | **pre-commit hooks** on the laptop: Ruff (lint + format), pyright, large-file blocker, basic file hygiene (trailing whitespace, end-of-file newline, YAML/TOML syntax, merge-conflict markers, Unix line endings, case conflicts). **Not selected:** secret detection, notebook output stripping | Catch mistakes before pushing; protect repo size and upload | — |
| D-03a | 2026-11-20 | **Code layout:** one package `src/tuncall/` with subpackages per component, one `pyproject.toml` / `uv.lock` (§6). Split into a uv workspace later only if serving needs it | Least setup; cross-component changes stay simple | uv workspace with one package per component |
| B-03a | 2026-11-20 | **Python 3.12** (`.python-version`, `requires-python = ">=3.12,<3.13"`) | Matches Ubuntu 24.04 and Colab; supported by the ML/audio stack | 3.11 |
| D-84e | 2026-11-20 | Large-file blocker threshold: **500 KB** (`check-added-large-files --maxkb=500`). Anything bigger goes to the HF Hub (B-04) | pre-commit default; nothing large belongs in git | — |
| D-07 | 2026-11-21 | **Privacy:** signed consent per speaker covering recording, training and publication of the audio under the dataset license. Call scripts use **only fictional personal data** (names, phone numbers, IDs), so no real PII is spoken. Speakers can withdraw until the dataset is made public. Only optional coarse metadata is kept (gender, age bracket, region). Audio stays private on the HF Hub until checked (B-04b) | Keeps the dataset publishable (P-04) with no PII to redact; a voice can't be anonymized, so consent must cover publication explicitly | Audio never published (models only) |
| D-20 | 2026-11-21 | **Zero-shot baselines (Phase 1):** Whisper large-v3, large-v3-turbo, medium, small, and FarukSTT | Shows size vs. accuracy and the starting point for fine-tuning; FarukSTT is the strongest public model to beat | CTC models (w2v-BERT 2.0, MMS, wav2vec2) |
| D-21a | 2026-11-21 | **Fine-tuning starts from Whisper large-v3-turbo** (OpenAI, MIT, 809M). The model is the owner's own: **FarukSTT is a baseline only, never a starting checkpoint**. Full fine-tune vs. LoRA stays open (D-21) | Demonstrates our own training rather than reusing someone's fine-tune; multilingual decoder (Arabic + French + English) suits code-switching; LoRA fits the 8 GB laptop GPU (B-01c); fast serving via faster-whisper | Whisper large-v3 (8 GB too tight, most runs would move to Colab); w2v-BERT 2.0 + CTC; two tracks (turbo + CTC); continuing from FarukSTT |
| D-12 | 2026-11-21 | **Public data for Phase 1–3: LinTO audio** (`linagora/linto-dataset-audio-ar-tn`, CC BY 4.0, ~81 h labelled, already all Arabic script). **test-general = LinTO test split** (~1.3 h), frozen after overlap/dedupe checks. Training sources can be added later as Phase 3 data-mix ablations (each needs a license check, D-10 conversion and dedupe against the test sets); **test sets never change**. Text corpora deferred to Phase 5 | Clean license, matches D-10, fully reproducible, keeps the FarukSTT comparison clean | FARUKxAUTO/tunisian-asr-cleaned (no license stated; FarukSTT's own training data); IWSLT Tunisian / LDC2022E02 (LDC agreement, not redistributable, not reproducible by readers); TunSwitch separately (already included in LinTO) |
| D-81, D-88 | 2026-11-21 | **Versioning = HF Hub revisions only.** Each dataset version is a tagged Hub commit (e.g. `v0.1`); every MLflow run records the Hub repo + revision hash; test-set file hashes are committed to git (§5b rule 5). No DVC | One tool, already where data lives (B-04); revision hashes are exact and immutable | DVC with an HF/S3 remote; lakeFS |
| D-87a | 2026-11-21 | **Config loader details:** YAML parsed with **PyYAML** (`safe_load`); a config may name a parent with `base: <relative path>`, which is **deep-merged** (nested mappings merge key by key, lists and other values are replaced); inheritance cycles are errors; config models subclass `StrictModel` (`extra="forbid"`) | PyYAML was already installed; deep merge lets run configs list only what they change | ruamel.yaml; section-level replace |
| D-80e | 2026-11-21 | **Tracking client:** `mlflow-skinny` (official client-only MLflow package). `tuncall.tracking.Tracker` queues all writes and sends them from **one background thread** with `MlflowClient.log_batch`; failed batches go to a local JSON-lines buffer and are replayed on the next successful flush; every error is logged, never raised. Artifacts over 50 MB are refused. Each run is tagged with git commit (+`-dirty`), dataset version and machine (auto-detected) | Meets D-80d fully; skinny client avoids the MLflow server/UI dependency tree | Full `mlflow` package; MLflow built-in async logging (no disk buffer, errors can surface at run end) |
| P-07 | 2026-11-21 | **Solo for now.** Recruiting speakers for role-played calls is decided in Phase 2 | Owner's answer | — |
| P-05 | 2026-11-21 | **Success targets (§2) are set at the end of Phase 1**, from the measured baseline table. Phase 0 closes without them | Targets before any measurement would be guesses | Numeric targets now |
| P-05a | 2026-11-22 | **§2 targets from the Phase 1 baselines:** test-domain WER ≥ 20% lower than FarukSTT (relative) with the paired ΔWER 95% CI below 0; test-general WER ≤ 40% and CER ≤ 17% (match FarukSTT); ≤ 10/1013 looping utterances on test-general. Diarization, NLP and system targets are set when their baselines exist (Phases 4–6) | FarukSTT (38.8% WER) is the strongest public model; base turbo (144%) is too easy a bar to mean anything; matching rather than beating on test-general because FarukSTT may have seen 3% of it | 10% or 30% domain margin; beating FarukSTT on test-general; base Whisper as the bar |
| D-13 | 2026-11-21 | **Phase 1 freezes test-general only** (LinTO test split). test-domain needs our own recordings, so it is frozen in Phase 2 and added as a column to the same baseline table by the same command | test-domain can't exist before Phase 2 | Recording a pilot domain set in Phase 1 |
| D-14 | 2026-11-21 | **Freezing test-general:** configs pin LinTO's Hub commit (`99796bfd5e8b48febaac60633d1ec97a658fa511`); `configs/data/test-general.yaml` records the sha256 of each test parquet plus a per-utterance digest; eval refuses to run on any mismatch | Exact and provable with no upload from home (N-01) | Mirroring the split to our own HF repo |
| D-22a | 2026-11-21 | **Phase 1 baselines run on HF Transformers** (torch). D-22 (serving engine) stays open | Loads every candidate as-is, incl. FarukSTT; same library as Phase 3 training, so before/after numbers come from the same code | faster-whisper (needs CTranslate2 conversion; output can differ) |
| D-26 | 2026-11-21 | **Benchmark decoding:** `language="ar"`, `task="transcribe"`, `num_beams=5`, temperature 0 only (no fallback) | Deterministic; matches the Whisper paper's setup; forcing Arabic stops language-ID errors and translation | Greedy decoding |
| D-70a, D-73 | 2026-11-21 | **ASR metrics:** WER/CER with **jiwer** after our own Arabic normalizer (`tuncall.eval`, D-10 rules: alef/hamza, ta marbuta, alef maqsura, diacritics, tatweel, digits, punctuation, whitespace). **95% bootstrap CIs**, 1000 resamples over utterances; **paired** bootstrap for model-vs-model deltas. Raw (unnormalized) WER is reported for reference | Small dependency, rules under our control and tested; CIs are required by D-12 (test-general is ~1.3 h) | HF `evaluate` + pyarabic/camel-tools |
| D-83a | 2026-11-21 | **CI smoke test postponed to Phase 3** (amends D-83 item 4). All real runs are on the laptop; the benchmark script has `--limit N` / `--models` for a quick local pre-flight | Low value with no training code; keeps CI fast | Eval-only smoke test in Phase 1; dropping it |
| D-15 | 2026-11-21 | **Phase 1 defaults:** model revisions pinned by Hub commit sha in configs; FarukSTT downloads root weights only (never `last-checkpoint/`); per-model predictions and reports go to MLflow via `Tracker`; the only results in git are the README table, rewritten by the benchmark script between `<!-- baselines:start/end -->` markers. New runtime deps: `torch`, `transformers`, `huggingface_hub`, `pyarrow`, `soundfile`, `jiwer` (no `datasets`: we read the exact files we hash) | Reproducible to the byte; git stays code + configs | `results/` folder in git; `datasets` library |
| D-14a | 2026-11-21 | **test-general utterances:** one utterance = one LinTO timestamped segment (recordings run up to ~3 min; segments ≤ 15 s), reference = the segment's cleaned `transcript` field; stereo averaged to mono; resampled to 16 kHz with **soxr** (source audio is 16 / 44.1 / 48 kHz). Result: 1013 utterances, 1.239 h | Whisper takes ≤ 30 s; `transcript` is LinTO's cleaned text; soxr is small, deterministic, high quality | Whole recordings; `transcript_raw`; torchaudio; scipy |
| B-01d | 2026-11-21 | **Phase 1 baselines (#19) run on Colab** (T4), under B-01c(b): ~15 GB of model downloads need data-center bandwidth. Same script and config as on the laptop; runs are tagged `machine=colab`; the README table comes back as a small file and is committed through a normal PR. Determinism checks (rerun → identical WER) are done on the same machine. The laptop installs torch/transformers later for Phase 3 | Saves ~2 h of home download and ~15 GB of disk for one-off runs | Laptop run (2 h download, disk 21 GB → ~3 GB free) |
| B-05 | 2026-11-21 | **torch packaging:** `torch` only through two conflicting optional extras, using uv's PyTorch-index pattern. `cpu` uses the PyTorch CPU index and is installed by CI and the laptop (~200 MB, so pyright can resolve imports). `gpu` uses the CUDA wheels (`cu126` index) and is used on Colab. `transformers` goes in the main dependencies (small). Install with `uv sync --extra cpu` (laptop/CI) or `uv sync --locked --extra gpu` (Colab) | Laptop disk is tight (B-01d) and CI shouldn't download ~3 GB of CUDA libs per PR | torch in main deps; one CUDA-only optional group |
| D-15a | 2026-11-21 | **Benchmark layout and defaults:** inference in `src/tuncall/asr/transcribe.py` (reused in Phase 3); orchestration, scoring and the table in `scripts/benchmark_asr.py`. Predictions are one JSONL per model with one line per utterance `{id, ref, hyp, ref_norm, hyp_norm, word_errors, word_len, char_errors, char_len}`, also logged to MLflow. `batch_size: 16` (config), `max_new_tokens: 440`. MLflow experiment `phase1-baselines`, run name = model key. README columns: model, WER [95% CI], CER [95% CI], ΔWER vs FarukSTT [paired 95% CI], raw WER, sorted by WER. Speed/RTF numbers and the per-language error breakdown are left to later phases | Owner accepted the proposed defaults | Everything in the script |
| D-15b | 2026-11-21 | **Resume:** each model's predictions are written to `<out>/<model>.jsonl` (write `.tmp`, then rename, only once the model has finished). A model whose file exists is skipped unless `--force`. The table is built from the cached files. `--out` (default `outputs/phase1`) can point at Google Drive on Colab. `--limit N` runs write to `<out>/limit-N/` and never touch the README | Colab sessions die; a 1.5–2 h run shouldn't restart from zero | No resume |
| D-15c | 2026-11-21 | **Benchmark `batch_size: 4` for every model** (amends D-15a's 16). One value for all models | With beam 5, each sequence keeps its own cross-attention cache: at 16 that is ~20 GB for whisper-large-v3 (and FarukSTT if large-sized), over a T4's 15 GB; at 4 it is ~5 GB. Batch size changes memory and speed, not transcripts (clips are padded to 30 s, beams are independent). Run takes longer (~2.5–3 h) | Per-model batch sizes (16 for small models); keeping 16 |
| D-15d | 2026-11-22 | **Baseline table gets a "Looping" column:** number of utterances whose normalized hypothesis has **> 20 words with < 30% distinct** (decoder stuck repeating). Looping utterances stay in every score; decoding (D-26) is unchanged | Zero-shot Whisper loops on ~6% of test-general and those clips carry 55–58% of its errors, pushing WER over 100%; the column explains that without touching the frozen test set or tuning decoding on it | Temperature fallback (changes D-26 after seeing test results, less reproducible); dropping looping clips (different test set per model); publishing unexplained |
| D-12a | 2026-11-21 | **#18 (LinTO train/test overlap) moves to Phase 3.** It only filters training data, and test-general can't change whatever it finds | Nothing is trained in Phase 1 | Doing it in Phase 1 |
| D-12b | 2026-11-21 | **FarukSTT contamination is measured in Phase 1.** Each test-general reference is normalized with `normalize_arabic`. It counts as "seen" if it has **≥ 4 words** and appears **exactly** among FarukSTT's normalized training transcripts (`FARUKxAUTO/tunisian-asr-cleaned`, pinned by Hub commit sha). Report the % of eligible utterances seen, how many were eligible, and the total. The script runs on Colab | Turns "may be contaminated" into a number next to FarukSTT's score; the 4-word minimum stops common short replies (ايه, باهي) from counting as matches by chance | Note only; any-length matches |
| D-08v2 | 2026-11-22 | **Domain data strategy (replaces D-08): no own recordings and no synthetic speech.** Phase 2 builds a multi-source training mix (D-12c, D-12d) and a frozen task-oriented test set (D-13a) from public Tunisian speech. A real call test set (TARIC, LDC2025S05, a company's calls) can be added later as a new column. D-07 (consent) and P-07's speaker recruiting apply only if recording ever resumes. ADR 0006 | Recording ~120 role-played calls with ~20 volunteers isn't feasible for the owner; no public Tunisian call-center audio exists (searched the HF Hub, the LDC catalog and recent papers) | Role-played recordings (D-08); TARIC-SLU (railway-desk dialogues, CC BY-NC 4.0; download site down, access means emailing the authors); LDC2025S05 (210 h Tunisian telephone conversations; LDC membership or fee, not redistributable; not pursued, no access expected) |
| D-08b | 2026-11-22 | **Synthetic speech dropped.** Pilot kept for reference: Habibi-TTS Unified (`SWivid/Habibi-TTS`, F5-TTS, dialect ID `TUN`, CC BY-NC-SA 4.0) spoke Tunisian support-call lines the owner judged good enough, at RTF 0.26 and 2 GB VRAM on the RTX 4070. Pilot files and downloads deleted | Its only purpose was call-center vocabulary; without a call test set its effect can't be measured, and real data grew to ~140 h clean + ~300 h noisy | 10 h synthetic ablation with LLM-written dialogues; other Tunisian TTS (silma-tts-derja, Tunisian XTTS fine-tunes: smaller or gated, all non-commercial) |
| D-12c | 2026-11-22 | **Core training sources** (each read through an adapter, D-16; only train splits are trained on): **LinTO** (D-12, ~81 h); **SLURP-TN train** (`Elyadata/SLURP-TN`, CC BY-NC-ND 4.0, 2,677 utterances, task-oriented requests by 55 speakers); **Omnilingual ASR corpus, Tunisian only** (`facebook/omnilingual-asr-corpus`, config `aeb_Arab` and nothing else, CC BY 4.0, 2,925 spontaneous answers, ~20 h estimated); **TEDxTN** via the segmented mirror `Hediske/tedxtn-tunisian-segmented` (108 TEDx talks, ~25 h, professional transcripts; upstream `fbougares/TEDxTN` annotations are CC BY-NC-ND 4.0 and ship without audio); **senior_tunisian_voice** (`Fares11/senior_tunisian_voice`, Apache-2.0 as stated, 5,377 records). Owner listened to TEDxTN and senior_tunisian_voice and judged both good quality | Owner's goal is a general Darija model of our own; more varied clean speech (~140 h total) | TuniSpeech-21h (transcripts + YouTube/Facebook links only); TunArTTS (3 h, one speaker reading dictionary entries); TuniFra (no download found); Bulbul (0.11 h Tunisian); OpenSLR 46 (MSA read speech); FarukSTT's training data (no license, D-12) |
| D-12d | 2026-11-22 | **Noisy-label sources are ablations only:** **Deep Confessions** (`oddadmix/arabic-audio-collection-tunisian-deep-confessions`, ~175 h podcast, AI-assisted transcripts, license "other") and **TunSwitch weakly labelled** (Zenodo 8370566 `weakly_labeled_tn.zip`, CC BY 4.0, automatic labels). Phase 3 trains once with and once without them; they stay only if WER on the real test sets improves. Both are downloaded and shrunk **on Colab** (16 kHz mono, compressed, non-speech tags such as `<laugh>` stripped) and pushed to **private derived HF repos of ours** that record the upstream revision (B-01c b) | Machine transcripts can teach the model another system's errors; 36 GB can't be downloaded from home (N-01) | Adding them to the core mix; skipping them; downloading them at home |
| D-12e | 2026-11-22 | **TuDiCoI** (SNCFT railway customer-service dialogues, text only) is used **only in Phase 5** (NLP), after checking the original corpus's license | Text only, no audio; a second domain adds little to ASR | TTS scripts from it |
| D-13a | 2026-11-22 | **test-domain = SLURP-TN test split** (893 utterances, ~1 h, real speakers making task-oriented requests; amends D-13). Its README label is "task-oriented requests", not "support calls". Caveat: SLURP-TN's Hub data has no speaker IDs, so speaker overlap with SLURP-TN train (D-12c) can't be ruled out; it's stated next to the score | The only real task-oriented Tunisian speech available; owner delegated the choice | LDC2025S05 test (no access); TARIC test (no access) |
| D-14b | 2026-11-22 | **Freezing test-domain:** the D-14 recipe: config `configs/data/test-domain.yaml` pins SLURP-TN's Hub commit, records the sha256 of each test parquet and a per-utterance digest of the **transliterated** references (D-10a); the transliteration lexicon's hash is recorded too; eval refuses to run on any mismatch | Same guarantees as test-general; the references we score against are the transliterated ones | Mirroring the split to our own repo |
| D-10a | 2026-11-22 | **Latin → Arabic-script transliteration lexicon:** every unique Latin-script word in sources that write French/English in Latin (SLURP-TN: 38 of 40 sampled test rows; TEDxTN: about half; senior_tunisian_voice) is mapped to Arabic script (e.g. `mail` → `ميل`) in one committed lexicon file, **reviewed once by the owner**, and applied identically to train and test | Keeps D-10 (all Arabic script, matching LinTO and the model's targets); the step D-10's implications anticipated | Mixed-script D-10 (inconsistent targets vs LinTO); script-insensitive scoring (hard to explain) |
| D-16 | 2026-11-22 | **Source adapters:** one module per source in `src/tuncall/data/sources/`, each yielding the same record `{id, source, split, audio (float32 16 kHz mono), text (D-10), speaker (optional), duration}`. Adapters **read each source at a pinned upstream revision**; only data we derive ourselves (D-12d) gets an HF repo of ours. The training mix is a YAML list of `{source, revision, weight}`. A **dedupe report** (normalized-text match) measures overlap between sources and against both frozen test sets before any training | Adding a source (e.g. LDC later) is one adapter + one config line; no re-hosting or uploads from home (N-01); exact versioning (D-81) | Re-hosting every source in our own repo; the `datasets` library (D-15) |
| D-25 | 2026-11-22 | **Telephone robustness is an explicit goal.** A telephone-band version of test-general (same frozen utterances through a fixed, seeded 8 kHz codec simulation, reproducible from code) becomes a benchmark column. The codec (G.711, AMR, …) is decided when it's built | Real customer audio is mostly phone audio; costs no new data | No telephone evaluation |
| D-15e | 2026-11-22 | **Benchmark columns for Phase 2:** `scripts/benchmark_asr.py` scores every baseline model on test-general, **test-domain** (SLURP-TN) and **telephone-band test-general** with the same command and metrics (D-70a, D-73, D-15d). FarukSTT's training data is checked for SLURP-TN overlap with the D-12b rule | One table, one command (D-13); contamination stated next to each FarukSTT score | Separate scripts per test set |
| P-05b | 2026-11-22 | **Domain and telephone targets are set from their baselines** (the P-05 rule), once the D-15e columns exist; supersedes P-05a's "≥ 20% below FarukSTT on test-domain". The test-general targets stay | The 20% margin assumed real in-domain training data, which no longer exists | Keeping the 20% margin unmeasured |
| P-09 | 2026-11-22 | **Goal restated:** the end product is an ASR model of our own that understands Tunisian Darija in general (code-switching included), beating the public baselines; the customer-support call domain is no longer the ASR target. Later phases (diarization, NLP, search) are unchanged and re-scoped when they start | Owner's direction; no call data exists | Call-center-only ASR |
| P-04a | 2026-11-22 | **Licensing set aside for now:** sources are chosen for model quality, even under NC/ND licenses (SLURP-TN, TEDxTN). Every source's license is reviewed before any model or dataset release | Owner's call: "worry about publishing later" | Excluding NC/ND sources from training |
| P-10 | 2026-11-22 | **Phase 2 exit criteria** (§5): adapters for every D-12c/D-12d source + the reviewed lexicon (D-10a); dedupe report (D-16); test-domain frozen (D-14b) with the D-15e columns in the baseline table; **`docs/datasheet.md`** following the "Datasheets for Datasets" questions (motivation, composition, collection, preprocessing, uses, distribution), with its numbers generated by a script | Each criterion is checkable; numbers need scripts (8b.4) | "Datasheet written; splits frozen" (written for own recordings) |
| S-01c | 2026-11-22 | **Disk before downloads:** core sources ≈ 14 GB (LinTO 5.7, Omnilingual `aeb_Arab` 2.9, TEDxTN 2.8, senior_tunisian_voice 2.4, SLURP-TN 0.3); 21 GB were free. The owner frees ~50 GB before large downloads start; downloads follow N-01 (once, overnight, rate-limited) | Phase 3 checkpoints need room too | Streaming sources at training time |
| D-16a | 2026-11-22 | **Adapter details (#33):** pinned revisions = each repo's Hub commit on 2026-11-22 (LinTO `99796bf…` as test-general, SLURP-TN `a5c2ec0…`, Omnilingual `8648ba8…`, TEDxTN mirror `8eef268…`, senior_tunisian_voice `670ed90…`), full shas in `configs/data/train-mix.yaml`; all weights 1.0 (every clip once; weight use set in Phase 3). Text fields: LinTO and senior_tunisian_voice `transcript` (cleaned, as D-14a, one record per segment), SLURP-TN `tun_transcription`, Omnilingual `raw_text`, TEDxTN `sentence`; punctuation is left to `normalize_arabic`. Adapters yield the source's text unchanged; the D-10a lexicon (#34) is applied after them, so `Record.text` is D-10 only once #34 lands. Record ids are `<source>/<split>/<upstream key>` (upstream file name for SLURP-TN and TEDxTN, `speaker/prompt/segment` for Omnilingual, `[subset/]audio_id/segment` for LinTO and senior_tunisian_voice); segments are cut exactly as test-general (D-14a). Revisions live only in `train-mix.yaml` (adapters take them as an argument). Per-source hours/utterances come from `scripts/source_stats.py` | Owner accepted the proposed defaults | Weights by quality now; `transcript_raw` |
| D-16b | 2026-11-22 | **Phase 2 data work runs from one Colab export:** `scripts/source_stats.py` runs once on Colab, decodes every split of every core source, prints hours per source/split and writes `outputs/sources/<source>-<split>.jsonl` (`{id, text, duration, speaker}`, a few MB) that the owner downloads. The lexicon word list (`scripts/latin_words.py`, #34) and the dedupe report (`scripts/dedupe_report.py`, #38) run at home from that export, with no audio. **Lexicon:** `configs/data/lexicon.tsv`, one `latin<TAB>arabic` line per word; a Latin word is a run of Latin letters (accents included, French elisions such as `c'est` kept whole), looked up lowercased; digits untouched. **Dedupe:** texts transliterated, then `normalize_arabic`, counted only with ≥ 4 words (D-12b rule); exact match; report in `docs/data/dedupe.md`, matching ids in git-ignored `outputs/dedupe/matches.jsonl`. Home disk is only needed from Phase 3 (training), so S-01c applies then | Phase 2 trains nothing; Colab has the disk and bandwidth, the exported text is small; owner chose this plan | Downloading 14 GB at home now; doing the word list and dedupe on Colab too; YAML lexicon; any-length dedupe matches |
| D-12f | 2026-11-22 | **senior_tunisian_voice dropped from the core mix (amends D-12c).** The Colab export (D-16b) showed it is a copy of LinTO's TunSwitchCS train subset: all 5,377 clips have the same audio id, text and duration. Adapter and mix entry removed | Training on both would count those clips twice; LinTO already has them | Keeping it (double weight for TunSwitchCS) |
| D-12g | 2026-11-22 | **Omnilingual non-speech tags are stripped in its adapter:** every `<…>` tag (`<hesitation>`, `<unintelligible>`, `<noise>`, `<laugh>`, malformed `<<…>` / `…>>`, empty `<>`) and stray `<`/`>` are removed, whitespace collapsed; clips left with no text (51) are skipped. Clips over 30 s (91) are left for Phase 3 | Same rule as D-12d; the model must not learn to write tags | Keeping tags |
| D-16c | 2026-11-22 | **Lexicon scope and test-general:** LinTO is not all Arabic script (D-12 assumed it was): 315 of 1,013 test-general references and 4,842 LinTO train clips contain Latin words. test-general's frozen files and digest stay unchanged; `scripts/benchmark_asr.py` transliterates **both reference and hypothesis** with the lexicon before `normalize_arabic` (config key `lexicon`), records the lexicon's sha256 with every score, and re-scores cached predictions so the table always uses the current lexicon. The Phase 1 README numbers are re-scored when the lexicon is final (#37). **Lexicon v1** covers the 2,169 Latin words used ≥ 5 times (82% of 74,277 Latin tokens); spellings drafted from written rules (seed of 60 by hand, the rest by helper agents); **reviewed and approved unchanged by the owner on 2026-11-22**. Extending it to all ~10k words is a later quality upgrade (D-10b) | Keeps the frozen set untouched while applying D-10a to train and test alike; ≥ 5 uses covers most tokens at a reviewable size | Re-freezing test-general with transliterated references; scoring Latin as-is; all ~10k words now |
| D-16d | 2026-11-22 | **Lexicon v1.1** = v1 + every Latin word still left in the two frozen test sets' references (390 words: 164 from SLURP-TN test, 226 from test-general), drafted with the D-16c rules; both test sets then have 0 Latin tokens. Listed in `outputs/lexicon/review-v1.1.tsv`; **reviewed and approved unchanged by the owner on 2026-11-22**. test-domain is frozen with v1.1 (`configs/data/test-domain.yaml`) | Leftover Latin would count as an error for every model; spelling words is not tuning (the entries apply to training text too) | Freezing with v1 (185 Latin tokens left in test-domain) |
| D-14c | 2026-11-22 | **test-domain storage (implements D-14b):** the test-general `Manifest` and `load_test_set` gain `layout: clips` (one utterance per parquet row, id = the audio file name, e.g. `16501-9`; reference = `text_column`, here `tun_transcription`) and `lexicon: {path, sha256}`; references are transliterated before the digest is taken, and loading refuses a changed parquet file, lexicon or digest. test-general's manifest is unchanged. `scripts/freeze_test_domain.py` writes `configs/data/test-domain.yaml` reproducibly. Any later lexicon change (e.g. D-10b) needs a re-freeze, logged as a new row | One verification path for both test sets; owner chose it | A separate test-domain loader |
| D-25a | 2026-11-22 | **Telephone channel (implements D-25):** a landline simulation in `tuncall.data.telephone`: 16 kHz → 8 kHz (soxr), 300–3400 Hz band (FFT), G.711 A-law 8-bit companding (continuous A-law curve), back to 16 kHz at the same length. No added noise and no seed, so it is deterministic. `scripts/benchmark_asr.py` applies it to any test set marked `channel: telephone` (D-15f). Owner chose both | A-law is Tunisia's landline standard; numpy + soxr only, same output on every machine; no noise keeps the column a pure channel effect | AMR-NB (needs an ffmpeg build with the encoder, output varies by build); both codecs as two columns; white noise at 20 dB SNR |
| D-15f | 2026-11-22 | **Benchmark over several test sets (implements D-15e):** the benchmark config lists `test_sets` (name → manifest + channel): test-general, test-domain and telephone (test-general through D-25a). Predictions are cached in `<out>/<test set>/<model>.jsonl`; `--test-sets` runs a subset, so Colab runs only the missing sets. The README gets a summary table (WER / CER per test set, sorted by test-general) and the full per-set tables. The Phase 1 test-general predictions are reused from MLflow and re-scored with the lexicon. The FarukSTT overlap check covers test-general and test-domain, with the lexicon applied to both sides (D-16c) | One command, one README block; no model re-run for test-general | One config per test set; re-running test-general |
| D-15g | 2026-11-22 | **Laptop batch-size fallback (amends D-15c):** `scripts/benchmark_asr.py --batch-size N` overrides the config, and `scripts/overnight_baselines.sh` runs each model at batch 4, retrying at 2 then 1 if the 8 GB laptop GPU runs out of memory. The batch size used is logged with every run. Models and datasets download to `HF_HOME` (an external drive on the laptop, set in `.env`); the script refuses to run if it isn't mounted | Batch size changes memory and speed, not transcripts (D-15c), apart from possible float rounding; Colab GPU runs kept failing. Owner chose it | Batch 2 for every model |
| D-16e | 2026-11-23 | **Dedupe exclusions (#38):** a train clip is never trained on if its normalized text (D-12b rule: lexicon, `normalize_arabic`, ≥ 4 words) appears in **any test or validation split** of any source. `scripts/dedupe_report.py` writes the ids to `configs/data/exclude.txt` (generated, one id per line), which `configs/data/train-mix.yaml` names with `exclude:`. First run: **49 clips** (46 LinTO train vs test-general, 1 SLURP-TN train vs test-domain, 2 SLURP-TN train vs its validation split). Sentences repeated only across or within train sets are kept (e.g. 7,546 repeated LinTO train texts, 1 LinTO/TEDxTN pair) | Keeps test scores and checkpoint selection free of leaked sentences; a generated file keeps the mix config readable and the list reproducible | Test sets only (validation leak would bias checkpoint choice); also dropping train-only duplicates (same sentence by different speakers is useful training data); the id list inline in train-mix.yaml |
| D-16f | 2026-11-23 | **Datasheet (#40, implements P-10):** `docs/datasheet.md` follows the Datasheets for Datasets questions with hand-written prose; its two tables (per source/split: utterances, hours, mean/max clip length, clips over 30 s, speakers, clips with Latin words, D-16e exclusions; and the training mix after exclusions) are rewritten between `<!-- name:start/end -->` markers by `scripts/datasheet.py` from the D-16b export. The D-12d noisy-label sources get their prose now and a "pending #39" table row until their export exists | One readable file, numbers from a script (8b.4), same marker mechanism as the README | A separate fully generated tables file; leaving the D-12d sources out until #39 |
| D-12h | 2026-11-23 | **Noisy-source preprocessing (#39, implements D-12d):** `scripts/prepare_noisy.py` runs on Colab per source: strips non-speech tags as D-12g (also TunSwitch's `<eps>`), drops clips left empty, **drops broken transcripts** (looping by the D-15d rule, or **> 25 characters per second**, both measured on `normalize_arabic` text because Deep Confessions is often fully diacritized), re-encodes audio as **16 kHz mono 16-bit FLAC** in ~300 MB parquet shards (`key, audio, text, duration`) and pushes **private** repos `mouaidMrz/tuncall-deep-confessions` and `mouaidMrz/tuncall-tunswitch-weak`; each card records the upstream (Deep Confessions Hub commit `73dcdff…`, TunSwitch `weakly_labeled_tn.zip` md5 `2316744…`) and the kept/dropped counts. The script also writes the D-16b text export, so dedupe and the datasheet cover them. Adapters `deep_confessions` and `tunswitch_weak` read the derived repos; they are registered sources but not in the core mix. Pre-flight on 200 real Deep Confessions clips: 2 dropped, both transcriber loops; normalized rate p99 17 chars/s. **Deep Confessions pushed 2026-11-23** at commit `364ea8516d0a109c36bd4f19b24a302f6df9d937`: 37,104 clips kept (174.4 h, 12.95 GB), 52 dropped (44 looping, 7 too dense, 1 empty); no text overlap with any test or other set; 1,225 clips over 30 s, longest 339 s. **TunSwitch weakly labelled pushed 2026-11-23** at commit `932da084fa55e1e33ca3b4e5c32d445ce1c973c1`: 121,104 clips kept (158.7 h, 11.3 GB in 38 shards), 3 dropped (1 looping, 2 too dense); no overlap with any test or validation split; longest 15 s. Its text export was rebuilt from the repo's `key`/`text`/`duration` columns (identical to the script's export) | Lossless audio keeps the ablation about labels, not compression; looping transcripts teach looping; ~68 MB per audio hour (≈ 12 + 11 GB to download at home) | Opus 32 kbps (~5x smaller, lossy); keeping every clip; rate on raw text (drops ~1% good diacritized clips) |
| D-21b | 2026-11-23 | **Fine-tuning method (closes D-21): LoRA on the laptop RTX 4070** with `peft`, through the **HF `Seq2SeqTrainer`** (new deps `peft`, `accelerate`). First run (`configs/phase3/lora.yaml`): rank 32, alpha 64, dropout 0.05 on `q/k/v/out_proj, fc1, fc2` of encoder and decoder; lr 1e-4, 500 warm-up steps, linear decay; effective batch 32 (8 × 4 accumulation); bf16; gradient checkpointing; 3 epochs; ~~dev pass every 1,000 steps~~ (every 500, D-21e); best + latest checkpoint kept (S-01). Trainer logs go to MLflow through a `Tracker` callback (`report_to: none`). Owner chose all | Fits 8 GB, so every run stays on the laptop (B-01c); the adapter is small; the trainer brings checkpoint/resume, accumulation and mixed precision | Full fine-tune on Colab T4 (every run on Colab); frozen encoder + full decoder; own PyTorch loop |
| D-21c | 2026-11-23 | **Training data feed and cleaning (closes D-10c):** `scripts/prepare_asr_data.py` decodes every train/dev clip once into a local cache under `${HF_HOME}/tuncall-cache/phase3` (one 16 kHz int16 file + `index.jsonl`, ~15 GB), read in random order through a memory map. Targets = D-10a lexicon transliteration with **harakat stripped**; **clips over 30 s are dropped**, as are clips left with no text. All mix weights must be 1.0 until weighting is decided | True shuffling each epoch and exact resume; diacritics are scored away anyway and only some sources have them; splitting long clips would need word timings we don't have | Streaming with a shuffle buffer; keeping harakat; splitting long clips |
| D-21d | 2026-11-23 | **Scoring fine-tuned models:** a benchmark model entry may add `adapter: <local dir>`; `asr.transcribe.load` merges that LoRA adapter into the base weights (`merge_and_unload`) before decoding, and the adapter's sha256 is logged with the run. `configs/phase3/benchmark.yaml` (base model, FarukSTT, LoRA models; same test sets, decoding and scoring as Phase 1, cached baseline predictions reused) writes the Phase 3 ablation table into its own README block (`phase3`). Owner chose all | Adapters stay local (111 MB, over the upload limit); merged weights are what serving runs, so the score describes the shipped model; accuracy is the same either way up to fp16 rounding | Hub repo@revision for adapters; keeping the adapter unmerged; LoRA rows in the baseline table |
| D-21e | 2026-11-24 | **Checkpoint and dev pass every 500 steps** (was 1,000 in D-21b; `eval_steps` = `save_steps` = 500 in `configs/phase3/lora.yaml`). The first run resumes from `checkpoint-1000` with the new interval: the trainer restores intervals from the checkpoint on resume, so the `IntervalsFromConfig` callback in `asr/train.py` re-applies the config's. Owner chose this | The first run hung after its step-1000 checkpoint (GPU wait, likely VRAM shared with the desktop); at ~19.6 s/step a stop lost up to ~5.5 h, now ~2.7 h. Saves stay on dev-pass steps, so the trainer's built-in best-checkpoint tracking keeps working. Costs ~9 more dev passes (~16 min each, ~+2.5 h) | Keeping 1,000; saving every 250 with a dev pass every 1,000 (needs our own best-checkpoint logic) |
| D-16g | 2026-11-24 | **Downloads retry on network errors:** every Hub download of a source or test set goes through `tuncall.data.hub.retrying`: wait 30 s, doubling up to 10 min between tries, give up after 12 h. 4xx answers other than 408/429 (wrong repo, revision, file or token) are raised at once. Finished and partial files are kept by the Hub cache, so a retry fetches only what is missing. Owner chose the 12 h limit | Overnight downloads (N-01) must survive a dropped connection without someone rerunning them | Retrying forever; giving up after 1 h; manual reruns |
| D-17a | 2026-11-23 | **Dev set (closes D-17):** LinTO has no validation split, so **500 LinTO train clips are held out by whole recording** (`<subset>/<file>`, taken in sha256 order of the recording name; `scripts/linto_dev_holdout.py` → `configs/data/dev-linto.txt`, never trained on) + **SLURP-TN validation** + **TEDxTN validation** (~1,800 clips). Decoded during training with **beam 5** as the benchmark (D-26) and scored with its normalization; the checkpoint with the lowest dev WER is kept. Owner chose all | Covers test-general's style (LinTO) and task-oriented speech; dev numbers in the benchmark's own decoding | Only the sources' validation splits (nothing LinTO-like); greedy dev decoding |
| D-83b | 2026-11-23 | **CPU smoke test (implements D-83a):** `tests/test_asr_train.py` trains a LoRA on a tiny randomly initialised Whisper built in the test from `openai/whisper-tiny`'s config, tokenizer and generation settings (pinned sha, no weights downloaded) with four synthetic clips: train → beam-search eval → checkpoint → resume, in plain `pytest`, so CI runs it | Checks the whole training path in seconds. `hf-internal-testing`'s tiny random Whisper was the first pick, but its config doesn't match its tokenizer (vocabulary of 50,257 vs Arabic token ids above 50,300) | Real `openai/whisper-tiny` weights (150 MB per CI run); skipping |
| P-11 | 2026-11-26 | **Phase 3 ablations deferred until the rest of the project is done:** Phase 3 closes with the first LoRA (`configs/phase3/lora.yaml`) and a model card draft; the ablation table is no longer an exit criterion. Deferred, to revisit if time allows: the D-12d with/without noisy-sources run (Deep Confessions, TunSwitch weakly labelled), telephone-band augmentation, LoRA rank, and more epochs (D-27). D-12d still holds: the noisy sources join the mix only if a with/without run shows a test WER gain. Owner chose this | The first LoRA already beats FarukSTT on WER on all three test sets and meets the test-general targets; each ablation takes the laptop GPU for 2 to 7 days | Running the D-12d ablation before Phase 4; dropping the ablations |
| D-21f | 2026-11-26 | **Model card draft (Phase 3 exit, P-11):** `docs/model-card.md`, hand-written Markdown next to the datasheet. Results are quoted from the README's generated "Fine-tuned ASR" table, with the command that produces them; it covers the base model, LoRA setup, training data and cleaning, test sets, results, comparison caveats (D-13a speaker overlap, D-12b FarukSTT contamination, simulated telephone channel), limitations and licensing status. It becomes a Hugging Face card only if released (P-04) | No new code; the generated README table stays the single source for the numbers | Hugging Face card with a results block written by the benchmark script |

**Implications of D-12:**
- test-general is small (~1.3 h), so every WER/CER on it is reported with bootstrap confidence intervals (D-73).
- FarukSTT's training data has unknown sources and may include LinTO test audio. Check for overlap where possible and state the risk next to any test-general comparison. The domain test set (our own recordings) is the uncontaminated headline comparison.
- Model cards must credit LinTO (CC BY 4.0) and the sources it bundles, as their licenses require.

**Implications of D-87:** ablation sweeps need a small script (e.g. generate configs or loop over overrides), and some duplication between configs is accepted. Keep shared defaults in a base file to limit it.

**Implications of B-04c:** a public model trained partly on a not-yet-public dataset can't be fully reproduced until the dataset is released, so the model card must say so. Model cards must credit every training source as its license requires (e.g. CC BY 4.0 attribution for LinTO), and the license of the starting checkpoint must allow redistribution (Whisper large-v3-turbo is MIT, D-21a). Check both before the first public model.

**Effect of D-80v2:** no MinIO and no local MLflow/Postgres containers for tracking. Docker is still used for the serving stack in later phases. Infrastructure skills are shown there (API, Postgres, job queue, monitoring, CI/CD).

**Implications of D-10:**
- Datasets that write French/English in Latin script will need a **Latin → Arabic-script transliteration step** before training. This is its own small component in C1 and needs its own tests.
- French loanwords have **no standard Arabic spelling** (e.g. كونكسيون vs كونيكسيون). That inflates WER for reasons that don't matter. Plan: (a) an Arabic orthographic normalizer (alef/hamza variants, ta marbuta/ha, alef maqsura/ya, remove diacritics and tatweel, numerals); (b) always report **CER next to WER**; (c) possibly a loanword spelling lexicon to standardize frequent words.
- Downstream (NER, search), product and offer names appear in Arabic script, so the entity normalizer must map them to canonical forms (e.g. "فورفي" → *forfait*).
- An Arabizi/Latin display converter stays an optional later feature (D-34).

**Finding for D-10 (2026-11-20):** the LinTO audio dataset writes **everything in Arabic script, including French/English words** (e.g. "اللوجسيل" = *le logiciel*). Other datasets (TunSwitch, FarukSTT's data) still need checking.

**Implications of B-01 (to keep in mind):**
- 8 GB VRAM makes Whisper large-v3 (1.55B params) fine-tuning tight even with LoRA + 8-bit + gradient checkpointing. Whisper large-v3-turbo (~809M) and smaller models fit comfortably. The base model choice (D-20) must be benchmarked against this limit.
- Colab free sessions time out and GPUs aren't guaranteed, so training scripts must checkpoint often and resume cleanly.
- Kaggle (free, about 30 GPU h/week, sessions up to about 12 h) is a third free option to consider.

**Implications of P-04:** a technical report needs rigorous experiments from day one: frozen test sets, seeds, confidence intervals, an ablation log. First-time arXiv submitters usually need an endorsement from an existing author in that category, so plan for that (e.g. a professor).

---

## 10. References

- FarukSTT — Whisper large-v3 fine-tuned on Tunisian Derja: https://huggingface.co/medyas/FarukSTT
- LinTO Tunisian datasets (paper): https://arxiv.org/abs/2504.02604
- LinTO audio dataset: https://huggingface.co/datasets/linagora/linto-dataset-audio-ar-tn
- Tunisian ASR / TunSwitch: https://huggingface.co/SalahZa/Tunisian_Automatic_Speech_Recognition
- tunis-ai collection (Tunisian text datasets): https://huggingface.co/tunis-ai
- TunSwitch on Zenodo (CC BY 4.0): https://zenodo.org/records/8370566
- IWSLT 2022 dialect task (Tunisian–English, LDC2022E02): https://iwslt.org/2022/dialect
- Whisper large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo

## 11. Glossary

- **WER / CER:** word / character error rate (lower is better).
- **DER:** diarization error rate.
- **Code-switching:** switching languages within an utterance (Darija ↔ French ↔ English).
- **Arabizi:** Arabic written in Latin characters and digits (e.g. "3aslema").
- **LoRA / QLoRA:** parameter-efficient fine-tuning (QLoRA = on a quantized base model).
- **VAD:** voice activity detection.
- **RAG:** retrieval-augmented generation.
- **ADR:** architecture decision record.
