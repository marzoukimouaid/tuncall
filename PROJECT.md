# Tunisian Darija Call Intelligence — Project Source of Truth

> **Codename:** `tuncall` (placeholder — P-01)
> **Owner:** Mouaid Marzouki
> **Doc version:** v0.16 — 2026-11-20
> **Status:** Phase 0 (repo bootstrap). Vision and architecture are drafted; **all technology choices are OPEN** until decided together and logged in §9.

This file is the single reference for the project. Every decision made gets written into the Decision Log (§9) with the date and reason. If code and this document disagree, update one so they match. Don't leave them drifting apart.

---

## 1. Vision

Build an **end-to-end speech and language intelligence system for Tunisian Darija**, the code-switched mix of Tunisian Arabic, French and English that people actually speak. It turns raw customer-support calls into structured, searchable and measurable information:

> audio in → *who said what, when* → *what they wanted, how they felt, which entities were mentioned* → *summaries, search, analytics, answers*

The project has three goals:

1. **Career:** show AI-engineering skill across the full model lifecycle (data → fine-tuning → evaluation → serving → monitoring), not just API calls.
2. **Contribution:** publish a domain dataset and models that are measurably better than the strongest public Darija baselines on that domain.
3. **Commercial:** be something a Tunisian / North African call center, telecom, bank or BPO would pay for, either as a product or as a freelance/consulting engagement.

### 1.1 Why this project, and where the bar is

- Generic Whisper fine-tuning on Darija **already exists**. The best public model, FarukSTT (fine-tuned from Whisper large-v3), reports about 32% WER on its own validation set; base Whisper large-v3 is above 50%. Linagora/LinTO has released CC BY 4.0 Tunisian audio and text datasets, and TunSwitch, IWSLT Tunisian and others are also available.
- So **the novelty can't be "I fine-tuned Whisper."** It has to come from:
  - **Domain adaptation** (customer-support speech) with a public, reproducible benchmark showing we beat existing models *on that domain*.
  - A **full system** around the model: diarization, NLP, retrieval, analytics.
  - **Engineering discipline:** evaluation harness, CI gates, reproducibility, serving performance, observability.

### 1.2 Non-goals (proposed, to confirm)

- Training a speech foundation model from scratch.
- Supporting all Arabic dialects (Tunisian first; others only if they come almost for free).
- Handling real customer recordings that contain PII before a privacy story exists (see D-07).

---

## 2. Success Criteria (proposed, to confirm — decision P-05)

| Area | Metric | Target (draft) |
|---|---|---|
| ASR | WER / CER on our held-out **support-call test set** | Beat FarukSTT and base Whisper on this set by a meaningful margin (exact number TBD after baselines) |
| ASR | WER on a **general** Darija set | No large regression vs. the base model (we don't overfit to the domain) |
| Diarization | DER on test calls | TBD after baseline |
| NLP | Macro-F1 on intent / sentiment; entity-level F1 on NER | TBD after baseline vs. a prompted large LLM |
| System | End-to-end latency per minute of audio; cost per hour of audio | TBD |
| Reproducibility | Someone else can reproduce headline numbers from the repo | Yes |
| Publication | Model card(s) + dataset card + write-up | Scope TBD (P-04) |

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
- Create **domain data**: support-call style audio (method TBD, D-08), transcribed and labelled.
- Strictly separated splits: `train / dev / test-domain / test-general`. **The test sets are frozen once created** and never used for training or tuning.
- Datasheet: sources, licenses, speakers, consent, demographics, known gaps.

**C2 — ASR**
- First, **benchmark** candidate base models on our test sets with no training (D-20).
- Then fine-tune the chosen model on domain data (method D-21), with ablations: data mix, augmentation (noise, telephone-band 8 kHz simulation, speed), LoRA rank, etc.
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
| **0. Foundations** | Repo, environment, compute, tracking, this doc finalized | All P-xx and core D-xx decisions logged | In progress | _link once created_ |
| **1. Benchmarks first** | Frozen test sets and baseline numbers for existing models | Baseline table in README, reproducible with one command | Not started | _link once created_ |
| **2. Domain data** | Domain dataset collected, transcribed, documented | Datasheet written; splits frozen | Not started | _link once created_ |
| **3. ASR fine-tuning** | Domain-adapted ASR beating baselines | Ablation table; model card draft | Not started | _link once created_ |
| **4. Diarization + pipeline** | Audio → diarized transcript, end to end | Pipeline eval (WER + DER) on test calls | Not started | _link once created_ |
| **5. NLP** | Intent / sentiment / NER / summary | 3-way comparison table; chosen approach shipped | Not started | _link once created_ |
| **6. Serving & product** | API, job queue, storage, dashboard, demo | Deployed demo; load/latency numbers | Not started | _link once created_ |
| **7. Knowledge layer** | Search + RAG + analytics | RAG eval (retrieval + faithfulness) | Not started | _link once created_ |
| **8. Hardening & publish** | CI gates, monitoring, write-up, releases | Public release (scope per P-04) | Not started | _link once created_ |
| **9. Stretch** | Real-time voice agent | Latency budget met | Not started | _link once created_ |

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
| Writing code, unit tests, quick debug runs | Laptop | uv `.venv`; pre-commit (tooling TBD, D-84) |
| Training & evaluation | Laptop (RTX 4070, 8 GB) | Scripts + config files (YAML + Pydantic, D-87); checkpoint & resume built in |
| Jobs needing > 8 GB GPU memory | Colab / Kaggle | Same scripts and configs, run from a clone of the repo; checkpoint often |
| Heavy data processing of public datasets + pushing clean versions | Colab | Data-center bandwidth; result pulled to the laptop once |
| Experiment tracking & small artifacts | DagsHub (hosted MLflow) | Async, batched, buffered logging (D-80d) |
| Full checkpoints, data cache, base models | Laptop disk | Retention: best + latest per run (S-01) |
| CI (lint, types, tests, tiny-sample smoke test) | GitHub Actions | Exact checks TBD (D-83, D-84) |
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
│   ├── asr/       # C2 (planned)
│   ├── eval/      # C8 (planned)
│   ├── config.py  # YAML + Pydantic loader, D-87 (planned)
│   └── tracking.py# resilient MLflow wrapper, D-80d (planned)
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
| Not enough domain audio | Fine-tuning gains too small to show | Role-played calls, augmentation, synthetic data (D-08); decide early in Phase 2 |
| Inconsistent transcription conventions across datasets | Noisy WER, misleading results | Fix one convention (D-10) and normalize everything before training or eval |
| Test-set contamination (public datasets overlap with model training data) | Inflated numbers | Domain test set recorded by us; dedupe checks |
| GPU limits | Slow iteration, can't train large models | Pick model sizes to fit compute (B-01); LoRA; smart scheduling |
| Slow, high-latency home connection | Blocked network, slow logging, failed uploads | Network rules N-01; async buffered logging D-80d; heavy transfers in the cloud |
| DagsHub free-tier limits change (storage listed as 200 GB officially, 20 GB elsewhere) | Tracking or artifacts disrupted | Keep artifacts small; MLflow client is standard, so we can move to self-hosted MLflow by changing the URL |
| A token (HF / DagsHub) gets committed to the public repo | Accounts compromised | Tokens only in environment variables / an untracked `.env`; `.env` in `.gitignore`. Secret-detection hook was not selected (D-84d) and can be added later |
| Scope creep | Nothing finished by internship season | Hard v1 cut; stretch goals stay stretch |
| Privacy / consent | Can't publish data; legal exposure | Consent forms; no real customer PII unless anonymized (D-07) |
| Licensing | Can't release a derived dataset or model | Track every source's license in the datasheet |

---

## 8. Open Questions Backlog

Grouped by area. Each one becomes a Decision Log entry once answered. **Nothing below is decided.** "Notes" are options or considerations, not choices.

### P — Project & ways of working
| ID | Question | Notes |
|---|---|---|
| P-01 | Project name / codename? | |
| P-02 | Weekly time budget? | Drives scope |
| P-03 | Target date for v1? | Internship applications typically open Jan–Mar |
| P-04 | Publication ambition? | HF model + dataset cards / technical blog / workshop paper (e.g. Arabic NLP venues) / demo video |
| P-05 | Confirm success metrics and targets (§2)? | |
| P-06 | End-state: portfolio only, open-source + consulting, or SaaS product? | Affects licensing, multi-tenancy, auth |
| P-07 | Will anyone else collaborate (friends for recordings/annotation, a professor)? | |

### D — Repository & workflow
| ID | Question | Notes |
|---|---|---|
| D-01 | Use git at all? Hosted where? | GitHub / GitLab / none |
| D-02 | Public from day one or private until v1? | |
| D-03 | Monorepo or multiple repos? | |
| D-04 | Branching / commit conventions? | trunk-based vs. GitFlow; Conventional Commits or not |
| D-05 | Task tracking? | GitHub Projects / Issues / Notion / none |
| D-06 | Architecture Decision Records (ADRs) alongside this file? | |
| D-07 | Privacy policy for audio (consent, PII redaction, retention)? | |
| D-08 | How do we get domain audio? | Record role-played calls with friends / partner with a company / public content (licensing!) / synthetic TTS / mix |
| D-09 | Annotation tool? | Label Studio / Argilla / custom / spreadsheets |
| D-10 | **Transcription convention for code-switching** | Arabic script for Darija + Latin for French/English? All Arabic script? Arabizi? Must match or map to public datasets |
| D-11 | Size targets for domain train / test sets (hours)? | |
| D-12 | Which public datasets to include? | LinTO (+ augmented), TunSwitch, IWSLT Tunisian, FARUK cleaned, tunis-ai text corpora; check each license |

### B — Compute & environment
| ID | Question | Notes |
|---|---|---|
| B-01 | GPU source and monthly budget? | Kaggle/Colab free / Colab Pro / rented (RunPod, Vast, Lambda…) / university GPU / local GPU |
| B-02 | Local machine specs (OS, RAM, GPU)? | |
| B-03 | Python environment tooling? | uv / poetry / conda / pip+venv |
| B-04 | Where do large files live (audio, checkpoints)? | HF Hub / S3-compatible (R2, B2, MinIO) / Drive |

### ASR
| ID | Question | Notes |
|---|---|---|
| D-20 | Base model candidates to benchmark? | Whisper large-v3 / large-v3-turbo / medium / small; FarukSTT; wav2vec2-XLS-R / w2v-BERT 2.0; MMS |
| D-21 | Fine-tuning method? | Full fine-tune vs. LoRA/QLoRA; continue from FarukSTT vs. from base Whisper |
| D-22 | Inference engine? | faster-whisper (CTranslate2) / whisper.cpp / HF Transformers / ONNX / TensorRT |
| D-23 | Diarization approach? | pyannote / NeMo / WhisperX pipeline / channel split |
| D-24 | Streaming ASR in v1 or batch only? | |
| D-25 | Telephone-quality (8 kHz) robustness as an explicit goal? | |

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
| D-70 | Eval harness: custom or built on a framework? | |
| D-71 | Run evals in CI? Which ones block a merge? | |
| D-72 | LLM-as-judge for summaries/RAG, calibrated against human ratings? | |
| D-73 | Statistical rigor: confidence intervals / bootstrap on WER deltas? | |

### MLOps & infra
| ID | Question | Notes |
|---|---|---|
| D-80 | Experiment tracking? | W&B / MLflow / none |
| D-81 | Data & model versioning? | DVC / HF Hub revisions / lakeFS / none |
| D-82 | Containerization? | Docker + compose / k8s (kind/k3s) / none |
| D-83 | CI/CD platform? | GitHub Actions / GitLab CI / none |
| D-87 | Config system for training? | Plain YAML / Hydra / other |
| D-88 | Data versioning beyond HF Hub revisions? | DVC (remote TBD) / HF revisions only |
| D-84 | Code quality tooling? | ruff, mypy/pyright, pre-commit, pytest coverage targets |
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

| ID | Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|---|
| — | 2026-11-20 | Work on the speech project (Darija) now; improved document AI later | Resume already covers document AI; speech adds a new, distinctive area | Doc-AI flagship first |
| B-01 | 2026-11-20 | Compute = local RTX 4070 Laptop (8 GB VRAM) + Google Colab free (T4, 16 GB). No paid GPU for now | Available resources | Paid rented GPUs, university GPU |
| D-01, D-02 | 2026-11-20 | GitHub, public from day 1 | Visible build-in-public history for recruiters | Private until v1, GitLab, no repo |
| P-03 | 2026-11-20 | No fixed deadline; build it properly. Phases stay as the quality bar, not a timeline | Owner's choice: depth over speed | Internship-season v1 cut |
| P-04 | 2026-11-20 | Publish: HF model(s) + dataset, demo video, and a technical report (PDF / blog, possibly arXiv) linked from a LinkedIn post | Maximum credibility | — |

| P-02 | 2026-11-20 | Time budget: about 14–20 h/week normally, ≤8 h/week during exams. Quality matters more than speed | Owner is a student | — |
| P-06 | 2026-11-20 | End-state: portfolio + open source. No multi-tenancy or billing; a permissive license (exact license TBD) | Focus on credibility and reach | Open core, SaaS |
| D-08 | 2026-11-20 | Domain data strategy (owner delegated this to Claude): **(1) Core:** scripted, role-played support calls recorded under signed consent; the only source for the published domain **test set** and the main domain train set. **(2) Train-only augmentation:** telephone-band simulation, noise, speed; synthetic TTS speech only as an ablation, kept only if it measurably helps. **(3) Opportunistic:** if a company shares real calls, use them as a *private* extra eval set and never publish them. **No scraped web audio** in the published dataset (licensing) | Keeps the dataset fully publishable and the test set clean; real-call eval is a bonus, not a dependency | Web-scraped audio as a main source; synthetic-only data |

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

**Implications of D-87:** ablation sweeps need a small script (e.g. generate configs or loop over overrides), and some duplication between configs is accepted. Keep shared defaults in a base file to limit it.

**Implications of B-04c:** a public model trained partly on a not-yet-public dataset can't be fully reproduced until the dataset is released, so the model card must say so. Model cards must credit every training source as its license requires (e.g. CC BY 4.0 attribution for LinTO), and the license of any starting checkpoint (e.g. FarukSTT) must allow redistribution. Check both before the first public model.

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

## 11. Glossary

- **WER / CER:** word / character error rate (lower is better).
- **DER:** diarization error rate.
- **Code-switching:** switching languages within an utterance (Darija ↔ French ↔ English).
- **Arabizi:** Arabic written in Latin characters and digits (e.g. "3aslema").
- **LoRA / QLoRA:** parameter-efficient fine-tuning (QLoRA = on a quantized base model).
- **VAD:** voice activity detection.
- **RAG:** retrieval-augmented generation.
- **ADR:** architecture decision record.
