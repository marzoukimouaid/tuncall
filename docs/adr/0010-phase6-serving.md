# 0010. Phase 6: local Compose deployment, Postgres as the queue, one GPU worker

- **Status:** accepted
- **Date:** 2026-11-28
- **Decision log ID:** P-16, P-17, D-61a/D-86a, D-54a, D-50a–d, D-51a, D-52a, D-53a–b,
  D-55c–d, D-22b, D-80f, D-82a, D-03b, D-60a, D-85a, D-87b, D-70e, D-83c, D-71a

## Context

Phase 6 turns the batch pipeline (Phases 4–5) into a product: upload a call, get its
CallRecord in a dashboard. The constraints are one laptop with an 8 GB GPU (B-01), no
budget for a GPU host, an upload speed of about 0.68 Mbps, training-source licenses not yet
reviewed (P-04a), and one user. The owner's rule: deploy locally now, but nothing chosen
here may block a public deployment by the end of the phases (at the latest Phase 8).

## Options

1. **Public demo now** (free CPU HF Space with a reduced pipeline, or a paid GPU): a link
   to share, but CPU is too slow for the real pipeline, a paid GPU breaks B-01, and the
   weights cannot be published before the license review.
2. **Local Docker Compose, ship-ready:** the full pipeline on the laptop, images built by
   CI and pushed to GHCR, config from files and env vars, weights mounted (never baked in),
   auth added later. Public demo in Phase 8.

For the moving parts: a separate queue (Redis + RQ/Celery) vs **Postgres as the queue**
(`FOR UPDATE SKIP LOCKED`); model servers (Triton, vLLM, BentoML) vs **one worker process**
holding every model; crash recovery by resetting rows at startup vs **leases** with an
attempt count.

## Decision

Option 2. Three containers: `db` (Postgres, also the job queue), `api` (FastAPI, no torch,
also serves the React dashboard) and `worker` (CUDA, loads every model once, one call at a
time, HF Transformers as benchmarked). A call is queued, processing, done or failed; a
claim is an attempt with a lease, an expired lease can be reclaimed by any worker, and
after 2 attempts the call fails with its error. The API is always async and published on
127.0.0.1 only (no auth). Serving numbers are measured end to end through the API on the
frozen test conversations, once, after tuning on dev.

## Consequences

- One service fewer than a Redis queue; the queue state is the call state, and the
  attempt number fences a worker whose lease was taken over.
- Throughput is capped at one call at a time; the load test reports that as it is.
- Public deployment (Phase 8) needs auth, a non-local bind, the weights on the HF Hub and
  a GPU host; nothing in the images or config has to change shape for it.
- The CUDA worker image is several GB: CI builds it only on `main`; if it outgrows the
  runner disk, it becomes a manual workflow.
- A faster ASR engine (faster-whisper, CTranslate2) is a later experiment that must pass
  the benchmark first.
