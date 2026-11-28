# Phase 6 (Serving & product): implementation plan

Handoff spec from the Phase 6 grilling session (2026-11-28). Every choice below was decided
by the owner in that session. **Anything not covered here is still open: ask the owner
before implementing it** (working agreement 8b.1).

## Before starting

- Phase 5 (PRs #62–#65) was being merged in another session during planning. **Start only
  once Phase 5 is on `main`**, then branch Phase 6 from `main` (as P-12).
- This file was written uncommitted, without touching `PROJECT.md` (the merge session was
  editing it). Step 1 below logs everything into `PROJECT.md` §9, ADR 0010 and the §11
  glossary.
- Owner rules still apply: ask before downloads and long runs (path, size, time); load
  `.env` before runs (`HF_HOME`); no AI attribution in commits/PRs; Ruff does formatting.

## Guiding constraint (owner, Q2)

Phase 6 deploys **locally** with Docker Compose and a demo video. But the system must be
**ready to ship by the end of the phases** (at the latest Phase 8). So no Phase 6 choice
may block a later public deployment: config via files and env vars, images pulled from a
registry, no weights baked into images, and auth to be added later.

## Decisions (proposed §9 IDs; adjust numbering if taken)

| ID | Decision | Alternatives rejected |
|---|---|---|
| P-16 | **Phase 6 exit criterion** (replaces "Deployed demo; load/latency numbers"): (1) `docker compose up` on a machine with an 8 GB NVIDIA GPU runs the whole stack (upload → queue → CallRecord in the dashboard); (2) API and dashboard images built by CI and pushed to GHCR; (3) README "Serving" block (latency per stage, load, cost per hour) from a script; (4) §2 System targets set from that run (P-05 rule); (5) demo video. Public live demo and HF weights → Phase 8 | Public demo in Phase 6 |
| D-61a / D-86a | **Deployment = local Docker Compose** on the laptop, which is the reference machine for all numbers. Public live demo moved to Phase 8, after the license review (P-04a). The project must be ship-ready by the end of the phases | Free CPU HF Space with reduced pipeline; paid GPU (ZeroGPU / rented, breaks B-01) |
| D-54a | **No auth**, single user, one local instance (P-06). Compose publishes the API port on **`127.0.0.1` only**; exposing it is a Phase 8 change together with auth | Login in Phase 6 |
| D-50a | **API = FastAPI** (reuses the `CallRecord` Pydantic model) | NestJS (schema in two languages) |
| D-52a | **One GPU worker process** loads all models once (`pipeline.load_models`), takes one call at a time and runs `pipeline.process`. The API process never imports torch | Triton / vLLM / BentoML model servers |
| D-53a | **PostgreSQL** | SQLite |
| D-51a | **Postgres is the job queue**: `calls.status`, the worker claims with `SELECT … FOR UPDATE SKIP LOCKED` | Redis + RQ / Arq / Celery |
| D-22b | **Serving keeps HF Transformers** (the engine the README numbers come from). A faster engine (faster-whisper, CTranslate2, merged LoRA) is a later experiment and must pass the benchmark before it ships | Switching engines in Phase 6 |
| D-55c | **Call status model** (see glossary): a Call has `status` ∈ queued / processing / done / failed, an attempt count and at most one CallRecord. No "Job" concept; reprocessing a call with a new model is not in Phase 6 | Separate Job entity |
| D-55d | **Crash recovery = lease**: a claim sets `claimed_at` and counts as an attempt; any worker may reclaim a `processing` call whose lease is older than a timeout (≈ 3× the expected time for a 30-minute call). After **2 attempts** → `failed` with the error message | Reset stuck rows at worker startup (only safe with one worker) |
| D-50b | **API shape, always async** (no sync path for short clips): `POST /calls` (multipart file + optional `num_speakers`, default 2; pyannote's count estimate is unreliable on telephone, P-13) → `202 {call_id}`; `GET /calls` (list + status); `GET /calls/{id}` (status + CallRecord when done); `GET /calls/{id}/audio`; `DELETE /calls/{id}`; `GET /health`. The dashboard polls every few seconds | Sync endpoint; WebSockets |
| D-50c | **Audio input:** ffmpeg in the worker image, so M4A/AAC phone formats work as well as WAV/FLAC/OGG/MP3; **max 30 minutes** per call | soundfile formats only |
| D-50d | **Audio storage:** uploads go to a Docker volume, the path in the DB; kept until the call is deleted, no automatic expiry | Expiry / object storage |
| D-80f | **Fine-tuned weights** (~1.6 GB: ASR LoRA 107 MB, Qwen intent + entities adapters 254 MB, MARBERTv2 sentiment + role 1.25 GB) are **mounted from the local `checkpoints/`** (paths already in `configs/phase5/pipeline.yaml`). Published to the HF Hub in Phase 8 after the license review (overnight upload, N-01). Images never contain weights; base models download by pinned revision (pyannote needs `HF_TOKEN`) | Private HF upload now (~5 h) |
| D-82a | **Containers:** Compose with `db` (Postgres), `api` (FastAPI, also serves the built dashboard as static files), `worker` (CUDA image, GPU via NVIDIA Container Toolkit). Built locally for development; GitHub Actions builds and pushes to **GHCR** on `main`, so nothing large is uploaded from home. Risk: the CUDA worker image is several GB vs ~14 GB free runner disk; if it does not fit, the worker build becomes a manual workflow | Kubernetes; pushing images from home |
| D-03b | **One package, new `serve` extra** (fastapi, uvicorn, psycopg). API code imports only `record.py` + the DB module, never torch. API image: `--extra serve`; worker image: `--extra serve --extra gpu` | uv workspace split |
| D-53b | **DB code:** psycopg 3 + plain SQL; CallRecord stored as **JSONB**, validated by the Pydantic model in and out; numbered `.sql` migrations applied at startup | SQLAlchemy + Alembic |
| D-60a | **Dashboard:** React + Vite + TypeScript + Tailwind in `web/`. Three screens: (1) upload, (2) call list with status, (3) call view: transcript by speaker with timestamps, each speaker's role guess, each segment's intent / entities / sentiment, audio playback that jumps to a segment. Search and analytics are Phase 7. UI text in English; transcript shown **right-to-left** (Arabic script, D-10) | Gradio / Streamlit |
| D-85a | **Observability in Phase 6:** the per-stage timings already in `CallRecord.meta.processing_ms_by_stage`, structured logs, and `GET /health`. Prometheus + Grafana → Phase 8 | Full metrics stack now |
| D-87b | **Serving config:** `configs/phase6/serve.yaml` (YAML + Pydantic, D-87) points at the pipeline config and sets limits, storage path and polling interval. Secrets and the database URL come from environment variables (`.env`; add names to `.env.example`) | — |
| D-70e | **Serving metrics** (§2 System row): latency = processing seconds per minute of audio, split by stage (diarization, ASR, NLP); load = N simultaneous uploads with N = 1, 4, 16, giving throughput (calls/hour), queue wait and p95 time to result (one worker, so throughput is capped, and that is reported honestly); cost per hour of audio = GPU-hours × hourly rental price of a comparable GPU (price, date and source recorded in the config, looked up once). Measured **end to end through the API**. Knobs are tuned on the **dev** simulated conversations; README numbers come from **one run on the frozen test** conversations, clean and telephone. Targets set from that run (P-05 rule, as P-13/P-15) | In-process timing only; electricity as cost |
| D-83c | **CI additions:** a Postgres service for the API tests; a `web` job (type-check + build); PRs build the Docker images without pushing (the worker build is skipped on PRs if too slow); pushes to GHCR only on `main`. Everything stays under the required `checks` job, so branch protection doesn't change | — |
| D-71a | **Serving tests:** API tests against a real Postgres; worker tests with a stub `process` returning a fixed CallRecord (covers claim, lease reclaim, retry, failure, delete). One manual end-to-end run on the laptop with the real models before the PR | — |
| P-17 | **Demo video audio:** a simulated test conversation (TEDxTN clips stitched together). The owner records the video | Owner role-playing a call; TTS |

## Glossary additions (`PROJECT.md` §11)

- **Call status:** where a Call is in processing: *queued* (uploaded, waiting), *processing*
  (claimed by a worker), *done* (has its CallRecord), *failed* (2 attempts used; error kept).
- **Attempt:** one worker's try at processing a Call. A crash or lease timeout ends it; a
  Call gets at most 2.
- **Lease:** the time a worker may hold a claimed Call before another worker may reclaim it.
- **Worker:** the GPU process that holds the models and processes Calls one at a time.
- **Real-time factor:** processing time divided by audio duration (here: seconds per minute
  of audio).

## Work breakdown (milestone M6, one issue and one PR each, stacked if needed)

1. **Docs:** the §9 rows above, ADR 0010 (template `docs/adr/0000-template.md`), §11
   glossary terms, the §5 Phase 6 row and exit criterion (P-16), §8 rows marked ✅.
2. **DB, queue and worker:** `serve` extra, migrations, `calls` table (status, attempts,
   `claimed_at`, audio path, `num_speakers`, error, CallRecord JSONB), claim / lease /
   retry / fail, worker loop around `pipeline.process`, stub-based tests with Postgres.
3. **API:** FastAPI endpoints (D-50b), upload limits (D-50c), audio volume (D-50d),
   `/health`, Postgres in CI.
4. **Dashboard:** the three screens (D-60a), served by the API as static files, `web` CI
   job.
5. **Compose and images:** Dockerfiles (api without torch, worker CUDA + ffmpeg),
   `compose.yaml` (127.0.0.1 binding, volumes for audio, `checkpoints/`, `HF_HOME`),
   GHCR workflow, image build check on PRs.
6. **Serving benchmark:** a script that drives the running stack through the API
   (D-70e), the dev tuning run, one test run, README "Serving" block, §2 System targets,
   §5 marks Phase 6 done. Then the owner records the demo video (P-17).

## Facts gathered during planning

- `pipeline.process(models, audio, ..., num_speakers=)` is the in-memory entry point;
  `run_file` reads audio with `soundfile` (so no M4A/AAC without ffmpeg, D-50c).
- `CallRecord.meta.processing_ms_by_stage` already exists (wall-clock ms per stage).
- All shipped models fit together in 8 GB; the NLP stage took 4.8 s on a 2-minute call
  (Phase 5 handoff).
