# syntax=docker/dockerfile:1
# Phase 6 images (D-82a, D-03b). No weights inside: fine-tuned checkpoints are mounted,
# base models download by pinned revision into the mounted HF cache (D-80f).
#   docker build --target api -t tuncall-api .
#   docker build --target worker -t tuncall-worker .

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
# UV_NO_CACHE: no second copy of the wheels (CUDA torch is GB) left in the layers.
ENV UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./

# API: FastAPI and Postgres only, no torch (D-52a); also serves the dashboard.
FROM base AS api
RUN uv sync --locked --no-dev --extra serve --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --extra serve
COPY --from=web /web/dist ./web/dist
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT
EXPOSE 8000
CMD ["uvicorn", "--factory", "tuncall.serve.app:app_from_env", "--host", "0.0.0.0", "--port", "8000"]

# Worker: CUDA torch (its wheels bring the CUDA runtime; the NVIDIA Container Toolkit
# provides the driver) and ffmpeg for M4A/AAC and friends (D-50c).
FROM base AS worker
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --locked --no-dev --extra serve --extra gpu --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --extra serve --extra gpu
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT
CMD ["python", "-m", "tuncall.serve.worker"]
