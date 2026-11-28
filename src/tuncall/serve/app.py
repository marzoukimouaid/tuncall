"""The HTTP API (D-50a, D-50b). Always async: upload returns a call id, the client polls.

uvicorn --factory tuncall.serve.app:app_from_env  (reads SERVE_CONFIG and DATABASE_URL)

``POST /calls`` stores the upload on the audio volume (D-50d) and queues it; the worker does
the rest. Also serves the built dashboard at ``/`` when ``web_dir`` exists (D-60a). Never
imports torch (D-52a).
"""

import os
import re
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tuncall.config import load_config
from tuncall.serve import ServeConfig, db, setup_logging

CHUNK = 1 << 20


class Created(BaseModel):
    call_id: uuid.UUID


def get_conn(request: Request) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """One connection per request: a single local user needs no pool (D-54a)."""
    with db.connect(request.app.state.database_url) as conn:
        yield conn


Conn = Annotated[psycopg.Connection[dict[str, Any]], Depends(get_conn)]


def create_app(cfg: ServeConfig, database_url: str) -> FastAPI:
    storage = cfg.storage_dir.resolve()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        storage.mkdir(parents=True, exist_ok=True)
        with db.connect(database_url) as conn:
            db.migrate(conn)
        yield

    app = FastAPI(title="TunCall API", lifespan=lifespan)
    app.state.database_url = database_url
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=cfg.allowed_hosts)

    @app.middleware("http")
    async def upload_size(request: Request, call_next):
        # Starlette spools the whole body to disk before the handler runs: refuse early.
        if request.method == "POST" and request.url.path == "/calls":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Content-Length required"}, status_code=411)
            if not length.isdigit() or int(length) > (cfg.max_upload_mb + 1) << 20:
                return JSONResponse(
                    {"detail": f"file is over {cfg.max_upload_mb} MB"}, status_code=413
                )
        return await call_next(request)

    @app.post("/calls", status_code=202)
    def upload(
        conn: Conn,
        file: Annotated[UploadFile, File()],
        num_speakers: Annotated[int | None, Form(ge=1, le=10)] = None,
    ) -> Created:
        call_id = uuid.uuid4()
        name = file.filename or "upload"
        # Keep a safe extension only: it is a format hint for ffmpeg, never a path.
        suffix = re.sub(r"[^A-Za-z0-9]", "", Path(name).suffix)[:8]
        path = storage / f"{call_id}{'.' + suffix if suffix else ''}"
        limit, size = cfg.max_upload_mb << 20, 0
        try:
            with path.open("wb") as out:
                while chunk := file.file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file is over {cfg.max_upload_mb} MB")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty file")
            db.create_call(conn, call_id, name, str(path), num_speakers or cfg.default_num_speakers)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return Created(call_id=call_id)

    @app.get("/calls")
    def calls(conn: Conn) -> list[db.Call]:
        return db.list_calls(conn)

    @app.get("/calls/{call_id}")
    def call(conn: Conn, call_id: uuid.UUID) -> db.Call:
        found = db.get_call(conn, call_id)
        if found is None:
            raise HTTPException(404, "no such call")
        return found

    @app.get("/calls/{call_id}/audio")
    def audio(conn: Conn, call_id: uuid.UUID) -> FileResponse:
        path = db.audio_path(conn, call_id)
        if path is None or not Path(path).is_file():
            raise HTTPException(404, "no such call")
        return FileResponse(path)

    @app.delete("/calls/{call_id}", status_code=204)
    def delete(conn: Conn, call_id: uuid.UUID) -> Response:
        path = db.delete_call(conn, call_id)
        if path is None:
            raise HTTPException(404, "no such call")
        Path(path).unlink(missing_ok=True)
        return Response(status_code=204)

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            with db.connect(database_url) as conn:
                return JSONResponse({"status": "ok", "calls": db.counts(conn)})
        except psycopg.Error as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=503)

    if cfg.web_dir.is_dir():  # mounted last so the API routes win
        app.mount("/", StaticFiles(directory=cfg.web_dir, html=True), name="web")
    return app


def app_from_env() -> FastAPI:
    setup_logging()
    cfg = load_config(os.environ.get("SERVE_CONFIG", "configs/phase6/serve.yaml"), ServeConfig)
    return create_app(cfg, os.environ["DATABASE_URL"])
