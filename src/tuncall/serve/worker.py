"""The GPU worker (D-52a): loads every model once, then processes one call at a time.

uv run python -m tuncall.serve.worker --config configs/phase6/serve.yaml

Each claimed call is decoded with ffmpeg (any format it reads, D-50c), run through
``pipeline.process`` and stored. An error ends the attempt: the call goes back to the queue
until it used ``max_attempts``, then it is ``failed`` with the message (D-55d). A call longer
than ``max_duration_s`` is failed at once (retrying cannot help).
"""

import argparse
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from tuncall.config import load_config
from tuncall.record import CallRecord
from tuncall.serve import ServeConfig, db, setup_logging

SAMPLE_RATE = 16_000
log = logging.getLogger("tuncall.worker")

Handler = Callable[[db.Claim], CallRecord]


class Rejected(Exception):
    """The call can never succeed (too long, not audio): fail it without a retry."""


def decode(path: str | Path, max_duration_s: float) -> tuple[np.ndarray, int]:
    """16 kHz mono float32 and the source channel count. Rejects audio over the limit."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, errors="replace", check=False,
    )  # fmt: skip
    first = probe.stdout.split()[0] if probe.stdout.strip() else ""
    if probe.returncode != 0 or not first.isdigit():
        raise Rejected(f"not a readable audio file: {probe.stderr.strip()[-300:]}")
    channels = int(first)
    # Decode at most one second past the limit, so an oversized file never fills memory.
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", str(max_duration_s + 1),
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True, check=False,
    )  # fmt: skip
    if out.returncode != 0:
        raise Rejected(
            f"ffmpeg could not decode the file: {out.stderr.decode(errors='replace')[-300:]}"
        )
    audio = np.frombuffer(out.stdout, dtype=np.float32)
    if len(audio) > max_duration_s * SAMPLE_RATE:
        raise Rejected(f"audio is longer than the {max_duration_s / 60:g}-minute limit")
    if len(audio) == 0:
        raise Rejected("audio is empty")
    return audio, channels


class Unsaved(Exception):
    """The connection dropped while storing an outcome; ``save`` stores it on a new one."""

    def __init__(self, save: Callable[[psycopg.Connection[Any]], None]) -> None:
        super().__init__("outcome not stored")
        self.save = save


def work_once(conn: psycopg.Connection[Any], handle: Handler, cfg: ServeConfig) -> bool:
    """Claim and process one call. False when the queue is empty.

    Raises ``Unsaved`` if the database connection drops while storing the outcome, so a
    finished result is not thrown away (the attempt number makes storing it later safe).
    """
    claim = db.claim(conn, cfg.lease_s, cfg.max_attempts)
    if claim is None:
        return False
    ctx = {"call_id": str(claim.id), "attempt": claim.attempt}
    log.info("call claimed", extra=ctx)
    t0 = time.perf_counter()
    try:
        record = handle(claim)
    except Rejected as e:
        reason = str(e)  # `e` is unbound once the except block ends

        def save(c: psycopg.Connection[Any]) -> None:
            status = db.fail(c, claim, reason, cfg.max_attempts, retry=False)
            log.warning("call rejected", extra=ctx | {"error": reason, "status": status})

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log.exception("call attempt failed", extra=ctx)

        def save(c: psycopg.Connection[Any]) -> None:
            status = db.fail(c, claim, error, cfg.max_attempts)
            log.info("call attempt ended", extra=ctx | {"error": error, "status": status})

    else:

        def save(c: psycopg.Connection[Any]) -> None:
            stored = db.complete(c, claim, record)
            log.info(
                "call done" if stored else "call result dropped (deleted or taken over)",
                extra=ctx
                | {
                    "seconds": round(time.perf_counter() - t0, 2),
                    "duration_s": record.duration_s,
                    "ms_by_stage": record.meta.processing_ms_by_stage,
                },
            )

    try:
        save(conn)
    except psycopg.OperationalError as e:
        raise Unsaved(save) from e
    return True


Connect = Callable[[], psycopg.Connection[Any]]


def _reconnect(connect: Connect, stop: threading.Event) -> psycopg.Connection[Any] | None:
    """Retry with backoff (1 s doubling to 30 s) until connected or stopped."""
    delay = 1.0
    while not stop.is_set():
        try:
            return connect()
        except psycopg.OperationalError:
            log.warning("database unreachable, retrying", extra={"in_s": delay})
            stop.wait(delay)
            delay = min(delay * 2, 30)
    return None


def run(
    connect: Connect, handle: Handler, cfg: ServeConfig, stop: threading.Event | None = None
) -> None:
    """Process calls until ``stop`` is set; sleeps ``poll_interval_s`` when the queue is empty.

    A dropped database connection is reopened (the models stay loaded), and an outcome that
    could not be stored is stored on the new connection.
    """
    stop = stop or threading.Event()
    conn = _reconnect(connect, stop)
    pending: Callable[[psycopg.Connection[Any]], None] | None = None
    while conn is not None and not stop.is_set():
        try:
            if pending is not None:
                pending(conn)
                pending = None
            if not work_once(conn, handle, cfg):
                stop.wait(cfg.poll_interval_s)
        except (Unsaved, psycopg.OperationalError) as e:
            if isinstance(e, Unsaved):
                pending = e.save
            log.warning("database connection lost, reconnecting", extra={"error": str(e)})
            conn.close()
            conn = _reconnect(connect, stop)


def pipeline_handler(cfg: ServeConfig) -> Handler:
    """Load every pipeline model once (this is where torch is imported)."""
    from tuncall.pipeline import PipelineConfig, load_models, process

    models = load_models(load_config(cfg.pipeline_config, PipelineConfig))

    def handle(claim: db.Claim) -> CallRecord:
        audio, channels = decode(claim.audio_path, cfg.max_duration_s)
        record, _ = process(
            models,
            audio,
            call_id=str(claim.id),
            source=claim.filename,
            num_speakers=claim.num_speakers,
            channels=channels,
        )
        return record

    return handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6/serve.yaml"))
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config, ServeConfig)
    url = os.environ["DATABASE_URL"]
    with db.connect(url) as conn:
        db.migrate(conn)
    log.info("loading models", extra={"config": cfg.model_dump(mode="json")})
    handle = pipeline_handler(cfg)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())  # finish the current call, then exit
    log.info("worker ready")
    run(lambda: db.connect(url), handle, cfg, stop)


if __name__ == "__main__":
    main()
