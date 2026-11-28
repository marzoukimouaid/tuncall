"""Queue and worker against a real Postgres, with a stub ``process`` (D-71a)."""

import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest
import soundfile as sf

from serve_fixtures import fixed_record
from tuncall.serve import ServeConfig, db
from tuncall.serve.worker import Rejected, decode, run, work_once

CFG = ServeConfig(lease_s=60, max_attempts=2)
Conn = psycopg.Connection[dict[str, Any]]


def add(conn: Conn, name: str = "a.wav", num_speakers: int | None = 2) -> uuid.UUID:
    call_id = uuid.uuid4()
    db.create_call(conn, call_id, name, f"/audio/{name}", num_speakers)
    return call_id


def expire_lease(conn: Conn, call_id: uuid.UUID) -> None:
    conn.execute(
        "UPDATE calls SET claimed_at = now() - interval '1 hour' WHERE id = %s", (call_id,)
    )


def test_claims_oldest_first_and_stores_the_record(conn: Conn) -> None:
    first, second = add(conn, "a.wav"), add(conn, "b.wav", None)
    seen: list[db.Claim] = []

    def handle(claim: db.Claim):
        seen.append(claim)
        return fixed_record(str(claim.id))

    assert work_once(conn, handle, CFG)
    assert work_once(conn, handle, CFG)
    assert not work_once(conn, handle, CFG)  # queue empty
    assert [c.id for c in seen] == [first, second]
    assert seen[0].num_speakers == 2 and seen[1].num_speakers is None
    call = db.get_call(conn, first)
    assert call is not None and call.status == "done" and call.attempts == 1
    assert call.record == fixed_record(str(first))
    assert call.finished_at is not None and call.error is None


def test_error_retries_once_then_fails(conn: Conn) -> None:
    call_id = add(conn)

    def boom(claim: db.Claim):
        raise RuntimeError("CUDA out of memory")

    work_once(conn, boom, CFG)
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "queued" and call.attempts == 1
    assert call.error == "RuntimeError: CUDA out of memory"
    work_once(conn, boom, CFG)
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "failed" and call.attempts == 2
    assert not work_once(conn, boom, CFG)


def test_retry_succeeds_and_clears_the_error(conn: Conn) -> None:
    call_id = add(conn)
    tries = iter([RuntimeError("flaky"), None])

    def handle(claim: db.Claim):
        if err := next(tries):
            raise err
        return fixed_record()

    work_once(conn, handle, CFG)
    work_once(conn, handle, CFG)
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "done" and call.attempts == 2
    assert call.error is None


def test_rejected_fails_without_retry(conn: Conn) -> None:
    call_id = add(conn)

    def reject(claim: db.Claim):
        raise Rejected("audio is longer than the 30-minute limit")

    work_once(conn, reject, CFG)
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "failed" and call.attempts == 1
    assert call.error == "audio is longer than the 30-minute limit"


def test_expired_lease_is_reclaimed_and_the_old_holder_is_fenced(conn: Conn) -> None:
    call_id = add(conn)
    stale = db.claim(conn, CFG.lease_s, CFG.max_attempts)  # a worker that then crashes
    assert stale is not None and stale.attempt == 1
    assert db.claim(conn, CFG.lease_s, CFG.max_attempts) is None  # lease still valid
    expire_lease(conn, call_id)
    fresh = db.claim(conn, CFG.lease_s, CFG.max_attempts)
    assert fresh is not None and fresh.id == call_id and fresh.attempt == 2
    # The crashed worker wakes up: its result and its error are both ignored.
    assert not db.complete(conn, stale, fixed_record())
    assert db.fail(conn, stale, "late", CFG.max_attempts) is None
    assert db.complete(conn, fresh, fixed_record())
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "done" and call.error is None


def test_expired_lease_on_the_last_attempt_fails(conn: Conn) -> None:
    call_id = add(conn)
    for _ in range(2):
        assert db.claim(conn, CFG.lease_s, CFG.max_attempts) is not None
        expire_lease(conn, call_id)
    assert db.claim(conn, CFG.lease_s, CFG.max_attempts) is None
    call = db.get_call(conn, call_id)
    assert call is not None and call.status == "failed" and call.attempts == 2
    assert call.error == "worker lease expired on attempt 2"


def test_deleted_while_processing(conn: Conn) -> None:
    call_id = add(conn)

    def handle(claim: db.Claim):
        assert db.delete_call(conn, claim.id) == "/audio/a.wav"
        return fixed_record()

    assert work_once(conn, handle, CFG)
    assert db.get_call(conn, call_id) is None
    assert db.counts(conn) == {}


def test_run_keeps_the_result_when_the_connection_drops(conn: Conn, db_url: str) -> None:
    call_id = add(conn)
    opened: list[Conn] = []

    def connect() -> Conn:
        opened.append(db.connect(db_url))
        return opened[-1]

    def handle(claim: db.Claim):
        # Postgres restarts while the call is processed: the worker's connection dies.
        conn.execute("SELECT pg_terminate_backend(%s)", (opened[-1].info.backend_pid,))
        return fixed_record(str(claim.id))

    stop = threading.Event()
    worker = threading.Thread(target=run, args=(connect, handle, CFG.model_copy(
        update={"poll_interval_s": 0.05}), stop))  # fmt: skip
    worker.start()
    deadline = time.monotonic() + 10
    while (call := db.get_call(conn, call_id)) and call.status != "done":
        assert time.monotonic() < deadline, f"stuck in {call.status}"
        time.sleep(0.05)
    stop.set()
    worker.join(5)
    assert len(opened) == 2 and call is not None and call.attempts == 1


def test_migrate_is_idempotent(conn: Conn) -> None:
    assert db.migrate(conn) == []


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
def test_decode_resamples_and_downmixes(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((44_100, 2), dtype=np.float32), 44_100)
    audio, channels = decode(path, max_duration_s=60)
    assert channels == 2 and audio.dtype == np.float32
    assert abs(len(audio) - 16_000) <= 1


@needs_ffmpeg
def test_decode_reads_m4a(tmp_path: Path) -> None:
    path = tmp_path / "call.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    audio, channels = decode(path, max_duration_s=60)
    assert channels == 1 and abs(len(audio) / 16_000 - 2) < 0.1


@needs_ffmpeg
def test_decode_rejects_long_and_non_audio(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16_000 * 5, dtype=np.float32), 16_000)
    with pytest.raises(Rejected, match="longer"):
        decode(path, max_duration_s=3)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio")
    with pytest.raises(Rejected, match="not a readable audio file"):
        decode(junk, max_duration_s=3)
