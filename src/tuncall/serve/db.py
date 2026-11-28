"""The ``calls`` table: storage and job queue in one (D-51a, D-53a, D-53b, D-55c, D-55d).

psycopg 3 and plain SQL. The CallRecord is stored as JSONB and validated by the Pydantic
model on the way in and out. Migrations are the numbered ``migrations/*.sql`` files, applied
in order at startup under an advisory lock, so the API and the worker can both start first.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from typing import Any, Literal, LiteralString, cast

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from tuncall.record import CallRecord

Status = Literal["queued", "processing", "done", "failed"]
_MIGRATION_LOCK = 7_246_001  # arbitrary advisory-lock key for migrations


class Call(BaseModel):
    """What the API shows of a call. ``record`` is set once the call is done."""

    id: uuid.UUID
    filename: str
    status: Status
    attempts: int
    num_speakers: int | None
    error: str | None
    created_at: datetime
    claimed_at: datetime | None
    finished_at: datetime | None
    record: CallRecord | None = None


@dataclass(frozen=True)
class Claim:
    """A call a worker holds. ``attempt`` is the fencing token: only the holder of the latest
    attempt may finish the call, so a worker whose lease was taken over cannot overwrite it."""

    id: uuid.UUID
    filename: str
    audio_path: str
    num_speakers: int | None
    attempt: int


def connect(url: str) -> psycopg.Connection[dict[str, Any]]:
    return psycopg.Connection[dict[str, Any]].connect(url, autocommit=True, row_factory=dict_row)


def migrate(conn: psycopg.Connection[Any]) -> list[str]:
    """Apply the migrations not yet applied; returns their names."""
    applied: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
        scripts = sorted(
            (
                f
                for f in (files("tuncall.serve") / "migrations").iterdir()
                if f.name.endswith(".sql")
            ),
            key=lambda f: f.name,
        )
        for script in scripts:
            if script.name not in done:
                conn.execute(
                    cast(LiteralString, script.read_text(encoding="utf-8"))
                )  # our own file
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (script.name,))
                applied.append(script.name)
    return applied


_COLUMNS = (
    "id, filename, status, attempts, num_speakers, error, created_at, claimed_at, finished_at"
)


def create_call(
    conn: psycopg.Connection[Any],
    call_id: uuid.UUID,
    filename: str,
    audio_path: str,
    num_speakers: int | None,
) -> None:
    conn.execute(
        "INSERT INTO calls (id, filename, audio_path, num_speakers) VALUES (%s, %s, %s, %s)",
        (call_id, filename, audio_path, num_speakers),
    )


def list_calls(conn: psycopg.Connection[Any], limit: int = 200) -> list[Call]:
    rows = conn.execute(f"SELECT {_COLUMNS} FROM calls ORDER BY created_at DESC LIMIT %s", (limit,))
    return [Call.model_validate(r) for r in rows]


def get_call(conn: psycopg.Connection[Any], call_id: uuid.UUID) -> Call | None:
    row = conn.execute(f"SELECT {_COLUMNS}, record FROM calls WHERE id = %s", (call_id,)).fetchone()
    return Call.model_validate(row) if row else None


def audio_path(conn: psycopg.Connection[Any], call_id: uuid.UUID) -> str | None:
    row = conn.execute("SELECT audio_path FROM calls WHERE id = %s", (call_id,)).fetchone()
    return row["audio_path"] if row else None


def delete_call(conn: psycopg.Connection[Any], call_id: uuid.UUID) -> str | None:
    """Delete the row; returns its audio path (for the caller to remove) or None if absent."""
    row = conn.execute(
        "DELETE FROM calls WHERE id = %s RETURNING audio_path", (call_id,)
    ).fetchone()
    return row["audio_path"] if row else None


def counts(conn: psycopg.Connection[Any]) -> dict[str, int]:
    rows = conn.execute("SELECT status, count(*) AS n FROM calls GROUP BY status")
    return {r["status"]: r["n"] for r in rows}


def claim(conn: psycopg.Connection[Any], lease_s: float, max_attempts: int) -> Claim | None:
    """Take the oldest queued call, or a ``processing`` one whose lease ran out (D-55d).

    Claiming counts as an attempt. An expired call that already used ``max_attempts`` is
    failed instead of reclaimed. ``SKIP LOCKED`` lets several workers claim at once.
    """
    with conn.transaction():
        conn.execute(
            "UPDATE calls SET status = 'failed', finished_at = now(),"
            " error = %s"
            " WHERE status = 'processing' AND attempts >= %s"
            " AND claimed_at < now() - make_interval(secs => %s)",
            (f"worker lease expired on attempt {max_attempts}", max_attempts, lease_s),
        )
        row = conn.execute(
            "UPDATE calls SET status = 'processing', attempts = attempts + 1, claimed_at = now()"
            " WHERE id = (SELECT id FROM calls"
            "  WHERE status = 'queued'"
            "   OR (status = 'processing' AND claimed_at < now() - make_interval(secs => %s))"
            "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)"
            " RETURNING id, filename, audio_path, num_speakers, attempts",
            (lease_s,),
        ).fetchone()
    if row is None:
        return None
    return Claim(
        row["id"], row["filename"], row["audio_path"], row["num_speakers"], row["attempts"]
    )


def complete(conn: psycopg.Connection[Any], claim: Claim, record: CallRecord) -> bool:
    """Store the record. False if the claim is stale (call deleted or taken over)."""
    cur = conn.execute(
        "UPDATE calls SET status = 'done', record = %s, error = NULL, finished_at = now()"
        " WHERE id = %s AND status = 'processing' AND attempts = %s",
        (Jsonb(record.model_dump(mode="json")), claim.id, claim.attempt),
    )
    return cur.rowcount == 1


def fail(
    conn: psycopg.Connection[Any], claim: Claim, error: str, max_attempts: int, retry: bool = True
) -> Status | None:
    """End an attempt with an error: back to ``queued`` while attempts remain (and ``retry``),
    else ``failed``. Returns the new status, or None if the claim is stale."""
    row = conn.execute(
        "UPDATE calls SET error = %s,"
        " status = CASE WHEN %s AND attempts < %s THEN 'queued' ELSE 'failed' END,"
        " finished_at = CASE WHEN %s AND attempts < %s THEN NULL ELSE now() END"
        " WHERE id = %s AND status = 'processing' AND attempts = %s RETURNING status",
        (error, retry, max_attempts, retry, max_attempts, claim.id, claim.attempt),
    ).fetchone()
    return row["status"] if row else None
