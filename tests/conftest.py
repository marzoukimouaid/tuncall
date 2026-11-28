"""Fixtures for the serving tests: a fresh Postgres schema per test (D-71a).

They need a real Postgres: set ``TEST_DATABASE_URL`` (CI runs a service container). Locally:
``docker run -d --rm -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:16-alpine``
and ``TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres``.
"""

import os
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from tuncall.serve import db


@pytest.fixture
def db_url() -> Iterator[str]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    with db.connect(url) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
        db.migrate(conn)
    yield url


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with db.connect(db_url) as c:
        yield c
