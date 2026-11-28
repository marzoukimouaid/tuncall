"""The API against a real Postgres (D-71a). The worker is played by direct ``db`` calls."""

import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from serve_fixtures import fixed_record
from tuncall.serve import ServeConfig, db
from tuncall.serve.app import create_app


@pytest.fixture
def client(db_url: str, tmp_path: Path) -> Iterator[TestClient]:
    cfg = ServeConfig(
        storage_dir=tmp_path / "uploads",
        max_upload_mb=1,
        web_dir=tmp_path / "web",
        allowed_hosts=["testserver"],
    )
    with TestClient(create_app(cfg, db_url)) as c:
        yield c


def upload(client: TestClient, data: bytes = b"RIFF....", name: str = "call.m4a", **form):
    return client.post("/calls", files={"file": (name, data)}, data=form)


def test_upload_queues_the_call(client: TestClient, tmp_path: Path) -> None:
    r = upload(client)
    assert r.status_code == 202
    call_id = r.json()["call_id"]
    call = client.get(f"/calls/{call_id}").json()
    assert call["status"] == "queued" and call["filename"] == "call.m4a"
    assert call["num_speakers"] == 2 and call["record"] is None  # default (D-50b)
    assert "audio_path" not in call
    assert [c["id"] for c in client.get("/calls").json()] == [call_id]
    stored = list((tmp_path / "uploads").iterdir())
    assert [p.name for p in stored] == [f"{call_id}.m4a"]
    assert client.get(f"/calls/{call_id}/audio").content == b"RIFF...."


def test_num_speakers_is_passed_and_validated(client: TestClient) -> None:
    call_id = upload(client, num_speakers="3").json()["call_id"]
    assert client.get(f"/calls/{call_id}").json()["num_speakers"] == 3
    assert upload(client, num_speakers="0").status_code == 422


def test_done_call_returns_its_record(client: TestClient, db_url: str) -> None:
    call_id = upload(client).json()["call_id"]
    with db.connect(db_url) as conn:
        claim = db.claim(conn, 60, 2)
        assert claim is not None
        db.complete(conn, claim, fixed_record(call_id))
    call = client.get(f"/calls/{call_id}").json()
    assert call["status"] == "done"
    assert call["record"] == fixed_record(call_id).model_dump(mode="json")
    assert client.get("/health").json() == {"status": "ok", "calls": {"done": 1}}


def test_upload_limits(client: TestClient, tmp_path: Path) -> None:
    assert upload(client, b"x" * ((1 << 20) + 1)).status_code == 413
    assert upload(client, b"").status_code == 400
    assert list((tmp_path / "uploads").iterdir()) == []  # nothing left behind
    assert client.get("/calls").json() == []


def test_oversized_body_is_refused_before_it_is_read(client: TestClient) -> None:
    big = b"x" * ((2 << 20) + 10)  # over max_upload_mb + 1 MB of multipart slack
    assert upload(client, big).status_code == 413
    r = client.post("/calls", content=b"x", headers={"content-length": "nope"})
    assert r.status_code == 413


def test_unknown_host_is_refused(db_url: str, tmp_path: Path) -> None:
    cfg = ServeConfig(storage_dir=tmp_path / "u", web_dir=tmp_path / "web")  # default hosts
    with TestClient(create_app(cfg, db_url)) as c:
        assert c.get("/calls").status_code == 400  # Host: testserver, as a rebinding page
    with TestClient(create_app(cfg, db_url), base_url="http://127.0.0.1:8080") as c:
        assert c.get("/calls").status_code == 200


def test_unsafe_filename_never_reaches_the_path(client: TestClient, tmp_path: Path) -> None:
    call_id = upload(client, name="../../etc/x.w/a v").json()["call_id"]
    assert [p.name for p in (tmp_path / "uploads").iterdir()] == [call_id]


def test_delete_removes_row_and_audio(client: TestClient, tmp_path: Path) -> None:
    call_id = upload(client).json()["call_id"]
    assert client.delete(f"/calls/{call_id}").status_code == 204
    assert list((tmp_path / "uploads").iterdir()) == []
    assert client.get(f"/calls/{call_id}").status_code == 404
    assert client.delete(f"/calls/{call_id}").status_code == 404


def test_unknown_and_malformed_ids(client: TestClient) -> None:
    missing = uuid.uuid4()
    assert client.get(f"/calls/{missing}").status_code == 404
    assert client.get(f"/calls/{missing}/audio").status_code == 404
    assert client.get("/calls/not-a-uuid").status_code == 422


def test_serves_the_dashboard(db_url: str, tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>TunCall</h1>")
    cfg = ServeConfig(storage_dir=tmp_path / "u", web_dir=web, allowed_hosts=["testserver"])
    with TestClient(create_app(cfg, db_url)) as c:
        assert "TunCall" in c.get("/").text
        assert c.get("/calls").json() == []  # API routes win over the static mount


def test_api_never_imports_torch() -> None:
    code = "import sys, tuncall.serve.app; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
