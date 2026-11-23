import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from prepare_noisy import card, clean, download, prepare

from tuncall.data.sources import SOURCES


def wav(seconds: float, sr: int = 44_100) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, 0.1 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("text", "seconds", "expected"),
    [
        ("ايه <pause> باهي <laugh>", 2.0, ("ايه باهي", None)),
        ("العالمية <eps>", 2.0, ("العالمية", None)),
        ("<pause> <breath>", 2.0, ("", "empty")),
        (" ".join(["نحكي"] * 30), 20.0, (" ".join(["نحكي"] * 30), "looping")),
        ("ب" * 60, 2.0, ("ب" * 60, "too many chars/s")),
        (" ".join(["بَرْشَة"] * 7), 2.0, (" ".join(["بَرْشَة"] * 7), None)),  # diacritics don't count
    ],
)
def test_clean_strips_tags_and_flags_broken_text(
    text: str, seconds: float, expected: tuple[str, str | None]
) -> None:
    assert clean(text, seconds) == expected


def test_prepare_round_trips_through_the_adapter(tmp_path: Path) -> None:
    clips = [
        ("vid_chunk_0", wav(1.0), "ايه <pause> باهي"),
        ("vid_chunk_1", wav(0.5), "<laugh>"),  # dropped: empty
        ("vid_chunk_2", wav(2.0, 16_000), "نحكي عليك توا"),
    ]
    export = tmp_path / "exports" / "deep_confessions-train.jsonl"
    stats = prepare("deep_confessions", clips, tmp_path / "out", export)
    assert stats == {"kept": 2, "dropped: empty": 1, "shards": 1, "kept seconds": 3}

    records = list(SOURCES["deep_confessions"]("train", "0" * 40, root=tmp_path / "out"))
    assert [r.id for r in records] == [
        "deep_confessions/train/vid_chunk_0",
        "deep_confessions/train/vid_chunk_2",
    ]
    assert [r.text for r in records] == ["ايه باهي", "نحكي عليك توا"]
    assert [len(r.audio) for r in records] == [16_000, 32_000]
    assert all(r.audio.dtype == np.float32 for r in records)

    lines = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
    assert [(x["id"], x["text"], x["duration"]) for x in lines] == [
        (r.id, r.text, r.duration) for r in records
    ]


def test_card_records_upstream_and_counts() -> None:
    text = card("tunswitch_weak", {"kept": 5, "dropped: empty": 1, "shards": 1})  # type: ignore[arg-type]
    assert "license: cc-by-4.0" in text
    assert "231674424efccfd4d3f19b2706bafd52" in text
    assert "| kept | 5 |" in text
    assert "shards" not in text


PAYLOAD = bytes(range(256)) * 400  # 100 KB


class FlakyRanges(BaseHTTPRequestHandler):
    """Serves PAYLOAD with Range support; the first response stops halfway, silently."""

    calls = 0

    def do_GET(self) -> None:
        type(self).calls += 1
        start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
        if start >= len(PAYLOAD):
            self.send_response(416)
            self.end_headers()
            return
        body = PAYLOAD[start:]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[: len(body) // 2] if type(self).calls == 1 else body)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    FlakyRanges.calls = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FlakyRanges)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/f.zip"
    httpd.shutdown()


def test_download_resumes_a_transfer_that_ends_early(server: str, tmp_path: Path) -> None:
    dest = tmp_path / "f.zip"
    download(server, dest)
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "f.zip.part").exists()
    assert FlakyRanges.calls == 2


def test_download_finishes_a_complete_part_file(server: str, tmp_path: Path) -> None:
    (tmp_path / "f.zip.part").write_bytes(PAYLOAD)
    FlakyRanges.calls = 1  # no short first response this time
    download(server, tmp_path / "f.zip")
    assert (tmp_path / "f.zip").read_bytes() == PAYLOAD
