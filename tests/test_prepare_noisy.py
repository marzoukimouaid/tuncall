import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from prepare_noisy import card, clean, prepare

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
