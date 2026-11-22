import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from pydantic import ValidationError

from tuncall.config import load_config
from tuncall.data.sources import SOURCES, SPLITS, Record, TrainMix

MIX = Path(__file__).resolve().parents[1] / "configs" / "data" / "train-mix.yaml"


def audio(
    seconds: float, sr: int = 16_000, channels: int = 1, name: str = "x.wav"
) -> dict[str, object]:
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone] * channels, axis=1), sr, format="WAV")
    return {"bytes": buf.getvalue(), "path": name}


def write(root: Path, rel: str, rows: list[dict[str, object]]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load(source: str, split: str, root: Path) -> list[Record]:
    return list(SOURCES[source](split, "0" * 40, root=root))


def check_audio(records: list[Record], seconds: list[float]) -> None:
    assert [len(r.audio) for r in records] == [round(s * 16_000) for s in seconds]
    assert [r.duration for r in records] == seconds
    assert all(r.audio.ndim == 1 and r.audio.dtype == np.float32 for r in records)


def test_slurp_tn_yields_one_record_per_row(tmp_path: Path) -> None:
    write(
        tmp_path,
        "data/train-00000-of-00001.parquet",
        [
            {
                "audio": audio(1.0, 48_000, 2, "5172-13.wav"),
                "tun_transcription": "ابعث mail",
                "intent": "x",
            },
            {
                "audio": audio(0.5, name="5891-13.wav"),
                "tun_transcription": "شنوة الطقس",
                "intent": "y",
            },
        ],
    )
    write(tmp_path, "data/test-00000-of-00001.parquet", [])  # other splits are ignored
    records = load("slurp_tn", "train", tmp_path)
    assert [r.id for r in records] == ["slurp_tn/train/5172-13", "slurp_tn/train/5891-13"]
    assert [r.text for r in records] == ["ابعث mail", "شنوة الطقس"]
    assert {(r.source, r.split, r.speaker) for r in records} == {("slurp_tn", "train", None)}
    check_audio(records, [1.0, 0.5])


def test_tedxtn_reads_sentence(tmp_path: Path) -> None:
    write(
        tmp_path,
        "data/train-00000-of-00005.parquet",
        [
            {
                "audio": audio(0.25, name="talk_0_250.wav"),
                "sentence": "soi disant اللي",
                "duration": 0.25,
            }
        ],
    )
    write(
        tmp_path,
        "data/train-00001-of-00005.parquet",
        [{"audio": audio(0.5, name="talk_250_750.wav"), "sentence": "donc مشينا", "duration": 0.5}],
    )
    records = load("tedxtn", "train", tmp_path)
    assert [r.id for r in records] == ["tedxtn/train/talk_0_250", "tedxtn/train/talk_250_750"]
    assert [r.text for r in records] == ["soi disant اللي", "donc مشينا"]
    check_audio(records, [0.25, 0.5])


def test_omnilingual_reads_aeb_arab_only_with_speakers(tmp_path: Path) -> None:
    def row(spk: str, seg: str, text: str) -> dict[str, object]:
        return {
            "language": "aeb_Arab",
            "speaker_id": spk,
            "prompt_id": "a001",
            "segment_id": seg,
            "audio": audio(0.5),
            "duration": 0.5,
            "raw_text": text,
        }

    rows = [
        row("spk01", "s01", "<hesitation> الكلب  <<noise> يحب <> >"),
        row("spk01", "s02", "<hesitation> <laugh>"),  # nothing left: skipped
    ]
    write(tmp_path, "data/aeb_Arab/train-00000-of-00006.parquet", rows)
    write(tmp_path, "data/ary_Arab/train-00000-of-00001.parquet", [row("spk09", "s01", "no")])
    records = load("omnilingual", "train", tmp_path)
    assert [(r.id, r.speaker, r.text) for r in records] == [
        ("omnilingual/train/spk01/a001/s01", "spk01", "الكلب يحب")
    ]
    check_audio(records, [0.5])


def segmented_row(audio_id: str, spans: list[tuple[float, float, str]]) -> dict[str, object]:
    return {
        "audio_id": audio_id,
        "audio": audio(spans[-1][1], 44_100, 2),
        "segments": [
            {"start": s, "end": e, "transcript": t, "transcript_raw": t + "!"} for s, e, t in spans
        ],
        "transcript": " ".join(t for _, _, t in spans),
    }


def test_linto_cuts_segments_from_every_subset(tmp_path: Path) -> None:
    write(
        tmp_path,
        "data/AmenyKH/train/train-00000-of-00002.parquet",
        [segmented_row("a", [(0.0, 1.0, "عسلامة"), (1.5, 2.0, "لاباس")])],
    )
    write(
        tmp_path,
        "data/MASC/train/train-00000-of-00001.parquet",
        [segmented_row("b", [(0, 1, "باهي")])],
    )
    write(
        tmp_path, "data/MASC/test/test-00000-of-00001.parquet", [segmented_row("c", [(0, 1, "x")])]
    )
    records = load("linto", "train", tmp_path)
    assert [r.id for r in records] == [
        "linto/train/AmenyKH/a/000",
        "linto/train/AmenyKH/a/001",
        "linto/train/MASC/b/000",
    ]
    assert [r.text for r in records] == ["عسلامة", "لاباس", "باهي"]
    check_audio(records, [1.0, 0.5, 1.0])


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_missing_split_is_refused(source: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no parquet files"):
        load(source, "nope", tmp_path)


def test_train_mix_config_loads() -> None:
    mix = load_config(MIX, TrainMix)
    assert sorted(e.source for e in mix.sources) == sorted(SOURCES)
    assert all(e.weight == 1.0 for e in mix.sources)


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"source": "nope", "revision": "a" * 40, "weight": 1.0}, "unknown source"),
        ({"source": "linto", "revision": "a" * 40, "weight": 0.0}, "greater than 0"),
        ({"source": "linto", "revision": "main", "weight": 1.0}, "pattern"),
    ],
)
def test_train_mix_rejects_bad_entries(entry: dict[str, object], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        TrainMix.model_validate({"sources": [entry]})


def test_every_source_lists_its_splits() -> None:
    assert sorted(SPLITS) == sorted(SOURCES)
