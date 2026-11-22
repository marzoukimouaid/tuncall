import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import yaml

from tuncall.config import load_config
from tuncall.data.testsets import (
    SAMPLE_RATE,
    LexiconRef,
    Manifest,
    ParquetFile,
    fetch,
    iter_cuts,
    iter_rows,
    load_test_set,
    segment_digest,
    sha256_file,
)

REL = "data/Tiny/test/test-00000-of-00001.parquet"


def wav_bytes(seconds: float, sr: int, channels: int) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone] * channels, axis=1), sr, format="WAV")
    return buf.getvalue()


def seg(start: float, end: float, text: str) -> dict[str, object]:
    return {"start": start, "end": end, "transcript": text, "transcript_raw": text}


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Manifest, Path]:
    """A one-file test set (48 kHz stereo + 16 kHz mono) and its manifest."""
    path = tmp_path / REL
    path.parent.mkdir(parents=True)
    rows = [
        {
            "audio_id": "a",
            "audio": {"bytes": wav_bytes(3, 48_000, 2), "path": "a.wav"},
            "segments": [seg(0.0, 1.0, "عسلامة"), seg(1.5, 3.0, "لاباس")],
            "transcript": "عسلامة لاباس",
        },
        {
            "audio_id": "b",
            "audio": {"bytes": wav_bytes(2, SAMPLE_RATE, 1), "path": "b.wav"},
            "segments": [seg(0.0, 2.0, "باهي")],
            "transcript": "باهي",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    segments = [s for segs, _ in iter_rows({REL: path}) for s in segs]
    manifest = Manifest(
        name="tiny",
        repo="local/tiny",
        revision="0" * 40,
        files=[ParquetFile(path=REL, sha256=sha256_file(path))],
        utterances=len(segments),
        hours=0.0,
        digest=segment_digest(segments),
    )
    return manifest, tmp_path


def test_load_cuts_segments_to_16k_mono(frozen: tuple[Manifest, Path]) -> None:
    manifest, root = frozen
    utts = load_test_set(manifest, root)
    assert [u.id for u in utts] == ["Tiny/a/000", "Tiny/a/001", "Tiny/b/000"]
    assert [u.text for u in utts] == ["عسلامة", "لاباس", "باهي"]
    assert [len(u.audio) for u in utts] == [16_000, 24_000, 32_000]
    assert all(u.audio.ndim == 1 and u.audio.dtype == np.float32 for u in utts)


def test_manifest_round_trips_through_yaml(frozen: tuple[Manifest, Path], tmp_path: Path) -> None:
    manifest, _ = frozen
    out = tmp_path / "m.yaml"
    out.write_text(yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True))
    assert load_config(out, Manifest) == manifest


def test_changed_file_is_rejected(frozen: tuple[Manifest, Path]) -> None:
    manifest, root = frozen
    with (root / REL).open("ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="sha256"):
        load_test_set(manifest, root)


def test_changed_cut_is_rejected(frozen: tuple[Manifest, Path]) -> None:
    manifest, root = frozen
    with pytest.raises(ValueError, match="digest"):
        load_test_set(manifest.model_copy(update={"digest": "0" * 64}), root)


def test_fetch_with_root_uses_local_files(frozen: tuple[Manifest, Path]) -> None:
    manifest, root = frozen
    assert fetch(manifest, root) == {REL: root / REL}


CLIPS = "data/test-00000-of-00001.parquet"


@pytest.fixture
def frozen_clips(tmp_path: Path) -> tuple[Manifest, Path]:
    """A SLURP-TN-like test set: one clip per row, Latin words mapped by a lexicon."""
    path = tmp_path / CLIPS
    path.parent.mkdir(parents=True)
    rows = [
        {"audio": {"bytes": wav_bytes(1, 48_000, 2), "path": "16501-9.wav"}, "t": "ابعث Mail"},
        {"audio": {"bytes": wav_bytes(0.5, SAMPLE_RATE, 1), "path": "77-1.wav"}, "t": "شنوة"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    lexicon = tmp_path / "lexicon.tsv"
    lexicon.write_text("mail\tميل\n", encoding="utf-8")
    draft = Manifest(
        name="tiny-clips",
        repo="local/tiny",
        revision="0" * 40,
        files=[ParquetFile(path=CLIPS, sha256=sha256_file(path))],
        utterances=0,
        hours=0.0,
        digest="",
        layout="clips",
        text_column="t",
        lexicon=LexiconRef(path=str(lexicon), sha256=sha256_file(lexicon)),
    )
    segments = [s for segs, _ in iter_cuts(draft, {CLIPS: path}) for s in segs]
    return draft.model_copy(
        update={"utterances": len(segments), "digest": segment_digest(segments)}
    ), tmp_path


def test_clips_load_whole_rows_with_transliterated_references(
    frozen_clips: tuple[Manifest, Path],
) -> None:
    manifest, root = frozen_clips
    utts = load_test_set(manifest, root)
    assert [(u.id, u.text) for u in utts] == [("16501-9", "ابعث ميل"), ("77-1", "شنوة")]
    assert [len(u.audio) for u in utts] == [16_000, 8_000]


def test_changed_lexicon_is_rejected(frozen_clips: tuple[Manifest, Path]) -> None:
    manifest, root = frozen_clips
    assert manifest.lexicon is not None
    Path(manifest.lexicon.path).write_text("mail\tميلل\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lexicon"):
        load_test_set(manifest, root)
