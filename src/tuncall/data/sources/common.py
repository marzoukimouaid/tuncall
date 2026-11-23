"""Shared parts of the source adapters (D-16): the common record, file lookup, decoding."""

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

from tuncall.data.testsets import SAMPLE_RATE, Segment, decode_segment


@dataclass(frozen=True)
class Record:
    id: str  # "<source>/<split>/..."
    source: str
    split: str
    audio: np.ndarray  # float32 mono at SAMPLE_RATE
    text: str  # as the source writes it; D-10a transliteration is applied later
    speaker: str | None
    duration: float  # seconds


def parquet_files(repo: str, revision: str, pattern: str, root: Path | None) -> list[Path]:
    """Files matching ``pattern`` under ``root``, or in the Hub cache at ``revision``."""
    if root is None:
        root = Path(
            snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=pattern)
        )
    paths = sorted(root.glob(pattern))
    if not paths:
        raise ValueError(f"{repo}@{revision}: no parquet files match {pattern}")
    return paths


def rows(paths: list[Path], columns: list[str]) -> Iterator[dict[str, Any]]:
    """Rows of every file in order, a batch at a time so big shards don't fill memory."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def read(encoded: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    return audio, sr


def decode(encoded: bytes) -> np.ndarray:
    """Whole clip as float32 mono at SAMPLE_RATE."""
    audio, sr = read(encoded)
    return decode_segment(audio, sr, Segment("", 0.0, len(audio) / sr, ""))


def record(
    source: str, split: str, key: str, audio: np.ndarray, text: str, speaker: str | None = None
) -> Record:
    rid = f"{source}/{split}/{key}"
    return Record(rid, source, split, audio, text, speaker, len(audio) / SAMPLE_RATE)


def clips(source: str, split: str, paths: list[Path], text: str) -> Iterator[Record]:
    """One record per row: the row's whole audio and its ``text`` column.

    The key is the upstream file name (``audio.path``) without extension, so ids
    trace back to the source and survive shard reordering.
    """
    for row in rows(paths, ["audio", text]):
        key = Path(row["audio"]["path"]).stem
        yield record(source, split, key, decode(row["audio"]["bytes"]), row[text])


def segments(source: str, split: str, paths: list[Path], subsets: bool = False) -> Iterator[Record]:
    """One record per timestamped segment of each recording (LinTO layout, D-14a).

    Cut exactly like test-general (``decode_segment``). With ``subsets``, the folder
    under ``data/`` (LinTO's subset) starts the key.
    """
    for path in paths:
        prefix = f"{path.parent.parent.name}/" if subsets else ""
        for row in rows([path], ["audio_id", "audio", "segments"]):
            audio, sr = read(row["audio"]["bytes"])
            for i, s in enumerate(row["segments"]):
                seg = Segment("", s["start"], s["end"], s["transcript"])
                key = f"{prefix}{row['audio_id']}/{i:03d}"
                yield record(source, split, key, decode_segment(audio, sr, seg), seg.text)


def derived(
    source: str, split: str, repo: str, revision: str, root: Path | None
) -> Iterator[Record]:
    """Our own derived repos (D-12d, ``scripts/prepare_noisy.py``): text already cleaned,
    audio 16 kHz mono FLAC, the upstream key in ``key``."""
    paths = parquet_files(repo, revision, f"data/{split}-*.parquet", root)
    for row in rows(paths, ["key", "audio", "text"]):
        yield record(source, split, row["key"], decode(row["audio"]["bytes"]), row["text"])
