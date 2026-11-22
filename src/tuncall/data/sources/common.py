"""Shared parts of the source adapters (D-16): the common record, file lookup, decoding."""

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import snapshot_download

from tuncall.data.testsets import SAMPLE_RATE


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


def decode(encoded: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return mono.astype(np.float32)


def record(id: str, audio: np.ndarray, text: str, speaker: str | None = None) -> Record:
    source, split, _ = id.split("/", 2)
    return Record(id, source, split, audio, text, speaker, len(audio) / SAMPLE_RATE)


def clips(source: str, split: str, paths: list[Path], text: str) -> Iterator[Record]:
    """One record per row: the row's whole audio and its ``text`` column."""
    for i, row in enumerate(rows(paths, ["audio", text])):
        yield record(f"{source}/{split}/{i:06d}", decode(row["audio"]["bytes"]), row[text])


def segments(prefix: str, paths: list[Path]) -> Iterator[Record]:
    """One record per timestamped segment of each recording (LinTO layout, D-14a).

    ``prefix`` may contain ``{subset}``, the folder under ``data/``.
    """
    for path in paths:
        subset = path.parent.parent.name
        for row in rows([path], ["audio_id", "audio", "segments"]):
            audio = decode(row["audio"]["bytes"])
            for i, s in enumerate(row["segments"]):
                clip = audio[round(s["start"] * SAMPLE_RATE) : round(s["end"] * SAMPLE_RATE)]
                rid = f"{prefix.format(subset=subset)}/{row['audio_id']}/{i:03d}"
                yield record(rid, clip, s["transcript"])
