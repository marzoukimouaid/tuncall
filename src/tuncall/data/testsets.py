"""Frozen test sets (D-13, D-14).

A test set is a pinned Hub revision plus a committed manifest (``configs/data/*.yaml``)
holding the sha256 of every parquet file and a digest of how utterances are cut from
them. Loading verifies both, so the audio and the references can't change silently.

An utterance is one timestamped segment of a recording; its reference is the segment's
cleaned ``transcript`` field (D-14a). Audio is downmixed to mono and resampled to 16 kHz.
"""

import hashlib
import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download

from tuncall.config import StrictModel

SAMPLE_RATE = 16_000


class ParquetFile(StrictModel):
    path: str
    sha256: str


class Manifest(StrictModel):
    name: str
    repo: str
    revision: str
    files: list[ParquetFile]
    utterances: int
    hours: float
    digest: str


@dataclass(frozen=True)
class Segment:
    id: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Utterance:
    id: str
    audio: np.ndarray  # float32 mono at SAMPLE_RATE
    text: str


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def iter_rows(files: Mapping[str, Path]) -> Iterator[tuple[list[Segment], bytes]]:
    """Yield (segments, encoded audio) per recording, in manifest file order."""
    for rel, local in files.items():
        subset = Path(rel).parts[1]  # data/<subset>/test/<file>.parquet
        for row in pq.read_table(local, columns=["audio_id", "audio", "segments"]).to_pylist():
            segments = [
                Segment(
                    f"{subset}/{row['audio_id']}/{i:03d}", s["start"], s["end"], s["transcript"]
                )
                for i, s in enumerate(row["segments"])
            ]
            yield segments, row["audio"]["bytes"]


def segment_digest(segments: list[Segment]) -> str:
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{s.id}\t{s.start!r}\t{s.end!r}\t{s.text}\n".encode())
    return h.hexdigest()


def fetch(manifest: Manifest, root: Path | None = None) -> dict[str, Path]:
    """Local path of each manifest file: under ``root`` if given, else from the Hub cache."""
    if root is not None:
        return {f.path: root / f.path for f in manifest.files}
    return {
        f.path: Path(
            hf_hub_download(manifest.repo, f.path, repo_type="dataset", revision=manifest.revision)
        )
        for f in manifest.files
    }


def decode_segment(audio: np.ndarray, sr: int, seg: Segment) -> np.ndarray:
    clip = audio[round(seg.start * sr) : round(seg.end * sr)].mean(axis=1)
    if sr != SAMPLE_RATE:
        clip = soxr.resample(clip, sr, SAMPLE_RATE)
    return clip.astype(np.float32)


def load_test_set(manifest: Manifest, root: Path | None = None) -> list[Utterance]:
    """Verify the frozen files and cuts, then decode every utterance to 16 kHz mono."""
    files = fetch(manifest, root)
    for f in manifest.files:
        actual = sha256_file(files[f.path])
        if actual != f.sha256:
            raise ValueError(f"{manifest.name}: {f.path} sha256 {actual} != frozen {f.sha256}")

    utterances: list[Utterance] = []
    all_segments: list[Segment] = []
    for segments, encoded in iter_rows(files):
        audio, sr = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
        for seg in segments:
            utterances.append(Utterance(seg.id, decode_segment(audio, sr, seg), seg.text))
        all_segments += segments

    digest = segment_digest(all_segments)
    if len(all_segments) != manifest.utterances or digest != manifest.digest:
        raise ValueError(
            f"{manifest.name}: {len(all_segments)} utterances / digest {digest} "
            f"!= frozen {manifest.utterances} / {manifest.digest}"
        )
    return utterances
