"""SLURP-TN: task-oriented requests, one clip per row (D-12c, D-13a)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, clips, parquet_files

REPO = "Elyadata/SLURP-TN"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/{split}-*.parquet", root)
    return clips("slurp_tn", split, paths, "tun_transcription")
