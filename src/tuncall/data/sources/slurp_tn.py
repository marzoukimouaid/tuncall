"""SLURP-TN: task-oriented requests, one clip per row (D-12c, D-13a)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, clips, parquet_files

REPO = "Elyadata/SLURP-TN"
REVISION = "a5c2ec062fa91b5a51b7c914859eb1e6525a6f5d"


def records(split: str, *, revision: str = REVISION, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/{split}-*.parquet", root)
    return clips("slurp_tn", split, paths, "tun_transcription")
