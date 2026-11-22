"""senior_tunisian_voice: LinTO-style segmented recordings (D-12c)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, parquet_files, segments

REPO = "Fares11/senior_tunisian_voice"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/{split}-*.parquet", root)
    return segments("senior", split, paths)
