"""TEDxTN talks via the segmented mirror, one clip per row (D-12c)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, clips, parquet_files

REPO = "Hediske/tedxtn-tunisian-segmented"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/{split}-*.parquet", root)
    return clips("tedxtn", split, paths, "sentence")
