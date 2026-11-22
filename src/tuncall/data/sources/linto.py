"""LinTO Tunisian audio, every subset, one record per segment (D-12, D-14a)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, parquet_files, segments

REPO = "linagora/linto-dataset-audio-ar-tn"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/*/{split}/*.parquet", root)
    return segments("linto", split, paths, subsets=True)
