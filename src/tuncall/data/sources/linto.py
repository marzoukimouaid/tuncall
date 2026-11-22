"""LinTO Tunisian audio, every subset, one record per segment (D-12, D-14a)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, parquet_files, segments

REPO = "linagora/linto-dataset-audio-ar-tn"
REVISION = "99796bfd5e8b48febaac60633d1ec97a658fa511"  # same as test-general (D-14)


def records(split: str, *, revision: str = REVISION, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/*/{split}/*.parquet", root)
    return segments(f"linto/{split}/{{subset}}", paths)
