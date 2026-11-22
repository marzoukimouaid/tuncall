"""senior_tunisian_voice: LinTO-style segmented recordings (D-12c)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, parquet_files, segments

REPO = "Fares11/senior_tunisian_voice"
REVISION = "670ed90cbeac65af8f1fd68cce3e1e28355a5892"


def records(split: str, *, revision: str = REVISION, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/{split}-*.parquet", root)
    return segments(f"senior/{split}", paths)
