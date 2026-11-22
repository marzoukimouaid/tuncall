"""Omnilingual ASR corpus, config ``aeb_Arab`` only (D-12c)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, decode, parquet_files, record, rows

REPO = "facebook/omnilingual-asr-corpus"
REVISION = "8648ba8946377697b427ae952076e49fc0e5e44d"


def records(split: str, *, revision: str = REVISION, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/aeb_Arab/{split}-*.parquet", root)
    columns = ["speaker_id", "prompt_id", "segment_id", "audio", "raw_text"]
    for row in rows(paths, columns):
        rid = f"omnilingual/{split}/{row['speaker_id']}/{row['prompt_id']}/{row['segment_id']}"
        yield record(rid, decode(row["audio"]["bytes"]), row["raw_text"], row["speaker_id"])
