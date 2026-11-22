"""Omnilingual ASR corpus, config ``aeb_Arab`` only (D-12c)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, decode, parquet_files, record, rows

REPO = "facebook/omnilingual-asr-corpus"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/aeb_Arab/{split}-*.parquet", root)
    columns = ["speaker_id", "prompt_id", "segment_id", "audio", "raw_text"]
    for row in rows(paths, columns):
        rid = f"{row['speaker_id']}/{row['prompt_id']}/{row['segment_id']}"
        yield record(
            "omnilingual",
            split,
            rid,
            decode(row["audio"]["bytes"]),
            row["raw_text"],
            row["speaker_id"],
        )
