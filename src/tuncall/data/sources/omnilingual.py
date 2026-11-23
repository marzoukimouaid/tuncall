"""Omnilingual ASR corpus, config ``aeb_Arab`` only (D-12c).

Its texts carry non-speech tags (``<hesitation>``, ``<noise>``, malformed ``<<laugh>``,
empty ``<>``); they are stripped and clips left without text are skipped (D-12g).
"""

import re
from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, decode, parquet_files, record, rows

REPO = "facebook/omnilingual-asr-corpus"
_TAG = re.compile(r"<+[^<>]*>+|[<>]")  # also stray brackets left by broken tags


def strip_tags(text: str) -> str:
    return " ".join(_TAG.sub(" ", text).split())


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    paths = parquet_files(REPO, revision, f"data/aeb_Arab/{split}-*.parquet", root)
    columns = ["speaker_id", "prompt_id", "segment_id", "audio", "raw_text"]
    for row in rows(paths, columns):
        text = strip_tags(row["raw_text"])
        if not text:
            continue
        spk = row["speaker_id"]
        key = f"{spk}/{row['prompt_id']}/{row['segment_id']}"
        yield record("omnilingual", split, key, decode(row["audio"]["bytes"]), text, spk)
