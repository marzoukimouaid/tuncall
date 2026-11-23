"""Deep Confessions podcast (D-12d): ~175 h, AI-assisted transcripts, ablations only.

Read from our private derived repo, built on Colab by ``scripts/prepare_noisy.py``
(tags stripped, broken transcripts dropped, D-12h); the card there records the upstream
commit."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, derived

REPO = "mouaidMrz/tuncall-deep-confessions"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    return derived("deep_confessions", split, REPO, revision, root)
