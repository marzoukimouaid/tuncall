"""TunSwitch weakly labelled (D-12d): ~159 h, automatic labels, ablations only.

Read from our private derived repo, built on Colab by ``scripts/prepare_noisy.py``
from Zenodo 8370566 ``weakly_labeled_tn.zip`` (md5 recorded in the repo card, D-12h)."""

from collections.abc import Iterator
from pathlib import Path

from tuncall.data.sources.common import Record, derived

REPO = "mouaidMrz/tuncall-tunswitch-weak"


def records(split: str, revision: str, root: Path | None = None) -> Iterator[Record]:
    return derived("tunswitch_weak", split, REPO, revision, root)
