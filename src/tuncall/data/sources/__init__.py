"""Source adapters (D-16): each reads one public corpus at a pinned Hub revision and
yields the common ``Record``. Adding a source is one module plus one line here."""

from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import Field, field_validator

from tuncall.config import StrictModel
from tuncall.data.sources import linto, omnilingual, slurp_tn, tedxtn
from tuncall.data.sources.common import Record

SOURCES: dict[str, Callable[..., Iterator[Record]]] = {
    "linto": linto.records,
    "slurp_tn": slurp_tn.records,
    "omnilingual": omnilingual.records,
    "tedxtn": tedxtn.records,
}

# Every split each source publishes. Only train is trained on (D-12c); the others are
# read for the dedupe report (D-16). linto/test is test-general, slurp_tn/test test-domain.
SPLITS: dict[str, tuple[str, ...]] = {
    "linto": ("train", "test"),
    "slurp_tn": ("train", "validation", "test"),
    "omnilingual": ("train",),
    "tedxtn": ("train", "validation", "test"),
}


class MixEntry(StrictModel):
    source: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # a Hub commit sha, never a branch
    weight: float = Field(gt=0)  # relative sampling weight; its use is decided in Phase 3

    @field_validator("source")
    @classmethod
    def known_source(cls, v: str) -> str:
        if v not in SOURCES:
            raise ValueError(f"unknown source {v!r}; known: {sorted(SOURCES)}")
        return v


class TrainMix(StrictModel):
    """The training mix (D-12c, D-16): ``configs/data/train-mix.yaml``."""

    sources: list[MixEntry]
    # Clip ids never trained on (D-16e), one per line; relative to the repo root.
    exclude: Path | None = None


def load_exclusions(path: Path) -> frozenset[str]:
    """Clip ids listed in an exclusion file (blank lines and ``#`` comments ignored)."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))


__all__ = ["SOURCES", "SPLITS", "MixEntry", "Record", "TrainMix", "load_exclusions"]
