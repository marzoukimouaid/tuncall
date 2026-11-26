"""Build the train and dev audio caches for ASR fine-tuning (D-21c).

Train = every train clip of the mix minus the D-16e exclusions and the LinTO dev hold-out.
Dev = the LinTO hold-out plus each ``dev_splits`` source's own validation split (D-17a).
Both are written under ``<cache>/train`` and ``<cache>/dev``; run once per mix change.

    uv run python scripts/prepare_asr_data.py --config configs/phase3/lora.yaml
"""

import argparse
import itertools
from collections.abc import Iterator
from pathlib import Path

from tuncall.asr.cache import build
from tuncall.asr.train import TrainConfig
from tuncall.config import load_config
from tuncall.data.lexicon import load_lexicon
from tuncall.data.sources import SOURCES, Record, TrainMix, load_exclusions

ROOT = Path(__file__).resolve().parents[1]


def split_mix(
    cfg: TrainConfig, mix: TrainMix, limit: int | None
) -> tuple[Iterator[Record], Iterator[Record]]:
    """(train, dev) record streams; ``limit`` keeps the first N clips of each read (pre-flight)."""
    if any(e.weight != 1.0 for e in mix.sources):
        raise ValueError("source weights other than 1.0 are not implemented yet")
    excluded = load_exclusions(ROOT / mix.exclude) if mix.exclude else frozenset()
    dev_ids = load_exclusions(ROOT / cfg.dev_holdout)
    revisions = {e.source: e.revision for e in mix.sources}

    def read(source: str, split: str) -> Iterator[Record]:
        return itertools.islice(SOURCES[source](split, revision=revisions[source]), limit)

    held_out: list[Record] = []  # ~500 clips, kept in memory so LinTO is decoded once

    def train() -> Iterator[Record]:
        for e in mix.sources:
            for r in read(e.source, "train"):
                if r.id in dev_ids:
                    held_out.append(r)
                elif r.id not in excluded:
                    yield r

    def dev() -> Iterator[Record]:  # consume only after train()
        if limit is None and len(held_out) != len(dev_ids):
            raise ValueError(f"{len(dev_ids) - len(held_out)} dev hold-out ids not found in LinTO")
        yield from held_out
        for source, split in cfg.dev_splits.items():
            yield from read(source, split)

    return train(), dev()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first N clips per source read (pre-flight)")
    args = parser.parse_args()
    cfg = load_config(args.config, TrainConfig)
    mix = load_config(ROOT / cfg.mix, TrainMix)
    lexicon = load_lexicon(ROOT / cfg.lexicon)
    out = cfg.cache_path() / (f"limit-{args.limit}" if args.limit else "")
    train, dev = split_mix(cfg, mix, args.limit)
    for name, records in (("train", train), ("dev", dev)):
        print(f"{name}: {build(records, out / name, lexicon)} -> {out / name}", flush=True)


if __name__ == "__main__":
    main()
