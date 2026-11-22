"""Utterances and hours per source of the training mix (D-16).

Decodes every clip of each source's split at the pinned revision in
configs/data/train-mix.yaml, so the first run downloads the sources (S-01c, N-01).

    uv run python scripts/source_stats.py [--split train] [--limit N]
"""

import argparse
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from tuncall.config import load_config
from tuncall.data.sources import SOURCES, Record, TrainMix

MIX = Path(__file__).resolve().parents[1] / "configs" / "data" / "train-mix.yaml"


def table(records: Iterable[Record]) -> str:
    """Markdown table of utterances and hours per source, in first-seen order, plus a total."""
    count: dict[str, int] = defaultdict(int)
    seconds: dict[str, float] = defaultdict(float)
    for r in records:
        count[r.source] += 1
        seconds[r.source] += r.duration
    lines = ["| source | utterances | hours |", "|---|---:|---:|"]
    lines += [f"| {s} | {count[s]} | {seconds[s] / 3600:.3f} |" for s in count]
    total = sum(seconds.values()) / 3600
    lines.append(f"| total | {sum(count.values())} | {total:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MIX)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, help="first N clips per source (quick check)")
    args = parser.parse_args()

    mix = load_config(args.config, TrainMix)
    records = (
        r
        for e in mix.sources
        for r in islice(SOURCES[e.source](args.split, e.revision), args.limit)
    )
    print(table(records))


if __name__ == "__main__":
    main()
