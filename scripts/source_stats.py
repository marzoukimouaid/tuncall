"""Utterances and hours per source and split of the training mix, plus a text export (D-16).

Decodes every clip of every split (``SPLITS``) of each source at the pinned revision in
configs/data/train-mix.yaml, so the first run downloads ~14 GB: run it on Colab (D-16b).
Prints a markdown table and writes ``<out>/<source>-<split>.jsonl`` with one line per
clip ``{id, text, duration, speaker}``; the lexicon and dedupe scripts read those files.

    uv run python scripts/source_stats.py [--out outputs/sources] [--limit N]
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any, TextIO

from tuncall.config import load_config
from tuncall.data.sources import SOURCES, SPLITS, Record, TrainMix

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "configs" / "data" / "train-mix.yaml"
OUT = ROOT / "outputs" / "sources"


def table(records: Iterable[Record]) -> str:
    """Markdown table of utterances and hours per source and split, in first-seen order."""
    count: dict[tuple[str, str], int] = defaultdict(int)
    seconds: dict[tuple[str, str], float] = defaultdict(float)
    for r in records:
        count[r.source, r.split] += 1
        seconds[r.source, r.split] += r.duration
    lines = ["| source | split | utterances | hours |", "|---|---|---:|---:|"]
    lines += [f"| {s} | {sp} | {count[s, sp]} | {seconds[s, sp] / 3600:.3f} |" for s, sp in count]
    total = sum(seconds.values()) / 3600
    lines.append(f"| total | | {sum(count.values())} | {total:.3f} |")
    return "\n".join(lines)


def export(records: Iterable[Record], out: Path) -> Iterator[Record]:
    """Pass records through, writing each one's text line to ``out/<source>-<split>.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        files: dict[str, TextIO] = {}
        for r in records:
            name = f"{r.source}-{r.split}.jsonl"
            if name not in files:
                files[name] = stack.enter_context((out / name).open("w", encoding="utf-8"))
            line = {"id": r.id, "text": r.text, "duration": r.duration, "speaker": r.speaker}
            files[name].write(json.dumps(line, ensure_ascii=False) + "\n")
            yield r


def read_export(out: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """(``<source>-<split>``, row) for every line ``export`` wrote, files in name order."""
    for path in sorted(out.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield path.stem, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MIX)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, help="first N clips per split (quick check)")
    args = parser.parse_args()

    mix = load_config(args.config, TrainMix)
    records = (
        r
        for e in mix.sources
        for split in SPLITS[e.source]
        for r in islice(SOURCES[e.source](split, e.revision), args.limit)
    )
    print(table(export(records, args.out)))


if __name__ == "__main__":
    main()
