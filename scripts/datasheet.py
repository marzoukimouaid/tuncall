"""Numbers for docs/datasheet.md (P-10, #40), from the ``source_stats.py`` text export.

Rewrites two generated blocks of the datasheet and leaves its prose alone:

- ``datasheet-composition``: per source and split, utterances, hours, clip lengths,
  clips over 30 s (Whisper's input limit), speakers, clips with Latin-script words
  (before the D-10a lexicon) and train clips dropped by the dedupe exclusions (D-16e).
- ``datasheet-mix``: what the training mix contains after those exclusions.

Sources named in ``PENDING`` have no export yet; they are listed with their issue.

    uv run python scripts/datasheet.py [--exports outputs/sources]
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from source_stats import MIX, OUT, ROOT, read_export

from tuncall.config import load_config
from tuncall.data.lexicon import LATIN
from tuncall.data.sources import TrainMix, load_exclusions
from tuncall.eval.report import replace_block

DATASHEET = ROOT / "docs" / "datasheet.md"
WHISPER_MAX_S = 30.0
# Noisy-label sources (D-12d) not exported yet: name -> the issue that adds them.
PENDING = {"deep_confessions": "#39", "tunswitch_weak": "#39"}


@dataclass
class Split:
    utterances: int = 0
    seconds: float = 0.0
    longest: float = 0.0
    over_limit: int = 0
    latin: int = 0
    excluded: int = 0
    excluded_seconds: float = 0.0
    speakers: set[str] = field(default_factory=set[str])


def collect(rows: list[tuple[str, dict[str, Any]]], excluded: frozenset[str]) -> dict[str, Split]:
    """Per ``<source>-<split>``, in first-seen order."""
    splits: dict[str, Split] = defaultdict(Split)
    for name, row in rows:
        s, seconds = splits[name], row["duration"]
        s.utterances += 1
        s.seconds += seconds
        s.longest = max(s.longest, seconds)
        s.over_limit += seconds > WHISPER_MAX_S
        s.latin += LATIN.search(row["text"]) is not None
        if row["speaker"]:
            s.speakers.add(row["speaker"])
        if row["id"] in excluded:
            s.excluded += 1
            s.excluded_seconds += seconds
    return dict(splits)


def composition(splits: dict[str, Split]) -> str:
    lines = [
        "| source | split | utterances | hours | mean s | max s | > 30 s | speakers "
        "| with Latin words | excluded (D-16e) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in splits.items():
        source, split = name.rsplit("-", 1)
        speakers = str(len(s.speakers)) if s.speakers else "not given"
        latin = f"{s.latin} ({100 * s.latin / s.utterances:.0f}%)"
        excluded = str(s.excluded) if split == "train" else "-"
        lines.append(
            f"| {source} | {split} | {s.utterances} | {s.seconds / 3600:.2f} "
            f"| {s.seconds / s.utterances:.1f} | {s.longest:.1f} | {s.over_limit} | {speakers} "
            f"| {latin} | {excluded} |"
        )
    for source, issue in PENDING.items():
        if not any(n.startswith(f"{source}-") for n in splits):
            lines.append(f"| {source} | - | pending {issue} |" + " |" * 7)
    return "\n".join(lines)


def mix(splits: dict[str, Split], sources: list[str]) -> str:
    lines = ["| source | utterances | hours |", "|---|---:|---:|"]
    total_n, total_s = 0, 0.0
    for source in sources:
        s = splits.get(f"{source}-train")
        if s is None:
            lines.append(f"| {source} | no export | |")
            continue
        n, sec = s.utterances - s.excluded, s.seconds - s.excluded_seconds
        total_n, total_s = total_n + n, total_s + sec
        lines.append(f"| {source} | {n} | {sec / 3600:.2f} |")
    lines.append(f"| **total** | **{total_n}** | **{total_s / 3600:.2f}** |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=MIX)
    parser.add_argument("--datasheet", type=Path, default=DATASHEET)
    args = parser.parse_args()

    train_mix = load_config(args.config, TrainMix)
    excluded = load_exclusions(ROOT / train_mix.exclude) if train_mix.exclude else frozenset()
    splits = collect(list(read_export(args.exports)), excluded)
    text = args.datasheet.read_text(encoding="utf-8")
    text = replace_block(text, "datasheet-composition", composition(splits))
    text = replace_block(text, "datasheet-mix", mix(splits, [e.source for e in train_mix.sources]))
    args.datasheet.write_text(text, encoding="utf-8")
    print(f"updated {args.datasheet}")


if __name__ == "__main__":
    main()
