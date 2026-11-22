"""Latin-script words left in the source texts, before and after the lexicon (D-10a, #34).

Reads the text export of ``source_stats.py`` (all splits: the lexicon is applied to
train and test alike). Prints Latin tokens per source and how many the lexicon leaves
(target 0), and writes the words still missing from the lexicon, most frequent first,
to ``outputs/lexicon/candidates.tsv`` (word, count, sources, one example) for review.

    uv run python scripts/latin_words.py [--exports outputs/sources]
"""

import argparse
import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from source_stats import OUT, ROOT, read_export

from tuncall.data.lexicon import latin_words, load_lexicon

LEXICON = ROOT / "configs" / "data" / "lexicon.tsv"
CANDIDATES = ROOT / "outputs" / "lexicon" / "candidates.tsv"

type Candidate = tuple[str, int, str, str]  # word, count, sources, example text


def report(
    texts: Iterable[tuple[str, str]], lexicon: dict[str, str]
) -> tuple[str, list[Candidate]]:
    """Per-source table of Latin tokens and the words the lexicon doesn't cover yet."""
    tokens: Counter[str] = Counter()
    left: Counter[str] = Counter()
    unique_left: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    example: dict[str, str] = {}
    for source, text in texts:
        tokens[source] += 0  # keep sources without Latin in the table
        for w in latin_words(text):
            tokens[source] += 1
            if w in lexicon:
                continue
            left[source] += 1
            unique_left[source].add(w)
            counts[w] += 1
            sources[w].add(source)
            example.setdefault(w, text)

    lines = [
        "| source | latin tokens | left after lexicon | unique left |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {s} | {tokens[s]} | {left[s]} | {len(unique_left[s])} |" for s in tokens]
    lines.append(f"| total | {tokens.total()} | {left.total()} | {len(counts)} |")
    candidates = [
        (w, n, ",".join(sorted(sources[w])), example[w])
        for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return "\n".join(lines), candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=OUT)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--out", type=Path, default=CANDIDATES)
    args = parser.parse_args()

    lexicon = load_lexicon(args.lexicon) if args.lexicon.exists() else {}
    texts = ((name.split("-")[0], row["text"]) for name, row in read_export(args.exports))
    table, candidates = report(texts, lexicon)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(candidates)
    print(f"{len(candidates)} words missing from the lexicon -> {args.out}")


if __name__ == "__main__":
    main()
