"""How much of each test set appears verbatim in FarukSTT's training transcripts (D-12b, D-15e).

A reference counts as "seen" if, after the D-10a lexicon and ``normalize_arabic``, it has at
least 4 words and equals a transcript of FarukSTT's train split treated the same way (D-16c). Only
the ``sentence`` column is read (parquet range requests), so no audio is downloaded.

    uv run python scripts/farukstt_overlap.py
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from tuncall.config import load_config
from tuncall.data.conversations import ConversationSet, load_conversations
from tuncall.data.lexicon import load_lexicon, transliterate
from tuncall.data.testsets import Manifest, fetch, iter_cuts, sha256_file
from tuncall.eval.normalize import normalize_arabic
from tuncall.eval.report import replace_block
from tuncall.tracking import Tracker

REPO = "FARUKxAUTO/tunisian-asr-cleaned"
REVISION = "70621ea0b391c96f5422a754ccb975a640d35066"
MIN_WORDS = 4
ROOT = Path(__file__).resolve().parents[1]
TEST_SETS = ("test-general", "test-domain", "conversations-test")  # last: Phase 4 (D-13b)
LEXICON = ROOT / "configs" / "data" / "lexicon.tsv"
README = ROOT / "README.md"


def overlap(refs: dict[str, str], train: set[str]) -> tuple[list[str], list[str]]:
    """(eligible ids, seen ids); ``train`` must already be normalized."""
    norm = {i: normalize_arabic(t) for i, t in refs.items()}
    eligible = [i for i, t in norm.items() if len(t.split()) >= MIN_WORDS]
    return eligible, [i for i in eligible if norm[i] in train]


def train_transcripts(lexicon: dict[str, str]) -> set[str]:
    fs = HfFileSystem()
    paths = sorted(
        p
        for p in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        if p.startswith("data/train-") and p.endswith(".parquet")
    )
    out: set[str] = set()
    for p in paths:
        # Small blocks, no pre-buffering: fetch the text column chunks, not the audio around them.
        with fs.open(f"datasets/{REPO}@{REVISION}/{p}", block_size=2**20) as f:
            table = pq.read_table(f, columns=["sentence"], pre_buffer=False)
        column = table.column("sentence")
        out.update(normalize_arabic(transliterate(s, lexicon)) for s in column.to_pylist() if s)
    return out


def conversation_refs(name: str, lexicon: dict[str, str]) -> dict[str, str]:
    """One reference per TEDxTN clip placed in the simulated conversations (already
    transliterated by ``load_conversations``)."""
    del lexicon
    manifest = load_config(ROOT / "configs" / "data" / f"{name}.yaml", ConversationSet)
    convs = load_conversations(manifest)
    return {f"{c.id}/{i}": clip.text for c in convs for i, clip in enumerate(c.clips)}


def test_set_refs(name: str, lexicon: dict[str, str]) -> dict[str, str]:
    if name.startswith("conversations-"):
        return conversation_refs(name, lexicon)
    manifest = load_config(ROOT / "configs" / "data" / f"{name}.yaml", Manifest)
    files = fetch(manifest)
    for f in manifest.files:
        if sha256_file(files[f.path]) != f.sha256:
            raise ValueError(f"{f.path} does not match the frozen sha256")
    cuts = iter_cuts(manifest, files)
    return {s.id: transliterate(s.text, lexicon) for segments, _ in cuts for s in segments}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase1"))
    args = parser.parse_args()

    lexicon = load_lexicon(LEXICON)
    train = train_transcripts(lexicon)
    sentences = []
    args.out.mkdir(parents=True, exist_ok=True)
    for name in TEST_SETS:
        refs = test_set_refs(name, lexicon)
        eligible, seen = overlap(refs, train)
        pct = 100 * len(seen) / len(eligible)
        sentences.append(
            f"{pct:.1f}% of {name} utterances with ≥ {MIN_WORDS} words "
            f"({len(seen)}/{len(eligible)} eligible, {len(refs)} total) appear verbatim in "
            f"FarukSTT's training data"
        )
        ids_path = args.out / f"farukstt-overlap-{name}.txt"
        ids_path.write_text("".join(f"{i}\n" for i in seen), encoding="utf-8")
        with Tracker(
            "phase1-baselines",
            config={"repo": REPO, "revision": REVISION, "min_words": MIN_WORDS, "test_set": name},
            dataset=f"{REPO}@{REVISION}",
            buffer_path=args.out / "tracking-buffer.jsonl",
            run_name=f"farukstt-overlap/{name}",
        ) as tracker:
            tracker.log_metrics(
                {"seen_pct": pct, "seen": len(seen), "eligible": len(eligible), "total": len(refs)},
                step=0,
            )
            tracker.log_artifact(ids_path)

    sentence = (
        "; ".join(sentences) + f" (`{REPO}` train split, D-12b, lexicon applied to both sides; "
        "`scripts/farukstt_overlap.py`)."
    )
    print(sentence)
    text = replace_block(README.read_text(encoding="utf-8"), "farukstt-overlap", sentence)
    README.write_text(text, encoding="utf-8")
    print(f"updated {README}")


if __name__ == "__main__":
    main()
