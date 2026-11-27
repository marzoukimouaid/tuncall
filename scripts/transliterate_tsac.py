"""Convert TSAC's Arabizi train comments to Arabic script with Gemini (D-31a, D-32c).

First writes configs/data/tsac-validation.txt (10% of each sentiment of TSAC train, after
dropping comments that are also in test) if it does not exist. Then sends the Arabizi
comments of the remaining train rows to Gemini, 25 per request; a reply with the wrong count
is redone one comment at a time. Replies are cached, so a run stopped by the free tier's
daily quota resumes where it left off. Writes configs/data/tsac-arabizi.tsv (id, text) with
everything converted so far, and a spot-check sample to outputs/phase5/.

    uv run python scripts/transliterate_tsac.py [--limit N]
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from tuncall.config import load_config
from tuncall.data.testsets import REPO_ROOT, sha256_file
from tuncall.nlp.data import (
    TSAC_ARABIZI,
    TSAC_VALIDATION,
    DataGen,
    dumps,
    file_sha256,
    is_arabic,
    read_ids,
    tsac_holdout,
    tsac_train_pool,
)
from tuncall.nlp.llm import LLM, QuotaExhausted, parse_json

TSAC_SUBSET = REPO_ROOT / "configs/data/tsac-arabizi-ids.txt"

PROMPT = """You convert Tunisian Darija social-media comments from Latin letters (Arabizi, \
often mixed with French or English) to Arabic script.
Rules:
- Keep every word and its meaning. Do not translate, correct, summarize or add words.
- Write French and English words in Arabic letters as a Tunisian pronounces them \
(bravo -> برافو, mail -> مايل, merci -> مرسي).
- Arabizi digits: 2=ء, 3=ع, 5=خ, 7=ح, 8=غ, 9=ق.
- Leave out emojis and links.
You get a JSON list of N comments. Reply with only a JSON object {"texts": [...]} holding \
exactly N strings, the conversions in the same order."""


def convert(llm: LLM, texts: list[str]) -> list[str] | None:
    """Conversions of ``texts``, or ``None`` if the reply is not N strings."""
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": dumps(texts)},
    ]
    reply = parse_json(llm.chat(messages, temperature=0.0))
    out = reply.get("texts") if isinstance(reply, dict) else None
    if not isinstance(out, list) or len(out) != len(texts):
        return None
    if not all(isinstance(t, str) for t in out):
        return None
    return [" ".join(t.split()) for t in out]


def from_files(folder: Path, todo: list[tuple[str, str]]) -> tuple[dict[str, str], bool]:
    """Write the work list to ``folder/arabizi-todo.jsonl``; read ``arabizi-*.out.jsonl``.

    A reply with empty text marks a comment with nothing to convert (emojis only). Complete
    when every comment of ``todo`` has a reply.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "arabizi-todo.jsonl").write_text(
        "".join(dumps({"id": i, "text": t}) + "\n" for i, t in todo), encoding="utf-8"
    )
    wanted, done, answered = dict(todo), {}, set()
    for path in sorted(folder.glob("arabizi-*.out.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in wanted:
                answered.add(row["id"])
                if text := " ".join(row["text"].split()):
                    done[row["id"]] = text
    return done, answered == set(wanted)


def from_api(cfg: DataGen, todo: list[tuple[str, str]], full: bool) -> tuple[dict[str, str], bool]:
    llm = LLM(cfg.provider, cfg.model, REPO_ROOT / cfg.cache_dir)
    done: dict[str, str] = {}
    size, complete = cfg.tsac.batch_size, False

    def work(batch: list[tuple[str, str]]) -> list[str]:
        out = convert(llm, [t for _, t in batch])
        if out is None:  # D-32c: redo this batch one comment at a time
            out = [(convert(llm, [t]) or [""])[0] for _, t in batch]
        return out

    batches = [todo[i : i + size] for i in range(0, len(todo), size)]
    try:
        # Requests are slow (Gemini thinks first), so a few run at once; order is kept.
        with ThreadPoolExecutor(cfg.tsac.workers) as pool:
            for batch, out in zip(batches, pool.map(work, batches), strict=True):
                done |= {i: t for (i, _), t in zip(batch, out, strict=True) if t}
                print(f"\r{len(done)}/{len(todo)} converted", end="", file=sys.stderr)
        complete = full
    except QuotaExhausted as err:
        print(f"\nDaily quota used up ({err}). Rerun tomorrow to resume.", file=sys.stderr)
    print(file=sys.stderr)
    return done, complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase5/data.yaml")
    parser.add_argument("--limit", type=int, help="convert only the first N comments")
    args = parser.parse_args()
    cfg = load_config(args.config, DataGen)

    pool = tsac_train_pool()
    if not TSAC_VALIDATION.exists():
        ids = tsac_holdout(pool, cfg.tsac.validation_fraction, cfg.seed)
        header = "# TSAC train comments held out as validation, never trained on (D-31a).\n"
        header += "# Generated by scripts/transliterate_tsac.py; do not edit.\n"
        TSAC_VALIDATION.write_text(header + "".join(f"{i}\n" for i in ids), encoding="utf-8")
        print(f"{len(ids)} validation ids -> {TSAC_VALIDATION}")
    held = read_ids(TSAC_VALIDATION)
    todo = [(i, t) for i, t, _ in pool if i not in held and not is_arabic(t)]
    todo = todo[: args.limit]

    if cfg.provider == "claude":  # D-31c: replies written in a Claude session, read from files
        if TSAC_SUBSET.exists():  # D-31c: only this owner-sized subset is converted
            subset = read_ids(TSAC_SUBSET)
            todo = [(i, t) for i, t in todo if i in subset]
        done, complete = from_files(REPO_ROOT / cfg.cache_dir, todo)
        complete = complete and args.limit is None
    else:
        done, complete = from_api(cfg, todo, args.limit is None)

    rows = sorted(done.items(), key=lambda kv: int(kv[0]))
    TSAC_ARABIZI.write_text(
        "id\ttext\n" + "".join(f"{i}\t{t}\n" for i, t in rows), encoding="utf-8"
    )
    manifest = {"model": cfg.model, "prompt_sha256": file_sha256(PROMPT), "comments": len(rows)}
    manifest |= {"failed": len(todo) - len(rows), "complete": complete}
    manifest["sha256"] = sha256_file(TSAC_ARABIZI)
    TSAC_ARABIZI.with_suffix(".yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    print(f"{len(rows)}/{len(todo)} Arabizi comments converted -> {TSAC_ARABIZI}")

    original = dict(todo)
    sample = random.Random(cfg.seed).sample(rows, min(50, len(rows)))
    spot = REPO_ROOT / "outputs/phase5/spot-check-arabizi.md"
    spot.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TSAC Arabizi conversions: spot-check sample\n"]
    lines += [f"{n}. `{original[i]}`\n   → {t}\n" for n, (i, t) in enumerate(sample, 1)]
    spot.write_text("\n".join(lines), encoding="utf-8")
    print(f"spot-check sample -> {spot}")


if __name__ == "__main__":
    main()
