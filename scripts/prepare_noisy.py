"""Shrink a noisy-label source and push it to a private Hub repo of ours (D-12d, #39).

Run on Colab (B-01c b): the inputs are ~20 GB (Deep Confessions) and ~16 GB (TunSwitch
weakly labelled), too big for the home connection (N-01). For each clip:

1. text: non-speech tags (``<pause>``, ``<laugh>``, ``<eps>``, ...) are stripped as for
   Omnilingual (D-12g); clips left without text are dropped;
2. broken transcripts are dropped (D-12h): looping text (the D-15d rule) or more than
   ``MAX_CHARS_PER_S`` characters per second of audio, both measured on the
   ``normalize_arabic`` text (Deep Confessions is often fully diacritized, which would
   otherwise double the count); the stored text keeps its diacritics;
3. audio: 16 kHz mono, stored as lossless 16-bit FLAC.

Writes ``<work>/<source>/data/train-NNNNN.parquet`` (columns ``key``, ``audio``, ``text``,
``duration``), a dataset card recording the upstream revision and the counts, and the
text export ``outputs/sources/<source>-train.jsonl`` that the dedupe report and the
datasheet read (D-16b). ``--push`` uploads the folder to ``<hf user>/tuncall-<name>``.

    uv run python scripts/prepare_noisy.py deep_confessions --work /content/noisy --push
    uv run python scripts/prepare_noisy.py tunswitch_weak --work /content/noisy --limit 200
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from benchmark_asr import is_loop
from source_stats import OUT

from tuncall.data.sources.common import decode, rows
from tuncall.data.sources.omnilingual import strip_tags
from tuncall.data.testsets import SAMPLE_RATE
from tuncall.eval.normalize import normalize_arabic

MAX_CHARS_PER_S = 25.0
SHARD_BYTES = 300_000_000

DC_REPO = "oddadmix/arabic-audio-collection-tunisian-deep-confessions"
DC_REVISION = "73dcdffe8c7c057dc4d55e97481383d88a51dfd7"  # Hub commit, 2026-11-23
TS_URL = "https://zenodo.org/records/8370566/files/weakly_labeled_tn.zip?download=1"
TS_MD5 = "231674424efccfd4d3f19b2706bafd52"  # Zenodo's checksum for weakly_labeled_tn.zip


@dataclass(frozen=True)
class Upstream:
    repo: str  # our private derived repo, without the user
    license: str
    origin: str  # markdown line for the card


UPSTREAM = {
    "deep_confessions": Upstream(
        "tuncall-deep-confessions",
        "other",
        f"[`{DC_REPO}`](https://huggingface.co/datasets/{DC_REPO}) at commit `{DC_REVISION}`",
    ),
    "tunswitch_weak": Upstream(
        "tuncall-tunswitch-weak",
        "cc-by-4.0",
        "[TunSwitch, Zenodo 8370566](https://zenodo.org/records/8370566) "
        f"`weakly_labeled_tn.zip`, md5 `{TS_MD5}`",
    ),
}

type Clip = tuple[str, bytes, str]  # upstream key, encoded audio, raw text


def clean(text: str, seconds: float) -> tuple[str, str | None]:
    """Cleaned text and, if the clip must be dropped, why."""
    text = strip_tags(text)
    if not text:
        return text, "empty"
    norm = normalize_arabic(text)
    if is_loop(norm):
        return text, "looping"
    if len(norm) / seconds > MAX_CHARS_PER_S:
        return text, "too many chars/s"
    return text, None


def flac(audio: Any) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def deep_confessions(work: Path) -> Iterator[Clip]:
    """Shard by shard, so ``--limit`` downloads only what it reads."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(DC_REPO, repo_type="dataset", revision=DC_REVISION)
    for name in sorted(f for f in files if f.startswith("data/train-")):
        path = hf_hub_download(DC_REPO, name, repo_type="dataset", revision=DC_REVISION)
        for row in rows([Path(path)], ["chunk_id", "audio", "transcript_text"]):
            yield row["chunk_id"], row["audio"]["bytes"], row["transcript_text"]


def tunswitch_weak(work: Path) -> Iterator[Clip]:
    archive = work / "weakly_labeled_tn.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(TS_URL) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=16 << 20)
        tmp.rename(archive)
    md5 = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as f:
        while chunk := f.read(16 << 20):
            md5.update(chunk)
    if md5.hexdigest() != TS_MD5:
        raise ValueError(f"{archive}: md5 {md5.hexdigest()} != {TS_MD5}; delete it and rerun")
    with zipfile.ZipFile(archive) as z:
        with z.open("weakly_audios/train.csv") as f:
            table = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
        for row in table:
            name = Path(row["wav"]).name  # the csv's directory is the authors' machine
            yield Path(name).stem, z.read(f"weakly_audios/{name}"), row["wrd"]


READERS = {"deep_confessions": deep_confessions, "tunswitch_weak": tunswitch_weak}


def prepare(source: str, clips: Iterable[Clip], out: Path, export: Path) -> Counter[str]:
    """Clean, re-encode and shard ``clips``; return kept/dropped counts and seconds."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    shard: list[dict[str, Any]] = []
    size, kept_seconds = 0, 0.0

    def flush() -> None:
        nonlocal shard, size
        if shard:
            pq.write_table(
                pa.Table.from_pylist(shard), data / f"train-{stats['shards']:05d}.parquet"
            )
            stats["shards"] += 1
        shard, size = [], 0

    export.parent.mkdir(parents=True, exist_ok=True)
    with export.open("w", encoding="utf-8") as ex:
        for key, encoded, raw in clips:
            audio = decode(encoded)
            seconds = len(audio) / SAMPLE_RATE
            text, dropped = clean(raw, seconds)
            if dropped:
                stats[f"dropped: {dropped}"] += 1
                continue
            stats["kept"] += 1
            kept_seconds += seconds
            encoded = flac(audio)
            shard.append(
                {
                    "key": key,
                    "audio": {"bytes": encoded, "path": f"{key}.flac"},
                    "text": text,
                    "duration": seconds,
                }
            )
            size += len(encoded)
            line = {
                "id": f"{source}/train/{key}",
                "text": text,
                "duration": seconds,
                "speaker": None,
            }
            ex.write(json.dumps(line, ensure_ascii=False) + "\n")
            if size >= SHARD_BYTES:
                flush()
        flush()
    stats["kept seconds"] = round(kept_seconds)
    return stats


def card(source: str, stats: Counter[str]) -> str:
    up = UPSTREAM[source]
    counts = "\n".join(f"| {k} | {v} |" for k, v in sorted(stats.items()) if k != "shards")
    return (
        f"---\nlicense: {up.license}\nlanguage:\n- aeb\npretty_name: tuncall {source}\n---\n\n"
        f"# tuncall: {source} (derived, private)\n\n"
        f"Derived from {up.origin} by `scripts/prepare_noisy.py` of the tuncall project "
        "(D-12d, D-12h): non-speech tags stripped, looping text and texts over "
        f"{MAX_CHARS_PER_S:g} characters per second dropped, audio 16 kHz mono FLAC. "
        "Labels are machine-made (noisy); used only for with/without ablations. "
        "Upstream rights apply; not for redistribution.\n\n"
        f"| count | value |\n|---|---:|\n{counts}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(READERS))
    parser.add_argument("--work", type=Path, required=True, help="scratch dir with ~60 GB free")
    parser.add_argument("--exports", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, help="first N clips only (pre-flight)")
    parser.add_argument("--push", action="store_true", help="upload to a private Hub repo")
    args = parser.parse_args()
    if args.push and args.limit:
        parser.error("--push uploads the whole source; drop --limit")

    out = args.work / args.source
    if out.exists():
        shutil.rmtree(out)  # a rerun starts clean; finished shards aren't reused
    args.work.mkdir(parents=True, exist_ok=True)
    clips = islice(READERS[args.source](args.work), args.limit)
    stats = prepare(args.source, clips, out, args.exports / f"{args.source}-train.jsonl")
    (out / "README.md").write_text(card(args.source, stats), encoding="utf-8")
    hours = stats["kept seconds"] / 3600
    print(f"{args.source}: {dict(stats)} ({hours:.1f} h kept)")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        repo = f"{api.whoami()['name']}/{UPSTREAM[args.source].repo}"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=out, repo_type="dataset")
        sha = api.dataset_info(repo).sha
        print(f"pushed {repo} at commit {sha}")


if __name__ == "__main__":
    main()
