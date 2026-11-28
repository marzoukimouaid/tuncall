"""Phase 5 data (D-30a, D-31a, D-34a): SLURP-TN intents/entities, TSAC sentiment, synthetic calls.

Every text and entity value goes through ``asr.cache.target_text`` (D-10a lexicon, harakat
stripped), so gold text looks like our ASR's output. Entity markup ``[type : value]`` is
parsed before transliteration, each value on its own.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from tuncall.asr.cache import target_text
from tuncall.config import StrictModel
from tuncall.data.sources.common import parquet_files
from tuncall.data.testsets import REPO_ROOT, sha256_file

SLURP_REPO = "Elyadata/SLURP-TN"
TSAC_REPO = "fbougares/tsac"
TSAC_REVISION = "dea79d1a628b8bcb210d3055cb25b90e129a16a4"  # refs/convert/parquet of main 80e5de3
TSAC_VALIDATION = REPO_ROOT / "configs/data/tsac-validation.txt"
TSAC_ARABIZI = REPO_ROOT / "configs/data/tsac-arabizi.tsv"
SENTIMENTS = ("positive", "negative")  # TSAC target 0 = "1", 1 = "-1"
ROLES = ("agent", "customer")

_MARKUP = re.compile(r"\[\s*([^:\]]+?)\s*:\s*([^\]]*?)\s*\]")


@dataclass(frozen=True)
class Utterance:
    id: str  # SLURP-TN file stem, as in the Phase 1 benchmark outputs
    text: str
    intent: str
    entities: tuple[tuple[str, str], ...]  # (type, value)


@dataclass(frozen=True)
class Comment:
    id: str  # TSAC row id within its split
    text: str
    sentiment: str
    arabic: bool  # written in Arabic script originally (D-31a: only these are scored)


def parse_markup(markup: str, lexicon: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """(type, value) pairs from SLURP-TN markup, each value transliterated on its own."""
    return tuple((t, target_text(v, lexicon)) for t, v in _MARKUP.findall(markup))


def slurp_tn(
    split: str, revision: str, lexicon: dict[str, str], root: Path | None = None
) -> list[Utterance]:
    """SLURP-TN ``train``/``validation``/``test`` rows; the audio column is never read."""
    cols = ["audio.path", "tun_transcription", "tun_slu_annoation", "intent"]
    out = []
    for path in parquet_files(SLURP_REPO, revision, f"data/{split}-*.parquet", root):
        for row in pq.read_table(path, columns=cols).to_pylist():
            text = target_text(row["tun_transcription"], lexicon)
            if text:
                ents = parse_markup(row["tun_slu_annoation"], lexicon)
                out.append(Utterance(Path(row["path"]).stem, text, row["intent"], ents))
    return out


def is_arabic(text: str) -> bool:
    """More Arabic letters than Latin ones."""
    arabic = sum("؀" <= c <= "ۿ" for c in text)
    return arabic > sum(c.isascii() and c.isalpha() for c in text)


def tsac_raw(split: str, root: Path | None = None) -> list[tuple[str, str, str]]:
    """(id, comment, sentiment) of TSAC ``train`` or ``test`` as published, empty ones dropped."""
    paths = parquet_files(TSAC_REPO, TSAC_REVISION, f"default/{split}/*.parquet", root)
    rows = [r for p in paths for r in pq.read_table(p).to_pylist()]
    return [
        (r["id"], r["sentence"].strip(), SENTIMENTS[r["target"]])
        for r in rows
        if r["sentence"].strip()
    ]


def tsac_train_pool(root: Path | None = None) -> list[tuple[str, str, str]]:
    """TSAC train without comments whose text is also in TSAC test (971 rows)."""
    test = {text for _, text, _ in tsac_raw("test", root)}
    return [r for r in tsac_raw("train", root) if r[1] not in test]


def tsac_holdout(pool: list[tuple[str, str, str]], fraction: float, seed: int) -> list[str]:
    """Validation ids: ``fraction`` of each sentiment, drawn with ``seed``."""
    rng = random.Random(seed)
    chosen = []
    for s in SENTIMENTS:
        ids = [i for i, _, label in pool if label == s]
        chosen += rng.sample(ids, round(len(ids) * fraction))
    return sorted(chosen, key=int)


def read_ids(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line for line in lines if line and not line.startswith("#")}


def read_arabizi(path: Path = TSAC_ARABIZI) -> dict[str, str]:
    """TSAC train id -> Gemini's Arabic-script conversion (D-31a)."""
    manifest = path.with_suffix(".yaml")
    if not manifest.exists():
        raise ValueError(f"{manifest} missing: run scripts/transliterate_tsac.py")
    frozen = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not frozen["complete"]:
        raise ValueError(f"{path}: conversion unfinished; rerun scripts/transliterate_tsac.py")
    if (actual := sha256_file(path)) != frozen["sha256"]:
        raise ValueError(f"{path} sha256 {actual} != recorded {frozen['sha256']}")
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return dict(line.split("\t", 1) for line in lines if line)


def tsac(split: str, lexicon: dict[str, str], root: Path | None = None) -> list[Comment]:
    """TSAC ``train``/``validation``/``test`` in Arabic script.

    validation is the committed holdout from train. Train and validation drop comments whose
    text, after ``target_text`` and casefolding, is also in test; train also drops those in
    validation (TSAC repeats comments, so exact-copy checks alone leak).
    train swaps each Arabizi comment for its conversion and skips the unconverted ones;
    validation and test keep Arabizi comments with ``arabic=False`` (they are not scored).
    """

    def key(text: str) -> str:
        return target_text(text, lexicon).casefold()

    test = tsac_raw("test", root)
    if split == "test":
        rows = test
    else:
        held = read_ids(TSAC_VALIDATION)
        pool = tsac_train_pool(root)
        seen = {key(t) for _, t, _ in test}
        if split == "train":
            seen |= {key(t) for i, t, _ in pool if i in held}
        rows = [r for r in pool if (r[0] in held) == (split == "validation")]
        rows = [r for r in rows if key(r[1]) not in seen]
    converted = read_arabizi() if split == "train" else {}
    out = []
    for i, text, label in rows:
        arabic = is_arabic(text)
        if split == "train" and not arabic:
            if i not in converted:
                continue
            text = converted[i]
        text = target_text(text, lexicon)
        if text:
            out.append(Comment(i, text, label, arabic))
    return out


class CallLine(StrictModel):
    role: str
    text: str


class Call(StrictModel):
    id: str
    domain: str
    scenario: str
    lines: list[CallLine]


class CallSet(StrictModel):
    """``configs/data/synthetic-calls.yaml``: the frozen calls file and its split."""

    model: str
    prompt_sha256: str
    seed: int
    calls: str
    calls_sha256: str
    splits: dict[str, list[str]]  # train / validation / test -> call ids


def file_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_calls(split: str, manifest: CallSet, lexicon: dict[str, str]) -> list[Call]:
    """Calls of one split, text transliterated; refuses a calls file that is not the frozen one."""
    path = REPO_ROOT / manifest.calls
    if (actual := sha256_file(path)) != manifest.calls_sha256:
        raise ValueError(f"{path} sha256 {actual} != frozen {manifest.calls_sha256}")
    wanted = set(manifest.splits[split])
    calls = [
        Call.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    out = []
    for c in calls:
        if c.id in wanted:
            lines = [CallLine(role=ln.role, text=target_text(ln.text, lexicon)) for ln in c.lines]
            out.append(c.model_copy(update={"lines": lines}))
    return out


def call_turns(
    call: Call, shuffle_seed: int | None = None
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """(speaker, text) lines and the gold role per speaker (D-70d).

    Speakers are named S1, S2 in order of first appearance in the given line order, so with
    ``shuffle_seed`` (lines shuffled per call) "S1 is the agent" stops being a free answer.
    """
    lines = list(call.lines)
    if shuffle_seed is not None:
        random.Random(f"{shuffle_seed}:{call.id}").shuffle(lines)
    names: dict[str, str] = {}
    for ln in lines:
        names.setdefault(ln.role, f"S{len(names) + 1}")
    return [(names[ln.role], ln.text) for ln in lines], {s: r for r, s in names.items()}


def dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


class TsacGen(StrictModel):
    validation_fraction: float
    batch_size: int
    workers: int


class CallsGen(StrictModel):
    count: int
    min_lines: int
    max_lines: int
    max_attempts: int
    splits: dict[str, int]


class DataGen(StrictModel):
    """``configs/phase5/data.yaml``."""

    provider: str
    model: str
    cache_dir: Path
    seed: int
    tsac: TsacGen
    calls: CallsGen
