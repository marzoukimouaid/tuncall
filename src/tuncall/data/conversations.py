"""Simulated two-speaker conversations stitched from TEDxTN clips (D-13b, ADR 0008).

Each TEDxTN talk has one speaker, so a talk is a speaker. A conversation alternates
turns of two talks; a turn is 1-3 consecutive clips of its talk. Turn changes leave a
0.2-1.0 s pause, or (10% of them) overlap the previous turn by up to 0.5 s. Each clip
is used at most once, so conversations share no audio (owner's choice, 2026-11-26).

What is frozen is a committed turns TSV (conversation, clip, start, speaker) plus a YAML
manifest pinning the TEDxTN revision, the lexicon, the TSV's sha256 and the sha256 of the
rebuilt clean audio. Loading rebuilds the audio from the pinned clips and checks all three.
The telephone version is ``tuncall.data.telephone`` applied to the clean audio (D-25a).
Results on these sets are always labelled "simulated".
"""

import csv
import hashlib
import random
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tuncall.config import StrictModel
from tuncall.data.lexicon import load_lexicon, transliterate
from tuncall.data.sources import tedxtn
from tuncall.data.testsets import REPO_ROOT, SAMPLE_RATE, LexiconRef, sha256_file

TARGET_S = 120.0  # a conversation ends once it is this long
MIN_LEFT_S = 60.0  # stop when the second-richest talk has less audio than this left
MAX_CLIPS = 3
PAUSE_S = (0.2, 1.0)
OVERLAP_P = 0.1
OVERLAP_MAX_S = 0.5
CLIP_KEY = re.compile(r"^(?P<talk>.+)_(?P<start>\d+)_(?P<end>\d+)$")  # times in ms


class ConversationSet(StrictModel):
    name: str
    repo: str
    revision: str
    split: str
    seed: int
    turns: str  # TSV path relative to the repo root
    turns_sha256: str
    audio_sha256: str  # clean audio of every conversation, in order
    lexicon: LexiconRef
    conversations: int
    hours: float


@dataclass(frozen=True)
class Placed:
    conversation: str
    clip: str  # TEDxTN file name without extension
    start: float  # seconds into the conversation
    speaker: str  # S1 / S2


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Conversation:
    id: str
    audio: np.ndarray  # clean, float32 mono at SAMPLE_RATE
    turns: list[Turn]  # what the oracle row transcribes
    clips: list[Turn]  # one per placed clip: the speech regions DER is scored against


def parse(clip: str) -> tuple[str, int]:
    """(talk, start ms) of a TEDxTN clip name."""
    m = CLIP_KEY.match(clip)
    if m is None:
        raise ValueError(f"not a TEDxTN clip name: {clip}")
    return m["talk"], int(m["start"])


def plan(durations: dict[str, float], seed: int, prefix: str) -> list[Placed]:
    """Place clips (name -> seconds) into conversations. Pure: same input, same output.

    Each conversation pairs the two talks with the most audio left, so talks are used
    evenly. Clips are taken in talk order, so the speech in a turn runs on naturally.
    """
    queues: dict[str, deque[str]] = {}
    for clip in sorted(durations, key=parse):
        queues.setdefault(parse(clip)[0], deque()).append(clip)
    rng = random.Random(seed)
    placed: list[Placed] = []

    def left(t: str) -> float:
        return sum(durations[c] for c in queues[t])

    n = 0
    while len(queues) >= 2:
        a, b = sorted(queues, key=lambda t: (-left(t), t))[:2]
        if left(b) < MIN_LEFT_S:
            break
        cid = f"{prefix}-{n:03d}"
        n += 1
        who, other = (a, b) if rng.random() < 0.5 else (b, a)
        label = {who: "S1", other: "S2"}
        end = 0.0
        turn_start = 0.0
        own_end = {a: 0.0, b: 0.0}  # a speaker never overlaps their own last turn
        first = True
        while end < TARGET_S and queues[who]:
            if not first:
                if rng.random() < OVERLAP_P:
                    start = end - rng.uniform(0.0, min(OVERLAP_MAX_S, end - turn_start))
                else:
                    start = end + rng.uniform(*PAUSE_S)
            else:
                start = 0.0
            turn_start = pos = max(start, own_end[who])
            for i in range(rng.randint(1, MAX_CLIPS)):
                if not queues[who]:
                    break
                if i:
                    pos += rng.uniform(*PAUSE_S)
                clip = queues[who].popleft()
                placed.append(Placed(cid, clip, round(pos, 3), label[who]))
                pos += durations[clip]
            end = own_end[who] = max(end, pos)
            first = False
            who, other = other, who
        for t in (a, b):
            if not queues[t]:
                del queues[t]
    return placed


def write_turns(path: Path, placed: Iterable[Placed]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["conversation", "clip", "start", "speaker"])
        for p in placed:
            w.writerow([p.conversation, p.clip, f"{p.start:.3f}", p.speaker])


def read_turns(path: Path) -> list[Placed]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Placed(r["conversation"], r["clip"], float(r["start"]), r["speaker"])
            for r in csv.DictReader(f, delimiter="\t")
        ]


def assemble(
    placed: list[Placed], audio: dict[str, np.ndarray], text: dict[str, str]
) -> list[Conversation]:
    """Mix the placed clips (overlaps add up) and group them into reference turns.

    Consecutive rows of one speaker form one turn; the TSV is written in turn order.
    """
    by_conv: dict[str, list[Placed]] = {}
    for p in placed:
        by_conv.setdefault(p.conversation, []).append(p)
    out: list[Conversation] = []
    for cid, rows in by_conv.items():
        spans = [(p, round(p.start * SAMPLE_RATE)) for p in rows]
        mix = np.zeros(max(s + len(audio[p.clip]) for p, s in spans), dtype=np.float32)
        turns: list[Turn] = []
        placed_clips: list[Turn] = []
        for p, s in spans:
            clip = audio[p.clip]
            mix[s : s + len(clip)] += clip
            end = (s + len(clip)) / SAMPLE_RATE
            placed_clips.append(Turn(p.speaker, s / SAMPLE_RATE, end, text[p.clip]))
            if turns and turns[-1].speaker == p.speaker:
                last = turns[-1]
                turns[-1] = Turn(last.speaker, last.start, end, f"{last.text} {text[p.clip]}")
            else:
                turns.append(Turn(p.speaker, s / SAMPLE_RATE, end, text[p.clip]))
        out.append(Conversation(cid, mix, turns, placed_clips))
    return out


def audio_digest(conversations: Iterable[Conversation]) -> str:
    h = hashlib.sha256()
    for c in conversations:
        h.update(c.id.encode())
        h.update(c.audio.tobytes())
    return h.hexdigest()


def split_audio(
    split: str, revision: str, lexicon: dict[str, str], root: Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Clip name -> audio and -> reference text (lexicon applied) for a TEDxTN split."""
    audio: dict[str, np.ndarray] = {}
    text: dict[str, str] = {}
    for r in tedxtn.records(split, revision, root):
        key = r.id.split("/", 2)[2]
        audio[key], text[key] = r.audio, transliterate(r.text, lexicon)
    return audio, text


def load_conversations(manifest: ConversationSet, root: Path | None = None) -> list[Conversation]:
    """Rebuild the frozen set and verify the TSV, lexicon and audio hashes."""
    turns_path = REPO_ROOT / manifest.turns
    checks = [
        (turns_path, manifest.turns_sha256),
        (REPO_ROOT / manifest.lexicon.path, manifest.lexicon.sha256),
    ]
    for path, frozen in checks:
        actual = sha256_file(path)
        if actual != frozen:
            raise ValueError(f"{manifest.name}: {path} sha256 {actual} != frozen {frozen}")
    lexicon = load_lexicon(REPO_ROOT / manifest.lexicon.path)
    audio, text = split_audio(manifest.split, manifest.revision, lexicon, root)
    conversations = assemble(read_turns(turns_path), audio, text)
    digest = audio_digest(conversations)
    if len(conversations) != manifest.conversations or digest != manifest.audio_sha256:
        raise ValueError(
            f"{manifest.name}: {len(conversations)} conversations / audio {digest} "
            f"!= frozen {manifest.conversations} / {manifest.audio_sha256}"
        )
    return conversations
