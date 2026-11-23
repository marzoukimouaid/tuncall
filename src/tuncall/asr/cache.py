"""Local audio cache for ASR training (D-21c).

Training needs clips in a fresh random order every epoch, which the adapters' front-to-back
parquet reads can't give. ``build`` decodes a stream of records once into ``audio.i16`` (all
clips back to back, 16 kHz int16) and ``index.jsonl`` (``{id, offset, length, text}``);
``Clips`` then reads any clip by position through a memory map.

Training text is the D-10a lexicon transliteration with harakat stripped (D-10c); clips over
Whisper's 30 s window or left with no text are dropped.
"""

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tuncall.data.lexicon import transliterate
from tuncall.data.sources import Record

SAMPLE_RATE = 16_000
MAX_SECONDS = 30.0
# Harakat, tanween, shadda, sukun (U+064B-U+065F) and superscript alef, as normalize_arabic.
_HARAKAT = dict.fromkeys([*range(0x064B, 0x0660), 0x0670])


def target_text(text: str, lexicon: dict[str, str]) -> str:
    """The training target for a transcript (D-10a, D-10c)."""
    return " ".join(transliterate(text, lexicon).translate(_HARAKAT).split())


def build(records: Iterable[Record], out: Path, lexicon: dict[str, str]) -> dict[str, int]:
    """Write the cache to ``out`` (via ``out.tmp`` + rename) and return what was kept/dropped."""
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    counts = {"kept": 0, "too_long": 0, "empty": 0}
    offset = 0
    with (tmp / "audio.i16").open("wb") as audio, (tmp / "index.jsonl").open("w") as index:
        for r in records:
            text = target_text(r.text, lexicon)
            if r.duration > MAX_SECONDS:
                counts["too_long"] += 1
                continue
            if not text:
                counts["empty"] += 1
                continue
            pcm = (np.clip(r.audio, -1, 1) * 32767).astype("<i2")
            audio.write(pcm.tobytes())
            row = {"id": r.id, "offset": offset, "length": len(pcm), "text": text}
            index.write(json.dumps(row, ensure_ascii=False) + "\n")
            offset += len(pcm)
            counts["kept"] += 1
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    return counts


class Clips:
    """Random access to a built cache: ``clips[i] -> {"id", "audio" (float32), "text"}``."""

    def __init__(self, path: Path) -> None:
        with (path / "index.jsonl").open(encoding="utf-8") as f:
            self.index = [json.loads(line) for line in f]
        self._audio = np.memmap(path / "audio.i16", dtype="<i2", mode="r")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, object]:
        row = self.index[i]
        pcm = self._audio[row["offset"] : row["offset"] + row["length"]]
        return {"id": row["id"], "audio": pcm.astype(np.float32) / 32767, "text": row["text"]}
