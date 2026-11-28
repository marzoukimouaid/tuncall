"""Speaker diarization with pyannote ``speaker-diarization-community-1``, off the shelf (D-23).

The model is gated on the Hub: its terms are accepted once and ``HF_TOKEN`` is read from
the environment. Speakers are renamed ``S1``, ``S2``, ... in order of first speech.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tuncall.data.testsets import SAMPLE_RATE

MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Diarization:
    regions: list[Span]  # may overlap: what DER is scored on
    turns: list[Span]  # no overlap, same-speaker neighbours merged: what gets transcribed


def load(revision: str, device: str = "cuda") -> Any:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL, revision=revision, token=os.environ.get("HF_TOKEN"))
    if pipeline is None:  # pyannote returns None instead of raising on a gated/missing model
        raise RuntimeError(f"could not load {MODEL}@{revision}: accept its terms and set HF_TOKEN")
    return pipeline.to(torch.device(device))


def spans(annotation: Any) -> list[tuple[float, float, str]]:
    return [(s.start, s.end, label) for s, _, label in annotation.itertracks(yield_label=True)]


def relabel(
    regions: list[tuple[float, float, str]], turns: list[tuple[float, float, str]]
) -> Diarization:
    """Rename speakers by first speech and merge consecutive turns of one speaker."""
    names: dict[str, str] = {}
    for _, _, label in sorted(regions):
        names.setdefault(label, f"S{len(names) + 1}")
    merged: list[Span] = []
    for start, end, label in sorted(turns):
        if merged and merged[-1].speaker == names[label]:
            merged[-1] = Span(merged[-1].start, end, merged[-1].speaker)
        else:
            merged.append(Span(start, end, names[label]))
    return Diarization([Span(s, e, names[label]) for s, e, label in sorted(regions)], merged)


def diarize(pipeline: Any, audio: np.ndarray, num_speakers: int | None = None) -> Diarization:
    """16 kHz mono float32 in. ``num_speakers=None`` lets pyannote estimate the count."""
    out = pipeline(
        {"waveform": torch.from_numpy(audio)[None], "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
    )
    return relabel(spans(out.speaker_diarization), spans(out.exclusive_speaker_diarization))
