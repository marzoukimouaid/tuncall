"""Batch pipeline: an audio file in, a ``CallRecord`` out (D-24, D-23a, ADR 0008).

Diarize first, then transcribe each turn with ``asr.transcribe`` (turns over 30 s use
long-form decoding), then, if the config names NLP models, tag each turn with its intent,
entities and sentiment and each speaker with a role (D-55a). Batch only; the API and job
queue are Phase 6.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from tuncall.asr import ModelRef
from tuncall.asr.cache import target_text
from tuncall.asr.transcribe import load, transcribe
from tuncall.config import StrictModel
from tuncall.data.lexicon import load_lexicon, transliterate
from tuncall.data.testsets import REPO_ROOT, SAMPLE_RATE, Segment, decode_segment
from tuncall.diarize import MODEL, Diarization, Span, diarize
from tuncall.diarize import load as load_diarizer
from tuncall.eval.normalize import normalize_arabic
from tuncall.nlp.predict import NlpModel, load_predictors
from tuncall.record import CallRecord, Entity, Meta
from tuncall.record import Segment as RecordSegment
from tuncall.tracking import git_commit


class NlpSettings(StrictModel):
    tasks: dict[str, NlpModel]  # the shipped model per task (D-55a)
    qwen_config: Path | None = None  # the training config of the Qwen adapters, if any ships


class PipelineConfig(StrictModel):
    asr: ModelRef
    diarization_revision: str  # Hub commit of pyannote/speaker-diarization-community-1
    num_speakers: int | None = 2  # D-23: a call is two people; None = estimate
    lexicon: Path = Path("configs/data/lexicon.tsv")  # for text_normalized (D-16c)
    batch_size: int = 4
    language: str = "ar"
    task: str = "transcribe"
    num_beams: int = 5
    max_new_tokens: int = 440
    nlp: NlpSettings | None = None  # Phase 5; None leaves the NLP fields empty


@dataclass
class Models:
    """Everything loaded once and reused across files."""

    cfg: PipelineConfig
    diarizer: Any
    whisper: Any
    processor: Any
    lexicon: dict[str, str]
    nlp: Mapping[str, Callable[[list[Any]], list[Any]]] = field(default_factory=dict)  # by task

    @property
    def asr_version(self) -> str:
        ref = self.cfg.asr
        return f"{ref.id}@{ref.revision}" + (f"+{ref.adapter}" if ref.adapter else "")

    @property
    def diarization_version(self) -> str:
        return f"{MODEL}@{self.cfg.diarization_revision}"


def load_models(cfg: PipelineConfig) -> Models:
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    adapter = REPO_ROOT / cfg.asr.adapter if cfg.asr.adapter else None
    whisper, processor = load(
        cfg.asr.id, cfg.asr.revision, device, torch.float16 if cuda else torch.float32, adapter
    )
    return Models(
        cfg,
        load_diarizer(cfg.diarization_revision, device),
        whisper,
        processor,
        load_lexicon(REPO_ROOT / cfg.lexicon),
        load_predictors(cfg.nlp.tasks, cfg.nlp.qwen_config, device) if cfg.nlp else {},
    )


def transcribe_turns(models: Models, audio: np.ndarray, turns: list[Span]) -> list[str]:
    cfg = models.cfg
    return transcribe(
        models.whisper,
        models.processor,
        [audio[round(t.start * SAMPLE_RATE) : round(t.end * SAMPLE_RATE)] for t in turns],
        batch_size=cfg.batch_size,
        language=cfg.language,
        task=cfg.task,
        num_beams=cfg.num_beams,
        max_new_tokens=cfg.max_new_tokens,
    )


def build_record(
    models: Models,
    turns: list[Span],
    texts: list[str],
    *,
    call_id: str,
    source: str,
    duration_s: float,
    channels: int,
    processing_ms: dict[str, float],
) -> CallRecord:
    segments = [
        RecordSegment(
            start=round(t.start, 3),
            end=round(t.end, 3),
            speaker=t.speaker,
            text_raw=text,
            text_normalized=normalize_arabic(transliterate(text, models.lexicon)),
            asr_model_version=models.asr_version,
        )
        for t, text in zip(turns, texts, strict=True)
    ]
    return CallRecord(
        call_id=call_id,
        source=source,
        duration_s=round(duration_s, 3),
        sample_rate=SAMPLE_RATE,
        channels=channels,
        segments=segments,
        meta=Meta(
            pipeline_version=f"{version('tuncall')}+{git_commit()}",
            diarization_model_version=models.diarization_version,
            processing_ms_by_stage=processing_ms,
        ),
    )


def annotate(models: Models, record: CallRecord) -> CallRecord:
    """Fill the Phase 5 fields from the shipped NLP models (D-55a).

    Each turn's text goes in the training convention (``target_text``, D-34a); the role model
    sees the whole call, and a speaker it gives no role keeps ``unknown``.
    """
    segs, nlp = record.segments, models.nlp
    texts = [target_text(s.text_raw, models.lexicon) for s in segs]
    fields: list[dict[str, Any]] = [{} for _ in segs]
    for task in ("intent", "sentiment"):
        if task in nlp:
            for f, label in zip(fields, nlp[task](texts), strict=True):
                f[task] = label
    if "entities" in nlp:
        for f, pairs in zip(fields, nlp["entities"](texts), strict=True):
            f["entities"] = [Entity(type=t, value=v) for t, v in pairs]
    if "role" in nlp and segs:
        roles = nlp["role"]([[(s.speaker, t) for s, t in zip(segs, texts, strict=True)]])[0]
        for f, s in zip(fields, segs, strict=True):
            f["role"] = roles.get(s.speaker) or "unknown"
    assert models.cfg.nlp is not None
    versions = {t: models.cfg.nlp.tasks[t].version for t in nlp}
    return record.model_copy(
        update={
            "segments": [s.model_copy(update=f) for s, f in zip(segs, fields, strict=True)],
            "meta": record.meta.model_copy(update={"nlp_model_versions": versions}),
        }
    )


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - t0) * 1000, 1)


def process(
    models: Models,
    audio: np.ndarray,
    *,
    call_id: str,
    source: str,
    num_speakers: int | None,
    channels: int = 1,
) -> tuple[CallRecord, Diarization]:
    """16 kHz mono float32 in. Returns the record and the raw diarization (for scoring).
    ``num_speakers=None`` lets pyannote estimate the count."""
    diarization, diarize_ms = timed(lambda: diarize(models.diarizer, audio, num_speakers))
    texts, asr_ms = timed(lambda: transcribe_turns(models, audio, diarization.turns))
    record = build_record(
        models,
        diarization.turns,
        texts,
        call_id=call_id,
        source=source,
        duration_s=len(audio) / SAMPLE_RATE,
        channels=channels,
        processing_ms={"diarize": diarize_ms, "asr": asr_ms},
    )
    if models.nlp:
        record, nlp_ms = timed(lambda: annotate(models, record))
        record.meta.processing_ms_by_stage["nlp"] = nlp_ms
    return record, diarization


def run_file(models: Models, path: Path) -> CallRecord:
    """Any file soundfile reads; downmixed to mono and resampled to 16 kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = decode_segment(data, sr, Segment("", 0.0, 0.0, ""))
    record, _ = process(
        models,
        audio,
        call_id=path.stem,
        source=str(path),
        num_speakers=models.cfg.num_speakers,
        channels=data.shape[1],
    )
    return record
