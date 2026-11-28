"""The ``CallRecord`` that flows through every pipeline stage (PROJECT.md §4, D-55).

Grown one phase at a time. Phase 4 holds the call info, the diarized and transcribed
segments and run metadata. Speakers are anonymous (``S1``, ``S2``, ...). No ``words[]``:
times are per turn only (D-23a). Phase 5 (D-55a) adds per segment the intent, entities and
sentiment, the speaker's role guess, and ``meta.nlp_model_versions``; they stay empty when
the pipeline runs without NLP models.
"""

from typing import Literal

from pydantic import Field

from tuncall.config import StrictModel


class Entity(StrictModel):
    type: str
    value: str


class Segment(StrictModel):
    start: float = Field(ge=0)  # seconds from the start of the recording
    end: float = Field(ge=0)
    speaker: str = Field(pattern=r"^S\d+$")
    role: Literal["agent", "customer", "unknown"] = "unknown"  # the speaker's role guess
    text_raw: str  # as the ASR model wrote it
    text_normalized: str  # D-10 normalization, as used for scoring
    asr_model_version: str  # model id + revision (+ adapter) that wrote text_raw
    intent: str | None = None  # SLURP-TN voice-assistant intents (see the model card)
    entities: list[Entity] = Field(default_factory=list)
    sentiment: Literal["positive", "negative"] | None = None


class Meta(StrictModel):
    pipeline_version: str  # tuncall version + git commit
    diarization_model_version: str
    processing_ms_by_stage: dict[str, float]  # wall-clock milliseconds
    # task -> model id + revision + fine-tuned path
    nlp_model_versions: dict[str, str] = Field(default_factory=dict)


class CallRecord(StrictModel):
    call_id: str
    source: str  # file path or dataset id the audio came from
    duration_s: float = Field(ge=0)
    sample_rate: int
    channels: int
    segments: list[Segment]
    meta: Meta
