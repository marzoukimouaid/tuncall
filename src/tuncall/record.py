"""The ``CallRecord`` that flows through every pipeline stage (PROJECT.md §4, D-55).

Grown one phase at a time. Phase 4 holds the call info, the diarized and transcribed
segments and run metadata. Speakers are anonymous (``S1``, ``S2``, ...) and their role is
always ``unknown`` until Phase 5. No ``words[]``: times are per turn only (D-23a).
"""

from typing import Literal

from pydantic import Field

from tuncall.config import StrictModel


class Segment(StrictModel):
    start: float = Field(ge=0)  # seconds from the start of the recording
    end: float = Field(ge=0)
    speaker: str = Field(pattern=r"^S\d+$")
    role: Literal["unknown"] = "unknown"
    text_raw: str  # as the ASR model wrote it
    text_normalized: str  # D-10 normalization, as used for scoring
    asr_model_version: str  # model id + revision (+ adapter) that wrote text_raw


class Meta(StrictModel):
    pipeline_version: str  # tuncall version + git commit
    diarization_model_version: str
    processing_ms_by_stage: dict[str, float]  # wall-clock milliseconds


class CallRecord(StrictModel):
    call_id: str
    source: str  # file path or dataset id the audio came from
    duration_s: float = Field(ge=0)
    sample_rate: int
    channels: int
    segments: list[Segment]
    meta: Meta
