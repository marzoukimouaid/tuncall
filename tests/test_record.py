import pytest
from pydantic import ValidationError

from tuncall.record import CallRecord

RECORD = {
    "call_id": "dev-000",
    "source": "conversations-dev",
    "duration_s": 12.5,
    "sample_rate": 16000,
    "channels": 1,
    "segments": [
        {
            "start": 0.0,
            "end": 4.2,
            "speaker": "S1",
            "text_raw": "عسلامة",
            "text_normalized": "عسلامه",
            "asr_model_version": "openai/whisper-large-v3-turbo@abc+lora-r32",
        }
    ],
    "meta": {
        "pipeline_version": "0.1.0+abc123",
        "diarization_model_version": "pyannote/speaker-diarization-community-1@def",
        "processing_ms_by_stage": {"diarize": 120.0, "asr": 900.0},
    },
}


def test_round_trips_through_json() -> None:
    record = CallRecord.model_validate(RECORD)
    assert record.segments[0].role == "unknown"
    assert CallRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    "segment_patch",
    [{"speaker": "agent"}, {"role": "customer"}, {"words": []}, {"start": -1.0}],
)
def test_rejects_fields_outside_phase4(segment_patch: dict[str, object]) -> None:
    bad = {**RECORD, "segments": [{**RECORD["segments"][0], **segment_patch}]}  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        CallRecord.model_validate(bad)
