"""A fixed CallRecord for the serving tests (stands in for ``pipeline.process``)."""

from tuncall.record import CallRecord, Meta, Segment


def fixed_record(call_id: str = "c1") -> CallRecord:
    return CallRecord(
        call_id=call_id,
        source="call.wav",
        duration_s=3.0,
        sample_rate=16_000,
        channels=1,
        segments=[
            Segment(
                start=0.0,
                end=1.5,
                speaker="S1",
                role="agent",
                text_raw="عسلامة",
                text_normalized="عسلامه",
                asr_model_version="m@r",
                intent="greet",
                sentiment="positive",
            )
        ],
        meta=Meta(
            pipeline_version="0.0.0+test",
            diarization_model_version="d@r",
            processing_ms_by_stage={"diarize": 1.0, "asr": 2.0, "nlp": 3.0},
        ),
    )
