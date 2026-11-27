from pathlib import Path

from tuncall.asr import ModelRef
from tuncall.diarize import Span, relabel
from tuncall.nlp.predict import NlpModel
from tuncall.pipeline import Models, NlpSettings, PipelineConfig, annotate, build_record


def test_relabel_names_by_first_speech_and_merges_turns() -> None:
    regions = [(5.0, 9.0, "SPEAKER_00"), (0.0, 4.0, "SPEAKER_01"), (3.5, 6.0, "SPEAKER_00")]
    turns = [(0.0, 3.5, "SPEAKER_01"), (3.5, 6.0, "SPEAKER_00"), (6.0, 9.0, "SPEAKER_00")]
    d = relabel(regions, turns)
    assert d.regions[0] == Span(0.0, 4.0, "S1")  # first to speak is S1
    assert {r.speaker for r in d.regions} == {"S1", "S2"}
    assert d.turns == [Span(0.0, 3.5, "S1"), Span(3.5, 9.0, "S2")]


def test_build_record_fills_phase4_fields() -> None:
    cfg = PipelineConfig(
        asr=ModelRef(id="openai/whisper-large-v3-turbo", revision="abc", adapter=None),
        diarization_revision="def",
    )
    models = Models(cfg, diarizer=None, whisper=None, processor=None, lexicon={"seo": "سيو"})
    record = build_record(
        models,
        [Span(0.0, 1.25, "S1"), Span(1.5, 3.0, "S2")],
        ["عسلامة seo", "إي"],
        call_id="dev-000",
        source="dev-000.wav",
        duration_s=3.0,
        channels=1,
        processing_ms={"diarize": 1.0, "asr": 2.0},
    )
    assert [s.speaker for s in record.segments] == ["S1", "S2"]
    assert record.segments[0].text_normalized == "عسلامه سيو"
    assert record.segments[0].asr_model_version == "openai/whisper-large-v3-turbo@abc"
    assert record.meta.diarization_model_version.endswith("@def")


def test_annotate_fills_phase5_fields() -> None:
    from tuncall.asr.train import ModelRef as HubRef

    ref = HubRef(id="m", revision="r")
    cfg = PipelineConfig(
        asr=ModelRef(id="w", revision="abc"),
        diarization_revision="def",
        nlp=NlpSettings(
            tasks={
                t: NlpModel(kind="encoder", base=ref, path=Path(t))
                for t in ("intent", "entities", "sentiment", "role")
            }
        ),
    )
    seen: list[object] = []

    def role(calls: list[list[tuple[str, str]]]) -> list[dict[str, str]]:
        seen.append(calls)
        return [{"S1": "agent"}]  # S2 gets no role: stays unknown

    nlp = {
        "intent": lambda xs: ["alarm_set"] * len(xs),
        "sentiment": lambda xs: [None, "negative"],
        "entities": lambda xs: [(("time", "غدوة"),), ()],
        "role": role,
    }
    models = Models(cfg, None, None, None, {"seo": "سيو"}, nlp)
    record = build_record(
        models,
        [Span(0.0, 1.0, "S1"), Span(1.0, 2.0, "S2")],
        ["عسلامة seo", "إي"],
        call_id="c",
        source="c.wav",
        duration_s=2.0,
        channels=1,
        processing_ms={},
    )
    out = annotate(models, record)
    assert seen == [[[("S1", "عسلامة سيو"), ("S2", "إي")]]]  # lexicon applied (D-34a)
    assert [s.role for s in out.segments] == ["agent", "unknown"]
    assert [s.sentiment for s in out.segments] == [None, "negative"]
    assert out.segments[0].intent == "alarm_set"
    assert out.segments[0].entities[0].value == "غدوة" and out.segments[1].entities == []
    assert out.meta.nlp_model_versions["role"] == "m@r+role"
