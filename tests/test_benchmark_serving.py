"""The serving benchmark's metrics on hand-made call timings (D-70e)."""

import pytest
from benchmark_serving import latency, load, percentile


def call(created: str, claimed: str, finished: str, duration_s: float, **ms: float) -> dict:
    day = "2026-11-28T10:00:"
    return {
        "created_at": day + created + "+00:00",
        "claimed_at": day + claimed + "+00:00",
        "finished_at": day + finished + "+00:00",
        "record": {"duration_s": duration_s, "meta": {"processing_ms_by_stage": ms}},
    }


def test_percentile_is_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 95) == 5
    assert percentile([7], 95) == 7


def test_latency_per_minute_of_audio_and_cost() -> None:
    calls = [
        call("00", "00", "10", 60, diarize=2000, asr=6000, nlp=1000),
        call("10", "10", "20", 60, diarize=2000, asr=6000),
    ]
    out = latency(calls, usd_per_hour=0.36)
    assert out["diarize_s_per_min"] == pytest.approx(2.0)
    assert out["asr_s_per_min"] == pytest.approx(6.0)
    assert out["nlp_s_per_min"] == pytest.approx(0.5)
    assert out["total_s_per_min"] == pytest.approx(10.0)  # 20 s worker time / 2 min
    # 20 s of GPU per 2 min of audio = 1/6 GPU-hour per audio hour.
    assert out["usd_per_audio_hour"] == pytest.approx(0.36 / 6)


def test_load_queue_wait_and_throughput() -> None:
    calls = [call("00", "00", "10", 60), call("00", "10", "20", 60), call("00", "20", "30", 60)]
    out = load(calls)
    assert out["calls_per_hour"] == pytest.approx(3 / 30 * 3600)
    assert out["queue_wait_mean_s"] == pytest.approx(10.0)
    assert out["queue_wait_p95_s"] == 20
    assert out["time_to_result_p50_s"] == 20 and out["time_to_result_p95_s"] == 30
