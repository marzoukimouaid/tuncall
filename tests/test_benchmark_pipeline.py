from pathlib import Path

import numpy as np
from benchmark_pipeline import by_speaker, cached, cp_counts, der_counts, write_atomic

from tuncall.data.conversations import Conversation, Turn
from tuncall.data.testsets import SAMPLE_RATE
from tuncall.diarize import Span


def test_by_speaker_keeps_time_order_per_speaker() -> None:
    spans = [("S1", "a b"), ("S2", "c"), ("S1", "d")]
    assert by_speaker(spans) == ["a b d", "c"]


def test_cpwer_ignores_label_swaps_but_counts_wrong_speaker_words() -> None:
    ref = ["a b c", "d e"]
    assert cp_counts(ref, ["d e", "a b c"]) == (0, 5)  # labels swapped: no error
    # "c" given to the other speaker: deleted from one, inserted in the other
    assert cp_counts(ref, ["a b", "c d e"]) == (2, 5)
    # a third estimated speaker: its words are insertions
    assert cp_counts(ref, ["a b c", "d e", "x"]) == (1, 5)
    # one speaker found: the other reference speaker is all deletions
    assert cp_counts(ref, ["a b c"]) == (2, 5)


def test_der_is_zero_for_a_perfect_relabelled_diarization() -> None:
    clips = [Turn("S1", 0.0, 2.0, ""), Turn("S2", 2.5, 4.0, "")]
    conv = Conversation("c", np.zeros(4 * SAMPLE_RATE, dtype=np.float32), clips, clips)
    errors, total = der_counts(conv, [Span(0.0, 2.0, "S2"), Span(2.5, 4.0, "S1")])
    assert errors == 0 and 3000 <= total < 3500  # the collar (0.25 s wide) is not scored
    # S2's clip given to S1: about its 1.5 s are confused, out of the same total
    errors, total2 = der_counts(conv, [Span(0.0, 2.0, "S1"), Span(2.5, 4.0, "S1")])
    assert total2 == total and 1000 <= errors <= 1500


def test_cache_counts_only_with_the_same_key(tmp_path: Path) -> None:
    path = tmp_path / "x" / "m.jsonl"
    assert not cached(path, {"a": 1})
    write_atomic(path, ["{}"], {"a": 1})
    assert cached(path, {"a": 1})
    assert not cached(path, {"a": 2})  # config changed: recompute
    path.with_suffix(".key.json").unlink()
    assert not cached(path, {"a": 1})  # no key (old or half-written file): recompute
