import pytest

from tuncall.eval.metrics import (
    bootstrap_ci,
    char_counts,
    error_rate,
    paired_delta_ci,
    word_counts,
)
from tuncall.eval.normalize import normalize_arabic


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أحمد إلى آخر", "احمد الي اخر"),  # alef variants, alef maqsura
        ("مدرسة", "مدرسه"),  # ta marbuta
        ("مُحَمَّدٌ", "محمد"),  # diacritics
        ("كـــبير", "كبير"),  # tatweel
        ("عندي ٣ أرقام ۴٥", "عندي 3 ارقام 45"),  # Arabic-Indic digits
        ("شنوة؟ باهي، يعطيك الصحة!", "شنوه باهي يعطيك الصحه"),  # punctuation
        ("  برشا \n  برشا  ", "برشا برشا"),  # whitespace
        ("ﻻ", "لا"),  # presentation form
        ("", ""),
    ],
)
def test_normalize_arabic(raw: str, expected: str) -> None:
    assert normalize_arabic(raw) == expected


def test_word_error_rate_hand_computed() -> None:
    # 1 substitution + 1 insertion over 3 + 1 reference words.
    counts = word_counts(["a b c", "d"], ["a x c e", "d"])
    assert counts == [(2, 3), (0, 1)]
    assert error_rate(counts) == pytest.approx(2 / 4)


def test_char_counts_include_spaces() -> None:
    assert char_counts(["ab c"], ["ab d"]) == [(1, 4)]


def test_empty_reference_utterance_counts_insertions() -> None:
    counts = word_counts(["", "a"], ["x y", "a"])
    assert counts == [(2, 0), (0, 1)]
    assert error_rate(counts) == 2.0


def test_error_rate_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        error_rate([(0, 0)])


def test_bootstrap_ci_contains_estimate_and_is_reproducible() -> None:
    counts = [(i % 3, 5) for i in range(200)]
    lo, hi = bootstrap_ci(counts, resamples=500, seed=1)
    assert lo <= error_rate(counts) <= hi
    assert lo < hi
    assert bootstrap_ci(counts, resamples=500, seed=1) == (lo, hi)


def test_paired_delta_ci_detects_a_clear_difference() -> None:
    worse = [(2, 5)] * 100
    better = [(1, 5)] * 100
    lo, hi = paired_delta_ci(worse, better, resamples=200)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.2)
