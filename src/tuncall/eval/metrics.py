"""Corpus WER/CER with bootstrap confidence intervals (D-70a, D-73).

Scores are computed from per-utterance (errors, reference length) counts, so a bootstrap
resample only re-sums counts instead of re-aligning text.
"""

import random
from collections.abc import Sequence

import jiwer

Counts = list[tuple[int, int]]  # per utterance: (S + D + I errors, reference length)


def _counts(out: jiwer.WordOutput | jiwer.CharacterOutput) -> tuple[int, int]:
    errors = out.substitutions + out.deletions + out.insertions
    return errors, out.hits + out.substitutions + out.deletions


def word_counts(refs: Sequence[str], hyps: Sequence[str]) -> Counts:
    assert len(refs) == len(hyps), "refs and hyps must align"
    return [_counts(jiwer.process_words(r, h)) for r, h in zip(refs, hyps, strict=True)]


def char_counts(refs: Sequence[str], hyps: Sequence[str]) -> Counts:
    assert len(refs) == len(hyps), "refs and hyps must align"
    return [_counts(jiwer.process_characters(r, h)) for r, h in zip(refs, hyps, strict=True)]


def error_rate(counts: Counts) -> float:
    total = sum(n for _, n in counts)
    if total == 0:
        raise ValueError("reference is empty")
    return sum(e for e, _ in counts) / total


def _resample_indices(n: int, resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.choices(range(n), k=n) for _ in range(resamples)]


def _percentiles(values: list[float], level: float) -> tuple[float, float]:
    values = sorted(values)
    tail = (1 - level) / 2
    return values[int(tail * (len(values) - 1))], values[round((1 - tail) * (len(values) - 1))]


def _rate_at(counts: Counts, idx: list[int]) -> float:
    total = sum(counts[i][1] for i in idx)
    return sum(counts[i][0] for i in idx) / total if total else 0.0


def bootstrap_ci(
    counts: Counts, resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the corpus error rate, resampling utterances with replacement."""
    samples = _resample_indices(len(counts), resamples, seed)
    return _percentiles([_rate_at(counts, idx) for idx in samples], level)


def paired_delta_ci(
    counts_a: Counts, counts_b: Counts, resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """CI of rate(a) - rate(b), with both systems scored on the same resampled utterances."""
    assert len(counts_a) == len(counts_b), "systems must be scored on the same utterances"
    samples = _resample_indices(len(counts_a), resamples, seed)
    deltas = [_rate_at(counts_a, idx) - _rate_at(counts_b, idx) for idx in samples]
    return _percentiles(deltas, level)
