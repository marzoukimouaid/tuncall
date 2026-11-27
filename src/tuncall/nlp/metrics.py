"""Phase 5 scoring (D-70d): JSON reply parsers and metrics with bootstrap CIs (D-73).

Every model answers in one JSON format per task (D-32b): ``{"intent": ...}``,
``{"entities": [{"type": ..., "value": ...}]}``, ``{"sentiment": ...}``, ``{"S1": "agent", ...}``.
A reply that is not valid JSON, or names a label outside the label set, counts as wrong
(``None``) with no retry; an invalid entity reply counts as predicting no entities.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from tuncall.eval.metrics import _percentiles, _resample_indices  # pyright: ignore[reportPrivateUsage]
from tuncall.eval.normalize import normalize_arabic
from tuncall.nlp.llm import parse_json

Pairs = Sequence[tuple[str, str]]  # (type, value)
Score = dict[str, float]  # value, lo, hi


def parse_label(reply: str, key: str, labels: Sequence[str]) -> str | None:
    """``obj[key]`` if the reply is a JSON object and it is one of ``labels``."""
    obj = parse_json(reply)
    value = obj.get(key) if isinstance(obj, dict) else None
    return value if value in labels else None


def parse_entities(reply: str) -> tuple[tuple[str, str], ...] | None:
    """(type, value) pairs, or ``None`` if the reply does not follow the entity format."""
    obj = parse_json(reply)
    items: Any = obj.get("entities") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None
    pairs = []
    for item in items:
        if not (isinstance(item, dict) and set(item) == {"type", "value"}):
            return None
        if not (isinstance(item["type"], str) and isinstance(item["value"], str)):
            return None
        pairs.append((item["type"], item["value"]))
    return tuple(pairs)


def parse_roles(reply: str, speakers: Sequence[str], roles: Sequence[str]) -> dict[str, str | None]:
    """Role per speaker; a speaker missing from the reply or given an unknown role gets ``None``."""
    obj = parse_json(reply)
    obj = obj if isinstance(obj, dict) else {}
    return {s: obj.get(s) if obj.get(s) in roles else None for s in speakers}


def bootstrap(
    stat: Callable[[list[int]], float],
    n: int,
    resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Score:
    """``stat`` over all ``n`` items, with a percentile CI from resampling items."""
    if n == 0:
        raise ValueError("nothing to score")
    samples = _resample_indices(n, resamples, seed)
    lo, hi = _percentiles([stat(idx) for idx in samples], level)
    return {"value": stat(list(range(n))), "lo": lo, "hi": hi}


def macro_f1(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    """Mean F1 over every label seen in gold or predicted; ``None`` is a miss, not a label."""
    labels = set(gold) | {p for p in pred if p is not None}
    tp, fp, fn = Counter[str](), Counter[str](), Counter[str]()
    for g, p in zip(gold, pred, strict=True):
        if g == p:
            tp[g] += 1
            continue
        fn[g] += 1
        if p is not None:
            fp[p] += 1
    return sum(2 * tp[x] / (2 * tp[x] + fp[x] + fn[x]) for x in labels) / len(labels)


def accuracy(gold: Sequence[str], pred: Sequence[str | None]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def classification(gold: Sequence[str], pred: Sequence[str | None]) -> dict[str, Score]:
    """Macro-F1 and accuracy, CIs over the scored items (intent, sentiment, role)."""

    def at(metric: Callable[[Sequence[str], Sequence[str | None]], float]) -> Score:
        def stat(idx: list[int]) -> float:
            return metric([gold[i] for i in idx], [pred[i] for i in idx])

        return bootstrap(stat, len(gold))

    return {"macro_f1": at(macro_f1), "accuracy": at(accuracy)}


def entity_counts(gold: Pairs, pred: Pairs | None) -> tuple[int, int, int]:
    """(matched, predicted, gold) pairs of one utterance; values compared after normalize_arabic."""
    g = Counter((t, normalize_arabic(v)) for t, v in gold)
    p = Counter((t, normalize_arabic(v)) for t, v in pred or ())
    return sum((g & p).values()), sum(p.values()), sum(g.values())


def entity_f1(golds: Sequence[Pairs], preds: Sequence[Pairs | None]) -> Score:
    """Micro F1 over (type, value) pairs, positions ignored, CI over utterances."""
    counts = [entity_counts(g, p) for g, p in zip(golds, preds, strict=True)]

    def f1(idx: list[int]) -> float:
        tp, n_pred, n_gold = (sum(counts[i][k] for i in idx) for k in range(3))
        return 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0

    return bootstrap(f1, len(counts))
