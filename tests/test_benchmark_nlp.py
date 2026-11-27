from benchmark_nlp import MAIN, shipping, tables

from tuncall.nlp.tasks import TASKS


def row(value: float, ms: float) -> dict[str, object]:
    s = {"value": value, "lo": value, "hi": value}
    return {
        "n": 1,
        "ms_per_input": ms,
        "parameters": 1e6,
        "scores": {"macro_f1": s, "accuracy": s, "f1": s, "accuracy_shuffled": s},
    }


def test_shipping_prefers_the_faster_model_within_one_point() -> None:
    enc = {t: row(0.800, 1.0) for t in TASKS}
    qwen = {t: row(0.805, 50.0) for t in TASKS}  # within 1 point: the faster encoder ships
    qwen["entities"] = row(0.850, 50.0)  # clearly better: Qwen ships
    assert shipping({"encoder": enc, "qwen": qwen}) == {
        "intent": "encoder",
        "entities": "qwen",
        "sentiment": "encoder",
        "role": "encoder",
    }
    assert set(MAIN) == set(TASKS)
    assert "| encoder | gold text | 80.0 [80.0, 80.0]" in tables({"encoder": enc})
