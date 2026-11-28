import pytest

from tuncall.nlp.data import ROLES, Call, CallLine, call_turns
from tuncall.nlp.metrics import (
    accuracy,
    classification,
    entity_counts,
    entity_f1,
    macro_f1,
    parse_entities,
    parse_label,
    parse_roles,
)


def test_parse_label() -> None:
    labels = ["positive", "negative"]
    assert parse_label('{"sentiment": "positive"}', "sentiment", labels) == "positive"
    assert parse_label('```json\n{"sentiment": "negative"}\n```', "sentiment", labels) == "negative"
    assert parse_label('{"sentiment": "neutral"}', "sentiment", labels) is None
    assert parse_label("positive", "sentiment", labels) is None
    assert parse_label('["positive"]', "sentiment", labels) is None


def test_parse_entities() -> None:
    reply = '{"entities": [{"type": "time", "value": "الصباح"}]}'
    assert parse_entities(reply) == (("time", "الصباح"),)
    assert parse_entities('{"entities": []}') == ()
    assert parse_entities('{"entities": [{"type": "time"}]}') is None
    assert parse_entities('{"entities": [{"type": "time", "value": 3}]}') is None
    assert parse_entities("not json") is None


def test_parse_roles() -> None:
    got = parse_roles('{"S1": "customer", "S2": "boss"}', ["S1", "S2"], ROLES)
    assert got == {"S1": "customer", "S2": None}
    assert parse_roles("oops", ["S1"], ROLES) == {"S1": None}


def test_macro_f1_and_accuracy() -> None:
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", None]
    # a: tp1 fn1 fp0 -> 2/3; b: tp1 fp1 fn1 -> 1/2
    assert macro_f1(gold, pred) == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert accuracy(gold, pred) == 0.5
    assert macro_f1(["a"], ["a"]) == 1.0


def test_classification_ci_brackets_value() -> None:
    gold = ["a", "b"] * 50
    pred = ["a", "b"] * 40 + ["b", "a"] * 10
    scores = classification(gold, pred)
    for s in scores.values():
        assert s["lo"] <= s["value"] <= s["hi"]
    assert scores["accuracy"]["value"] == 0.8


def test_entity_pairs_ignore_positions_and_orthography() -> None:
    gold = [("place", "إلى تونس"), ("time", "غدوة")]
    pred = [("time", "غدوة"), ("place", "الى تونس"), ("place", "صفاقس")]
    assert entity_counts(gold, pred) == (2, 3, 2)
    assert entity_counts(gold, None) == (0, 0, 2)
    assert entity_f1([gold, []], [pred, []])["value"] == pytest.approx(2 * 2 / 5)
    assert entity_f1([[]], [[]])["value"] == 1.0


def test_call_turns_shuffled_names_by_first_appearance() -> None:
    lines = [CallLine(role="agent", text=f"a{i}") for i in range(5)]
    lines += [CallLine(role="customer", text=f"c{i}") for i in range(5)]
    call = Call(id="7", domain="bank", scenario="x", lines=lines)
    plain, roles = call_turns(call)
    assert plain[0] == ("S1", "a0") and roles == {"S1": "agent", "S2": "customer"}
    shuffled, sroles = call_turns(call, shuffle_seed=0)
    assert shuffled == call_turns(call, shuffle_seed=0)[0]
    assert sorted(t for _, t in shuffled) == sorted(t for _, t in plain)
    speaker, text = shuffled[0]
    assert speaker == "S1"
    assert sroles["S1"] == ("agent" if text.startswith("a") else "customer")
