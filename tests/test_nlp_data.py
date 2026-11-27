import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import openai
import pytest
from make_synthetic_calls import valid
from transliterate_tsac import convert

from tuncall.nlp.data import is_arabic, parse_markup, read_arabizi, tsac_holdout
from tuncall.nlp.llm import LLM, QuotaExhausted, parse_json


def test_markup_is_parsed_before_each_value_is_transliterated() -> None:
    markup = "ابعث mail [relation : لصحبتي] [person :سلمى] [app_name : gmail]"
    lexicon = {"gmail": "جيمايل", "mail": "مايل"}
    assert parse_markup(markup, lexicon) == (
        ("relation", "لصحبتي"),
        ("person", "سلمى"),
        ("app_name", "جيمايل"),
    )


def test_script_is_the_majority_of_letters() -> None:
    assert is_arabic("صراحة bravo أحمد")
    assert not is_arabic("3imed d8ij bravo wlh")
    assert not is_arabic("😀")


def test_holdout_is_stratified_and_seeded() -> None:
    pool = [(str(i), "x", "positive" if i % 3 else "negative") for i in range(300)]
    ids = tsac_holdout(pool, 0.1, seed=0)
    labels = {i: label for i, _, label in pool}
    assert sum(labels[i] == "negative" for i in ids) == 10
    assert len(ids) == 30
    assert ids == tsac_holdout(pool, 0.1, seed=0)


class FakeClient:
    def __init__(self, replies: list[Any]) -> None:
        self.replies, self.calls = replies, 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_: Any) -> Any:
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def rate_limit(message: str) -> openai.RateLimitError:
    req = httpx.Request("POST", "https://x")
    return openai.RateLimitError(message, response=httpx.Response(429, request=req), body=None)


def test_reply_is_cached_by_request(tmp_path: Path) -> None:
    client = FakeClient(["a", "b"])
    llm = LLM("groq", "m", tmp_path, client=client)
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.chat(msgs, temperature=0) == "a"
    assert llm.chat(msgs, temperature=0) == "a"
    assert llm.chat(msgs, temperature=1) == "b"  # other parameters, other key
    assert client.calls == 2
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["model"] == "m" and "seconds" in record


def test_rate_limit_is_waited_out_but_daily_quota_stops(tmp_path: Path) -> None:
    waits: list[float] = []
    client = FakeClient([rate_limit("per minute"), "ok"])
    llm = LLM("gemini", "m", tmp_path, client=client, sleep=waits.append)
    assert llm.chat([{"role": "user", "content": "1"}]) == "ok"
    assert waits == [10.0]
    llm = LLM("gemini", "m", tmp_path, client=FakeClient([rate_limit("quota per day")]))
    with pytest.raises(QuotaExhausted):
        llm.chat([{"role": "user", "content": "2"}])


def test_parse_json_accepts_a_fence_and_rejects_prose() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("Sure! {") is None


def test_wrong_count_reply_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(['{"texts": ["عسلامة"]}', '{"texts": ["برافو", "مرسي"]}'])
    llm = LLM("gemini", "m", tmp_path, client=client)
    assert convert(llm, ["3aslema", "bravo"]) is None
    assert convert(llm, ["bravo", "merci"]) == ["برافو", "مرسي"]


def test_call_needs_both_roles_and_arabic_lines() -> None:
    line = {"role": "agent", "text": "عسلامة  شنوة نجم نعاونك"}
    other = {"role": "customer", "text": "عندي مشكل"}
    assert valid({"lines": [line, other]}, 2, 4) is not None
    assert valid({"lines": [line, line]}, 2, 4) is None
    assert valid({"lines": [line, {"role": "customer", "text": "hello there"}]}, 2, 4) is None
    assert valid({"lines": [line, {"role": "boss", "text": "عندي"}]}, 2, 4) is None
    assert valid({"lines": [line]}, 2, 4) is None
    assert valid([line], 2, 4) is None


def test_unfinished_or_edited_conversions_are_refused(tmp_path: Path) -> None:
    tsv, word = tmp_path / "a.tsv", "برافو"
    tsv.write_text(f"id\ttext\n1\t{word}\n", encoding="utf-8")
    sha = hashlib.sha256(tsv.read_bytes()).hexdigest()
    meta = tmp_path / "a.yaml"
    meta.write_text(f"sha256: {sha}\ncomplete: false\n")
    with pytest.raises(ValueError, match="unfinished"):
        read_arabizi(tsv)
    meta.write_text(f"sha256: {sha}\ncomplete: true\n")
    assert read_arabizi(tsv) == {"1": word}
    tsv.write_text(f"id\ttext\n1\t{word}x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        read_arabizi(tsv)
