from pathlib import Path

import numpy as np
import pytest

from test_tracking import FakeClient
from tuncall.nlp.data import Call, CallLine
from tuncall.nlp.encoder import bio_tags, spans, vote
from tuncall.nlp.metrics import parse_entities, parse_label, parse_roles
from tuncall.nlp.tasks import DataRef, Example
from tuncall.tracking import Tracker

MARBERT = ("UBC-NLP/MARBERTv2", "fe88db9db8ccdb0c4e1627495f405c44a5f89066")
QWEN = ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")
DATA = DataRef(lexicon=Path("x"), slurp_revision="x", calls=Path("x"))


def test_bio_round_trip() -> None:
    words = ["حط", "منبه", "غدوة", "الصباح", "في", "تونس"]
    pairs = (("time", "غدوة الصباح"), ("place", "تونس"), ("person", "علي"))
    tags, missed = bio_tags(words, pairs)
    assert tags == ["O", "O", "B-time", "I-time", "O", "B-place"]
    assert missed == 1
    assert spans(words, tags) == (("time", "غدوة الصباح"), ("place", "تونس"))
    assert spans(["a", "b"], ["I-x", "I-y"]) == (("x", "a"), ("y", "b"))


def test_vote_uses_mean_agent_probability() -> None:
    rows = [("c", "S1", "", ""), ("c", "S1", "", ""), ("c", "S2", "", "")]
    assert vote(rows, np.array([0.9, 0.3, 0.2])) == {("c", "S1"): "agent", ("c", "S2"): "customer"}


def call(i: int) -> Call:
    lines = [CallLine(role="agent", text=f"مرحبا {i}"), CallLine(role="customer", text="عندي مشكل")]
    return Call(id=str(i), domain="bank", scenario="x", lines=lines * 2)


def test_chat_answers_parse_back() -> None:
    from tuncall.nlp.qwen import chat

    ex = Example("1", "حط منبه", (("time", "غدوة"),))
    _, answer, gold = chat("entities", ["time"], ex)
    assert parse_entities(answer) == gold
    _, answer, _ = chat("intent", ["alarm_set"], Example("1", "x", "alarm_set"))
    assert parse_label(answer, "intent", ["alarm_set"]) == "alarm_set"
    messages, answer, gold = chat("role", ["agent", "customer"], Example("1", "", None, call(1)), 0)
    assert parse_roles(answer, sorted(gold), ["agent", "customer"]) == gold
    assert messages[1]["content"].count("\n") == 3


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import AutoTokenizer, BertConfig, BertModel

    config = BertConfig.from_pretrained(MARBERT[0], revision=MARBERT[1])
    config.update({"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                   "intermediate_size": 32})  # fmt: skip
    path = tmp_path_factory.mktemp("tiny-bert")
    BertModel(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(MARBERT[0], revision=MARBERT[1]).save_pretrained(path)
    return path


@pytest.mark.parametrize("task", ["intent", "entities", "role"])
def test_encoder_trains_and_resumes(tiny_bert: Path, tmp_path: Path, task: str) -> None:
    from tuncall.asr.train import ModelRef
    from tuncall.nlp.encoder import EncoderConfig, train_encoder

    if task == "intent":
        data = [Example(str(i), f"حط منبه {i}", ["a", "b"][i % 2]) for i in range(8)]
        labels = ["a", "b"]
    elif task == "entities":
        data = [Example(str(i), f"حط منبه غدوة {i}", (("time", "غدوة"),)) for i in range(8)]
        labels = ["time"]
    else:
        data = [Example(str(i), "", None, call(i)) for i in range(4)]
        labels = ["agent", "customer"]
    cfg = EncoderConfig(
        experiment="t",
        data=DATA,
        models={"tiny": ModelRef(id=str(tiny_bert), revision="main")},
        max_length=32,
        output_root=tmp_path,
        trainer={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "save_only_model": task == "role",
            "metric_for_best_model": "score",
            "use_cpu": True,
        },
    )
    with Tracker(
        "t", config={}, dataset="d", client=FakeClient(), buffer_path=tmp_path / "b.jsonl"
    ) as tracker:
        preds, score = train_encoder(cfg, "tiny", task, labels, data, data, tracker)
        assert 0.0 <= score <= 1.0
        assert len(preds) == (1 if task == "role" else len(data))
        assert (tmp_path / "tiny" / task / "best").is_dir()
        cfg.trainer["num_train_epochs"] = 2  # resumes from the epoch-1 checkpoint
        train_encoder(cfg, "tiny", task, labels, data, data, tracker)

    from tuncall.nlp.predict import Encoder, parameters

    encoder = Encoder(task, tmp_path / "tiny" / task / "best", "cpu", batch_size=3)
    if task == "role":
        out = encoder([[("S1", "مرحبا"), ("S2", "عندي مشكل")], []])
        assert set(out[0]) == {"S1", "S2"} and out[1] == {}
        assert set(out[0].values()) <= {"agent", "customer"}
    else:
        out = encoder([ex.text for ex in data])
        assert len(out) == len(data)
        if task == "intent":
            assert set(out) <= {"a", "b"}
        else:
            assert all(isinstance(p, tuple) for p in out)
    assert parameters(encoder) > 0


def test_qwen_lora_trains_and_generates(tmp_path: Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from tuncall.asr.train import LoraSettings, ModelRef
    from tuncall.nlp.qwen import QwenConfig, chat, generate, train_qwen

    config = AutoConfig.from_pretrained(QWEN[0], revision=QWEN[1])
    config.update({"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                   "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32,
                   "layer_types": config.layer_types[:1]})  # fmt: skip
    base = tmp_path / "tiny-qwen"
    AutoModelForCausalLM.from_config(config).save_pretrained(base)
    AutoTokenizer.from_pretrained(QWEN[0], revision=QWEN[1]).save_pretrained(base)
    cfg = QwenConfig(
        experiment="t",
        data=DATA,
        base_model=ModelRef(id=str(base), revision="main"),
        quantize_4bit=False,
        max_length=256,
        shuffle_seed=0,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj", "v_proj"]),
        output_root=tmp_path / "out",
        eval_batch_size=2,
        trainer={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "use_cpu": True,
            "save_strategy": "epoch",
        },
    )
    data = [Example(str(i), "", None, call(i)) for i in range(3)]
    with Tracker(
        "t", config={}, dataset="d", client=FakeClient(), buffer_path=tmp_path / "b.jsonl"
    ) as tracker:
        model, tok = train_qwen(cfg, "role", ["agent", "customer"], data, tracker, str(base))
    prompts = [chat("role", ["agent", "customer"], ex)[0] for ex in data]
    replies = generate(model, tok, prompts, max_new_tokens=4, batch=2)
    assert len(replies) == 3 and all(isinstance(r, str) for r in replies)
    assert (tmp_path / "out/role/final").is_dir()

    import yaml

    from tuncall.nlp.predict import Qwen, QwenTask, parameters

    path = tmp_path / "qwen.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    qwen = Qwen(path, {"role": tmp_path / "out/role/final"}, "cpu")
    out = qwen.predict("role", [[("S1", "مرحبا"), ("S2", "عندي مشكل")]])
    assert set(out[0]) == {"S1", "S2"}  # values: a role or None (invalid reply)
    assert parameters(QwenTask(qwen, "role")) > 0
