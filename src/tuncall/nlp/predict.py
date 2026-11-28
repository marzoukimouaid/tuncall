"""Inference with the trained Phase 5 models, shared by the benchmark and the pipeline (D-55a).

Inputs are already in the training text convention (``asr.cache.target_text``, D-34a): a
string for intent, entities and sentiment, and a list of (speaker, text) turns for role.
Outputs: a label (``None`` if a Qwen reply is invalid), (type, value) pairs, or
{speaker: role}.
"""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)

from tuncall.asr.train import ModelRef
from tuncall.config import StrictModel, load_config
from tuncall.data.testsets import REPO_ROOT
from tuncall.nlp.data import ROLES
from tuncall.nlp.encoder import EncoderTask
from tuncall.nlp.metrics import parse_entities, parse_label, parse_roles
from tuncall.nlp.tasks import Example

Turns = list[tuple[str, str]]  # (speaker, text)


class NlpModel(StrictModel):
    kind: Literal["encoder", "qwen"]
    base: ModelRef  # the pinned Hub model it was trained from
    path: Path  # encoder: the fine-tuned model dir; qwen: the LoRA adapter dir

    @property
    def version(self) -> str:
        return f"{self.base.id}@{self.base.revision}+{self.path}"


class Encoder:
    """One fine-tuned encoder; bf16 on GPU."""

    def __init__(self, task: str, path: Path, device: str, batch_size: int = 64) -> None:
        cls = (
            AutoModelForTokenClassification
            if task == "entities"
            else AutoModelForSequenceClassification
        )
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.task, self.device, self.batch_size = task, device, batch_size
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = cls.from_pretrained(path, dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.et = EncoderTask(task, [], self.tok, 128)  # max_length as in training (D-32f)
        self.et.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.et.index = {lab: i for i, lab in enumerate(self.et.labels)}
        self.collate = (
            DataCollatorForTokenClassification(self.tok)
            if task == "entities"
            else DataCollatorWithPadding(self.tok)
        )

    @torch.no_grad()
    def _forward(self, rows: list[dict[str, Any]]) -> tuple[Any, Any]:
        logits, labels = [], []
        for i in range(0, len(rows), self.batch_size):
            batch = {
                k: v.to(self.device) for k, v in self.collate(rows[i : i + self.batch_size]).items()
            }
            lab = batch.pop("labels")
            logits += list(self.model(**batch).logits.float().cpu().numpy())
            labels += list(lab.cpu().numpy())
        if self.task == "entities":  # per row: padded to its batch's width
            return logits, labels
        return np.stack(logits), np.stack(labels)

    def __call__(self, inputs: list[Any]) -> list[Any]:
        if not inputs:
            return []
        if self.task == "role":
            meta = [(str(i), s, t, "") for i, turns in enumerate(inputs) for s, t in turns]
            if not meta:
                return [{} for _ in inputs]
            enc = self.tok([t for _, _, t, _ in meta], truncation=True, max_length=128)
            rows = [{k: enc[k][j] for k in enc} | {"labels": 0} for j in range(len(meta))]
            votes = self.et.predict(*self._forward(rows), meta)[0]
            return [{s: votes[(str(i), s)] for s, _ in turns} for i, turns in enumerate(inputs)]
        gold = () if self.task == "entities" else self.et.labels[0]
        rows, meta = self.et.encode([Example(str(i), t, gold) for i, t in enumerate(inputs)])
        return self.et.predict(*self._forward(rows), meta)


class Qwen:
    """Qwen3 4B (4-bit) with one LoRA adapter per task, switched per call."""

    def __init__(self, config: Path, adapters: dict[str, Path], device: str) -> None:
        from peft import PeftModel

        from tuncall.nlp.qwen import LABELS, QwenConfig, load_base

        self.cfg = load_config(REPO_ROOT / config, QwenConfig)
        self.base_model = self.cfg.base_model
        base, self.tok = load_base(self.cfg)
        if not self.cfg.quantize_4bit:  # 4-bit weights are placed on the GPU when loaded
            base = base.to(device)
        self.labels: dict[str, list[str]] = {}
        model: Any = None
        for task, path in adapters.items():
            path = REPO_ROOT / path
            if model is None:
                model = PeftModel.from_pretrained(base, str(path), adapter_name=task)
            else:
                model.load_adapter(str(path), adapter_name=task)
            self.labels[task] = json.loads((path / LABELS).read_text(encoding="utf-8"))
        self.model = model

    def predict(self, task: str, inputs: list[Any]) -> list[Any]:
        from tuncall.nlp.qwen import MAX_NEW_TOKENS, generate, messages, role_text

        if not inputs:
            return []
        labels = self.labels[task]
        texts = [role_text(x) if task == "role" else x for x in inputs]
        self.model.set_adapter(task)
        replies = generate(
            self.model,
            self.tok,
            [messages(task, labels, t) for t in texts],
            MAX_NEW_TOKENS[task],
            self.cfg.eval_batch_size,
        )
        if task == "role":
            return [
                parse_roles(r, sorted({s for s, _ in x}), ROLES)
                for r, x in zip(replies, inputs, strict=True)
            ]
        if task == "entities":
            return [parse_entities(r) or () for r in replies]
        return [parse_label(r, task, labels) for r in replies]


class QwenTask:
    """One task of a shared ``Qwen``, called like an ``Encoder``."""

    def __init__(self, qwen: Qwen, task: str) -> None:
        self.qwen, self.task, self.model = qwen, task, qwen.model

    def __call__(self, inputs: list[Any]) -> list[Any]:
        return self.qwen.predict(self.task, inputs)


Predict = Encoder | QwenTask


def load_predictors(
    models: dict[str, NlpModel], qwen_config: Path | None, device: str
) -> dict[str, Predict]:
    """A predictor per task; Qwen tasks share one base model."""
    out: dict[str, Predict] = {}
    adapters = {t: m.path for t, m in models.items() if m.kind == "qwen"}
    if adapters:
        if qwen_config is None:
            raise ValueError("a Qwen task needs qwen_config")
        qwen = Qwen(qwen_config, adapters, device)
        for task in adapters:  # nlp_model_versions must name the base the adapters run on
            if models[task].base != qwen.base_model:
                raise ValueError(f"{task}: base {models[task].base} != {qwen.base_model}")
        out |= {task: QwenTask(qwen, task) for task in adapters}
    for task, m in models.items():
        if m.kind == "encoder":
            out[task] = Encoder(task, REPO_ROOT / m.path, device)
    return out


def parameters(predict: Predict) -> int:
    """Parameters a task runs on: 4-bit weights unpacked, Qwen with its own adapter only."""
    total = 0
    for name, p in predict.model.named_parameters():
        if isinstance(predict, QwenTask) and "lora_" in name and f".{predict.task}." not in name:
            continue
        total += p.numel() * (2 if type(p).__name__ == "Params4bit" else 1)  # 2 weights a byte
    return total
