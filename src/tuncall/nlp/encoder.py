"""Encoder fine-tunes, one per task (D-32a, D-32f): MARBERTv2 and CAMeLBERT-DA.

- intent, sentiment: sequence classification.
- entities: BIO token tags built from the gold (type, value) pairs, each value found as a
  word sequence in the text; predicted spans are turned back into (type, value) pairs.
- role: each line is labelled agent/customer, and a speaker gets the role its lines lean to
  (mean probability), so line order never matters.

The HF Trainer keeps the best epoch on validation (``score`` = the task's main metric) and
resumes from the latest checkpoint in ``output_dir`` unless checkpoints are model-only
(``save_only_model``, to save disk), in which case a stopped task starts over.
"""

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from tuncall.asr.train import IntervalsFromConfig, ModelRef, TrackerCallback
from tuncall.config import StrictModel
from tuncall.nlp.data import call_turns
from tuncall.nlp.metrics import accuracy, entity_counts, macro_f1
from tuncall.nlp.tasks import DataRef, Example
from tuncall.tracking import Tracker


class EncoderConfig(StrictModel):
    experiment: str
    data: DataRef
    models: dict[str, ModelRef]  # name -> pinned Hub model
    max_length: int
    output_root: Path  # checkpoints go to output_root/<model>/<task>
    trainer: dict[str, Any]  # TrainingArguments keyword arguments


def bio_tags(words: list[str], pairs: tuple[tuple[str, str], ...]) -> tuple[list[str], int]:
    """BIO tag per word, and how many gold values were not found as words in the text."""
    tags, missed = ["O"] * len(words), 0
    for etype, value in pairs:
        span = value.split()
        n = len(span)
        starts = [
            i
            for i in range(len(words) - n + 1)
            if n and words[i : i + n] == span and all(t == "O" for t in tags[i : i + n])
        ]
        if not starts:
            missed += 1
            continue
        i = starts[0]
        tags[i : i + n] = [f"B-{etype}"] + [f"I-{etype}"] * (n - 1)
    return tags, missed


def spans(words: list[str], tags: list[str]) -> tuple[tuple[str, str], ...]:
    """(type, value) pairs from BIO tags; a stray I- starts a new span."""
    out: list[tuple[str, list[str]]] = []
    prev = "O"
    for word, tag in zip(words, tags, strict=True):
        if tag != "O":
            kind, etype = tag.split("-", 1)
            if kind == "I" and prev in (f"B-{etype}", f"I-{etype}"):
                out[-1][1].append(word)
            else:
                out.append((etype, [word]))
        prev = tag
    return tuple((t, " ".join(ws)) for t, ws in out)


def _role_lines(examples: list[Example]) -> list[tuple[str, str, str, str]]:
    """(call id, speaker, text, role) for every line, in the calls' written order."""
    rows = []
    for ex in examples:
        turns, roles = call_turns(ex.call)
        rows += [(ex.call.id, s, text, roles[s]) for s, text in turns]
    return rows


def vote(
    rows: list[tuple[str, str, str, str]], agent_prob: np.ndarray
) -> dict[tuple[str, str], str]:
    """Role per (call, speaker): agent if the mean agent probability of its lines is over 0.5."""
    probs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (call, spk, _, _), p in zip(rows, agent_prob, strict=True):
        probs[(call, spk)].append(float(p))
    return {k: "agent" if np.mean(v) > 0.5 else "customer" for k, v in probs.items()}


class EncoderTask:
    """One task's data, model head and scoring for one encoder."""

    def __init__(self, task: str, labels: list[str], tokenizer: Any, max_length: int) -> None:
        self.task, self.tok, self.max_length = task, tokenizer, max_length
        self.labels = (
            ["O"] + [f"{p}-{t}" for t in labels for p in ("B", "I")]
            if task == "entities"
            else labels
        )
        self.index = {lab: i for i, lab in enumerate(self.labels)}

    def encode(self, examples: list[Example]) -> tuple[list[dict[str, Any]], Any]:
        """Model inputs and what scoring needs to map predictions back."""
        if self.task == "entities":
            rows, words_all = [], []
            for ex in examples:
                words = ex.text.split()
                tags, _ = bio_tags(words, ex.gold)
                enc = self.tok(
                    words, is_split_into_words=True, truncation=True, max_length=self.max_length
                )
                seen, labels = [], []
                for w in enc.word_ids():  # label each word's first sub-token only
                    if w is None or w in seen:
                        labels.append(-100)
                    else:
                        labels.append(self.index.get(tags[w], 0))
                        seen.append(w)
                rows.append(dict(enc) | {"labels": labels})
                # labelled positions -> words (a word the tokenizer drops has no position)
                words_all.append([words[w] for w in seen])
            return rows, words_all
        if self.task == "role":
            lines = _role_lines(examples)
            texts, golds = [t for _, _, t, _ in lines], [r for *_, r in lines]
            meta = lines
        else:
            texts, golds, meta = [ex.text for ex in examples], [ex.gold for ex in examples], None
        enc = self.tok(texts, truncation=True, max_length=self.max_length)
        rows = [
            {k: enc[k][i] for k in enc} | {"labels": self.index.get(g, 0)}
            for i, g in enumerate(golds)
        ]
        return rows, meta

    def model(self, ref: ModelRef) -> Any:
        cls = (
            AutoModelForTokenClassification
            if self.task == "entities"
            else AutoModelForSequenceClassification
        )
        return cls.from_pretrained(
            ref.id,
            revision=ref.revision,
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id=self.index,
        )

    def predict(self, logits: np.ndarray, label_ids: np.ndarray, meta: Any) -> list[Any]:
        """Per scored unit: label, (type, value) pairs, or role per (call, speaker)."""
        if self.task == "entities":
            out = []
            for row_logits, row_labels, words in zip(logits, label_ids, meta, strict=True):
                ids = row_logits.argmax(-1)[row_labels != -100]
                tags = [self.labels[i] for i in ids]
                out.append(spans(words, tags))
            return out
        if self.task == "role":
            p = torch.softmax(torch.from_numpy(logits).float(), -1)[:, self.index["agent"]].numpy()
            return [vote(meta, p)]
        return [self.labels[i] for i in logits.argmax(-1)]

    def score(self, examples: list[Example], preds: list[Any]) -> float:
        """The task's main metric: macro-F1, pair F1, or role accuracy over speakers."""
        if self.task == "entities":
            counts = [entity_counts(ex.gold, p) for ex, p in zip(examples, preds, strict=True)]
            tp, n_pred, n_gold = (sum(c[k] for c in counts) for k in range(3))
            return 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0
        if self.task == "role":
            gold = {
                (ex.call.id, s): r for ex in examples for s, r in call_turns(ex.call)[1].items()
            }
            keys = sorted(gold)
            return accuracy([gold[k] for k in keys], [preds[0].get(k) for k in keys])
        return macro_f1([ex.gold for ex in examples], preds)


def train_encoder(
    cfg: EncoderConfig,
    name: str,
    task: str,
    labels: list[str],
    train: list[Example],
    val: list[Example],
    tracker: Tracker,
) -> tuple[list[Any], float]:
    """Fine-tune (resuming if a checkpoint exists); validation predictions and main score."""
    ref = cfg.models[name]
    tok = AutoTokenizer.from_pretrained(ref.id, revision=ref.revision)
    et = EncoderTask(task, labels, tok, cfg.max_length)
    train_rows, _ = et.encode(train)
    val_rows, meta = et.encode(val)

    def compute(pred: Any) -> dict[str, float]:
        return {"score": et.score(val, et.predict(pred.predictions, pred.label_ids, meta))}

    collator = (
        DataCollatorForTokenClassification(tok)
        if task == "entities"
        else DataCollatorWithPadding(tok)
    )
    out = cfg.output_root / name / task
    args = TrainingArguments(output_dir=str(out), report_to="none", **cfg.trainer)
    trainer = Trainer(
        model=et.model(ref),
        args=args,
        train_dataset=train_rows,  # pyright: ignore[reportArgumentType]
        eval_dataset=val_rows,  # pyright: ignore[reportArgumentType]
        data_collator=collator,
        compute_metrics=compute,
        callbacks=[IntervalsFromConfig(), TrackerCallback(tracker)],
    )
    # model-only checkpoints cannot resume training: a stopped task starts over
    resume = not args.save_only_model and any(out.glob("checkpoint-*"))
    for ckpt in [] if resume else out.glob("checkpoint-*"):
        shutil.rmtree(ckpt)
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))
    for ckpt in out.glob("checkpoint-*"):  # ~2 GB each with optimizer state; run is finished
        shutil.rmtree(ckpt)
    result = trainer.predict(val_rows)  # pyright: ignore[reportArgumentType]
    preds = et.predict(np.asarray(result.predictions), np.asarray(result.label_ids), meta)
    return preds, et.score(val, preds)
