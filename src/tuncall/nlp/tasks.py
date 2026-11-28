"""The four Phase 5 tasks as (input, gold) examples, shared by the encoder and the Qwen LoRA.

Intent and entities come from SLURP-TN, sentiment from TSAC (validation and test keep only
comments in Arabic script, the scored ones, D-31a), roles from the synthetic calls.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tuncall.config import StrictModel
from tuncall.data.testsets import REPO_ROOT
from tuncall.nlp.data import CallSet, call_turns, slurp_tn, synthetic_calls, tsac
from tuncall.nlp.metrics import Score, classification, entity_f1

TASKS = ("intent", "entities", "sentiment", "role")


class DataRef(StrictModel):
    """Where the task data lives; shared by the Phase 5 training configs."""

    lexicon: Path
    slurp_revision: str
    calls: Path  # configs/data/synthetic-calls.yaml


@dataclass(frozen=True)
class Example:
    id: str
    text: str  # role: unused, the call is in ``call``
    gold: Any  # intent/sentiment: label; entities: ((type, value), ...); role: {speaker: role}
    call: Any = None  # role: the nlp.data.Call


def load_task(
    task: str, split: str, ref: DataRef, lexicon: dict[str, str], root: Path | None = None
) -> list[Example]:
    if task in ("intent", "entities"):
        utts = slurp_tn(split, ref.slurp_revision, lexicon, root)
        if task == "intent":
            return [Example(u.id, u.text, u.intent) for u in utts]
        return [Example(u.id, u.text, u.entities) for u in utts]
    if task == "sentiment":
        comments = tsac(split, lexicon, root)
        return [
            Example(c.id, c.text, c.sentiment) for c in comments if split == "train" or c.arabic
        ]
    if task == "role":
        manifest = CallSet.model_validate(
            yaml.safe_load((REPO_ROOT / ref.calls).read_text(encoding="utf-8"))
        )
        calls = synthetic_calls(split, manifest, lexicon)
        return [Example(c.id, "", None, c) for c in calls]
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def label_set(task: str, train: list[Example]) -> list[str]:
    """Sorted labels seen in train: intents, sentiments, entity types or roles."""
    if task == "entities":
        return sorted({t for ex in train for t, _ in ex.gold})
    if task == "role":
        return sorted({ln.role for ex in train for ln in ex.call.lines})
    return sorted({ex.gold for ex in train})


def role_gold(val: list[Example], seed: int | None) -> dict[tuple[str, str], str]:
    return {(ex.call.id, s): r for ex in val for s, r in call_turns(ex.call, seed)[1].items()}


def scores(task: str, val: list[Example], preds: list[Any]) -> dict[str, Score]:
    """Scores on any split; role ``preds`` is one (gold, pred) pair of {(call, speaker): role}
    dicts per input order (plain, then shuffled)."""
    if task == "entities":
        return {"f1": entity_f1([ex.gold for ex in val], preds)}
    if task == "role":
        out = {}
        for name, (gold, pred) in zip(("accuracy", "accuracy_shuffled"), preds, strict=True):
            keys = sorted(gold)
            out[name] = classification([gold[k] for k in keys], [pred.get(k) for k in keys])[
                "accuracy"
            ]
        return out
    return classification([ex.gold for ex in val], preds)
