"""Phase 5 training (D-32a, D-32f): encoder fine-tunes or Qwen3 4B LoRA adapters, one per task,
then validation scores with bootstrap CIs (D-70d). Rerunning resumes from the checkpoints.

    uv run python scripts/train_nlp.py --config configs/phase5/encoder.yaml [--model marbertv2]
    uv run python scripts/train_nlp.py --config configs/phase5/qwen.yaml [--task intent]

``--limit N`` trains and scores on the first N examples of each split (pre-flight).
Scores go to ``outputs/phase5/validation/<model>-<task>.json`` and MLflow (``phase5-nlp``).
A model/task whose JSON exists is skipped: delete the JSON to retrain after a config change.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from tuncall.config import load_config
from tuncall.data.lexicon import load_lexicon
from tuncall.data.testsets import REPO_ROOT, sha256_file
from tuncall.nlp.data import ROLES, call_turns
from tuncall.nlp.metrics import (
    Score,
    classification,
    entity_f1,
    parse_entities,
    parse_label,
    parse_roles,
)
from tuncall.nlp.tasks import TASKS, Example, label_set, load_task
from tuncall.tracking import Tracker

OUT = REPO_ROOT / "outputs/phase5/validation"


def role_gold(val: list[Example], seed: int | None) -> dict[tuple[str, str], str]:
    return {(ex.call.id, s): r for ex in val for s, r in call_turns(ex.call, seed)[1].items()}


def scores(task: str, val: list[Example], preds: list[Any]) -> dict[str, Score]:
    """Validation scores; role ``preds`` is one {(call, speaker): role} dict per input order."""
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


def run_encoder(cfg: Any, name: str, task: str, data: tuple, tracker: Tracker) -> dict[str, Score]:
    from tuncall.nlp.encoder import train_encoder

    labels, train, val, _ = data
    preds, _ = train_encoder(cfg, name, task, labels, train, val, tracker)
    if task == "role":  # lines are labelled one by one, so shuffling changes nothing
        preds = [(role_gold(val, None), preds[0]), (role_gold(val, None), preds[0])]
    return scores(task, val, preds)


def run_qwen(cfg: Any, task: str, data: tuple, tracker: Tracker) -> dict[str, Score]:
    from tuncall.nlp.qwen import MAX_NEW_TOKENS, chat, generate, train_qwen

    labels, train, val, shuffle_seed = data
    model, tok = train_qwen(cfg, task, labels, train, tracker)

    def replies(seed: int | None) -> list[str]:
        prompts = [chat(task, labels, ex, seed)[0] for ex in val]
        return generate(model, tok, prompts, MAX_NEW_TOKENS[task], cfg.eval_batch_size)

    if task == "role":
        preds = []
        for seed in (None, shuffle_seed):
            pred = {}
            for ex, reply in zip(val, replies(seed), strict=True):
                speakers = sorted(call_turns(ex.call, seed)[1])
                pred |= {(ex.call.id, s): r for s, r in parse_roles(reply, speakers, ROLES).items()}
            preds.append((role_gold(val, seed), pred))
    elif task == "entities":
        preds = [parse_entities(r) for r in replies(None)]
    else:
        preds = [parse_label(r, task, labels) for r in replies(None)]
    return scores(task, val, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, action="append", help="default: all four")
    parser.add_argument("--model", action="append", help="encoder names; default: all")
    parser.add_argument("--limit", type=int, help="first N examples per split (pre-flight)")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if "base_model" in raw:
        from tuncall.nlp.qwen import QwenConfig

        cfg: Any = load_config(args.config, QwenConfig)
        names = ["qwen3-4b-lora"]
    else:
        from tuncall.nlp.encoder import EncoderConfig

        cfg = load_config(args.config, EncoderConfig)
        names = args.model or list(cfg.models)
    if args.limit:
        cfg = cfg.model_copy(update={"output_root": cfg.output_root / f"limit-{args.limit}"})
    lexicon = load_lexicon(REPO_ROOT / cfg.data.lexicon)
    shuffle_seed = getattr(cfg, "shuffle_seed", 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for task in args.task or TASKS:
        train = load_task(task, "train", cfg.data, lexicon)
        val = load_task(task, "validation", cfg.data, lexicon)
        if args.limit:
            train, val = train[: args.limit], val[: args.limit]
        data = (label_set(task, train), train, val, shuffle_seed)
        suffix = f"-limit-{args.limit}" if args.limit else ""
        main_scores = {}
        for name in names:
            out_dir = cfg.output_root / name / task if "models" in raw else cfg.output_root / task
            path = OUT / f"{name}-{task}{suffix}.json"
            if path.exists():  # done earlier; its checkpoints may be gone (kept: best only)
                main_scores[name] = next(iter(json.loads(path.read_text()).values()))["value"]
                print(name, task, "done earlier:", path)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with Tracker(
                cfg.experiment,
                config=cfg.model_dump(mode="json") | {"model": name, "task": task},
                dataset=f"{task} train {len(train)} validation {len(val)} "
                f"lexicon {sha256_file(REPO_ROOT / cfg.data.lexicon)[:12]}",
                buffer_path=out_dir / "tracking-buffer.jsonl",
                run_name=f"{name}-{task}" + (f"-limit-{args.limit}" if args.limit else ""),
            ) as tracker:
                if "models" in raw:
                    result = run_encoder(cfg, name, task, data, tracker)
                else:
                    result = run_qwen(cfg, task, data, tracker)
                flat = {f"val_{m}_{k}": v for m, s in result.items() for k, v in s.items()}
                tracker.log_metrics(flat, step=0)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(name, task, json.dumps({m: round(s["value"], 4) for m, s in result.items()}))
            main_scores[name] = next(iter(result.values()))["value"]
        if "models" in raw and len(main_scores) > 1:  # disk: keep only this task's best encoder
            winner = max(main_scores, key=main_scores.__getitem__)
            for name in main_scores:
                if name != winner:
                    shutil.rmtree(cfg.output_root / name / task / "best", ignore_errors=True)
            print(task, "best encoder on validation:", winner)


if __name__ == "__main__":
    main()
