"""Phase 5 benchmark (D-70d, D-55a): the kept encoder per task vs the Qwen3 4B LoRA.

    uv run python scripts/benchmark_nlp.py --set dev   # validation: speed + the shipping choice
    uv run python scripts/benchmark_nlp.py --set test  # once: writes the README phase5 block

Scores with 95% bootstrap CIs (D-73) on the scored unit; intent and entities on test are also
scored on our LoRA ASR's transcripts of the same audio (the gap = the cost of ASR errors).
Milliseconds per input are measured on this machine after a warm-up (role: per call).
``--contender`` runs one contender; its results are merged into the set's JSON
(``outputs/phase5/benchmark-<set>.json``). ``--limit N`` is the pre-flight.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from tuncall.asr.cache import target_text
from tuncall.config import StrictModel, load_config
from tuncall.data.lexicon import load_lexicon
from tuncall.data.testsets import REPO_ROOT, sha256_file
from tuncall.eval.report import replace_block
from tuncall.nlp.data import call_turns
from tuncall.nlp.predict import NlpModel, Predict, load_predictors, parameters
from tuncall.nlp.tasks import TASKS, DataRef, Example, load_task, role_gold, scores
from tuncall.tracking import Tracker

SPLIT = {"dev": "validation", "test": "test"}
OUT = REPO_ROOT / "outputs/phase5"
README = REPO_ROOT / "README.md"
MAIN = {
    "intent": "macro_f1",
    "entities": "f1",
    "sentiment": "macro_f1",
    "role": "accuracy_shuffled",
}


class BenchmarkConfig(StrictModel):
    data: DataRef
    shuffle_seed: int
    asr_transcripts: Path  # benchmark_asr.py output for SLURP-TN test (``id``, ``hyp``)
    qwen_config: Path
    contenders: dict[str, dict[str, NlpModel]]  # name -> task -> model


def timed(predict: Predict, inputs: list[Any]) -> tuple[list[Any], float]:
    """Predictions and milliseconds per input, after a warm-up batch."""
    predict(inputs[:2])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    preds = predict(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return preds, (time.perf_counter() - t0) * 1000 / len(inputs)


def evaluate(
    predict: Predict,
    task: str,
    examples: list[Example],
    seed: int,
    asr: dict[str, str] | None,
) -> dict[str, Any]:
    if task == "role":
        preds, ms = [], 0.0
        for s in (None, seed):
            turns = [call_turns(ex.call, s)[0] for ex in examples]
            if s is None:
                out, ms = timed(predict, turns)
            else:
                out = predict(turns)
            pred = {
                (ex.call.id, k): r
                for ex, d in zip(examples, out, strict=True)
                for k, r in d.items()
            }
            preds.append((role_gold(examples, s), pred))
        return {"n": len(examples), "ms_per_input": ms, "scores": scores(task, examples, preds)}
    preds, ms = timed(predict, [ex.text for ex in examples])
    result = {"n": len(examples), "ms_per_input": ms, "scores": scores(task, examples, preds)}
    if asr is not None and task in ("intent", "entities"):
        missing = [ex.id for ex in examples if ex.id not in asr]
        if missing:
            raise ValueError(f"no ASR transcript for {len(missing)} utterances, e.g. {missing[:3]}")
        asr_ex = [Example(ex.id, asr[ex.id], ex.gold) for ex in examples]
        result["scores_asr"] = scores(task, asr_ex, predict([ex.text for ex in asr_ex]))
    return result


def shipping(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    """D-55a: per task the best main score; within 1 point, the faster contender."""
    choice = {}
    for task in TASKS:
        rows = {name: r[task] for name, r in results.items() if task in r}
        best = max(rows, key=lambda n: rows[n]["scores"][MAIN[task]]["value"])
        top = rows[best]["scores"][MAIN[task]]["value"]
        close = [n for n in rows if top - rows[n]["scores"][MAIN[task]]["value"] <= 0.01]
        choice[task] = min(close, key=lambda n: rows[n]["ms_per_input"])
    return choice


def cell(score: dict[str, float]) -> str:
    return f"{100 * score['value']:.1f} [{100 * score['lo']:.1f}, {100 * score['hi']:.1f}]"


def tables(results: dict[str, dict[str, Any]]) -> str:
    """The quality and speed tables in Markdown."""
    lines = [
        "| Model | Input | Intent macro-F1 % | Intent accuracy % | Entity F1 % |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        for key, label in (("scores", "gold text"), ("scores_asr", "ASR transcript")):
            if key in r["intent"]:
                i, e = r["intent"][key], r["entities"][key]
                lines.append(
                    f"| {name} | {label} | {cell(i['macro_f1'])} | {cell(i['accuracy'])} "
                    f"| {cell(e['f1'])} |"
                )
    lines += [
        "",
        "| Model | Sentiment macro-F1 % | Role accuracy % | Role accuracy, lines shuffled % |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        s, ro = r["sentiment"]["scores"], r["role"]["scores"]
        lines.append(
            f"| {name} | {cell(s['macro_f1'])} | {cell(ro['accuracy'])} "
            f"| {cell(ro['accuracy_shuffled'])} |"
        )
    lines += [
        "",
        "| Model | ms per input: intent / entities / sentiment / role (per call) | Parameters "
        "(M) | Cost |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        ms = " / ".join(f"{r[t]['ms_per_input']:.1f}" for t in TASKS)
        params = sorted({round(r[t]["parameters"] / 1e6) for t in TASKS})
        lines.append(f"| {name} | {ms} | {' / '.join(map(str, params))} | $0 (local) |")
    return "\n".join(lines)


def readme_block(results: dict[str, dict[str, Any]], cfg: BenchmarkConfig, config: Path) -> str:
    r = next(iter(results.values()))
    n = {t: r[t]["n"] for t in TASKS}
    asr = REPO_ROOT / cfg.asr_transcripts
    return f"""{tables(results)}

Test sets: SLURP-TN test (frozen test-domain, {n["intent"]} utterances) for intent and entities,
TSAC test comments written in Arabic script ({n["sentiment"]}, D-31a), and {n["role"]} frozen
synthetic support calls for roles (D-31a). Every model was trained on gold text only. ASR
transcript = our LoRA ASR's output on the same SLURP-TN test audio
(`{cfg.asr_transcripts}`, sha256 `{sha256_file(asr)[:12]}`). All text follows the D-34a
convention (D-10a lexicon, harakat stripped). Entity F1 is over (type, value) pairs after
Arabic normalization, positions ignored. Role accuracy is per speaker; shuffled = the call's
lines in a fixed random order, so "the first speaker is the agent" earns nothing (D-70d).
95% bootstrap CIs over the scored unit (D-73). Some test intents have only 1 to 3 examples, so
intent macro-F1 is noisy. Sentiment may be inflated by near-duplicate TSAC comments (only
exact duplicates are removed). The synthetic calls were written by an LLM (D-31c), and roles
in them are easy. Speed: laptop RTX 4070, encoders in bf16, Qwen in 4-bit with greedy
decoding; parameters per task model (encoders differ by task). Labels are SLURP-TN's
voice-assistant intents, not call-centre intents (D-55a).
Reproduce: `uv run python scripts/benchmark_nlp.py --config {config} --set test`"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5/benchmark.yaml"))
    parser.add_argument("--set", choices=SPLIT, required=True)
    parser.add_argument("--contender", action="append", help="default: all")
    parser.add_argument("--limit", type=int, help="first N inputs per task (pre-flight)")
    args = parser.parse_args()
    cfg = load_config(args.config, BenchmarkConfig)
    lexicon = load_lexicon(REPO_ROOT / cfg.data.lexicon)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {t: load_task(t, SPLIT[args.set], cfg.data, lexicon) for t in TASKS}
    if args.limit:
        data = {t: d[: args.limit] for t, d in data.items()}
    asr = None
    if args.set == "test":
        rows = (REPO_ROOT / cfg.asr_transcripts).read_text(encoding="utf-8").splitlines()
        asr = {r["id"]: target_text(r["hyp"], lexicon) for r in map(json.loads, rows)}

    suffix = f"-limit-{args.limit}" if args.limit else ""
    path = OUT / f"benchmark-{args.set}{suffix}.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.contender or list(cfg.contenders):
        models = cfg.contenders[name]
        predictors = load_predictors(models, cfg.qwen_config, device)
        with Tracker(
            "phase5-nlp",
            config=cfg.model_dump(mode="json") | {"contender": name, "set": args.set},
            dataset=" ".join(f"{t} {len(d)}" for t, d in data.items()) + f" {SPLIT[args.set]}",
            buffer_path=OUT / "tracking-buffer.jsonl",
            run_name=f"benchmark-{args.set}-{name}{suffix}",
        ) as tracker:
            results[name] = {}
            for task in TASKS:
                r = evaluate(predictors[task], task, data[task], cfg.shuffle_seed, asr)
                r |= {"parameters": parameters(predictors[task]), "model": models[task].version}
                results[name][task] = r
                print(name, task, {m: round(s["value"], 4) for m, s in r["scores"].items()},
                      f"{r['ms_per_input']:.1f} ms")  # fmt: skip
                tracker.log_metrics(
                    {f"{task}_{m}": s["value"] for m, s in r["scores"].items()}
                    | {f"{task}_ms_per_input": r["ms_per_input"]},
                    step=0,
                )
        del predictors
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(tables(results))
    if args.set == "dev":
        print("shipping choice (D-55a):", shipping(results))
    elif not args.limit and set(results) == set(cfg.contenders):
        block = readme_block(results, cfg, args.config)
        README.write_text(replace_block(README.read_text(), "phase5", block), encoding="utf-8")
        print("README phase5 block updated")


if __name__ == "__main__":
    main()
