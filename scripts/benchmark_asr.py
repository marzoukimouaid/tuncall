"""Zero-shot ASR benchmark on test-general: the Phase 1 baseline table (D-15, D-15a, D-15b).

Each model's predictions are cached in ``<out>/<model>.jsonl``; a cached model is skipped
unless ``--force``. The table is built from the cached files and written into README.md only
when every configured model is cached and no ``--limit`` is set.

    uv run python scripts/benchmark_asr.py --config configs/phase1/baselines.yaml
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from tuncall.config import StrictModel, load_config
from tuncall.data.lexicon import load_lexicon, transliterate
from tuncall.data.testsets import Manifest, Utterance, load_test_set, sha256_file
from tuncall.eval.metrics import (
    bootstrap_ci,
    char_counts,
    error_rate,
    paired_delta_ci,
    word_counts,
)
from tuncall.eval.normalize import normalize_arabic
from tuncall.eval.report import replace_block
from tuncall.tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


class ModelRef(StrictModel):
    id: str
    revision: str


class BenchmarkConfig(StrictModel):
    manifest: Path
    experiment: str
    reference_model: str
    batch_size: int
    language: str
    task: str
    num_beams: int
    max_new_tokens: int
    lexicon: Path | None = None  # D-10a lexicon applied when scoring (D-16c)
    models: dict[str, ModelRef]


Row = dict[str, Any]


def score_rows(refs: list[tuple[str, str]], hyps: list[str], lexicon: dict[str, str]) -> list[Row]:
    """Score (id, reference) pairs. Both sides are transliterated with the lexicon (D-10a,
    D-16c) and normalized (D-10); the frozen reference itself is kept unchanged."""
    rows = []
    for (uid, ref), hyp in zip(refs, hyps, strict=True):
        ref_norm = normalize_arabic(transliterate(ref, lexicon))
        hyp_norm = normalize_arabic(transliterate(hyp, lexicon))
        [(we, wl)] = word_counts([ref_norm], [hyp_norm])
        [(ce, cl)] = char_counts([ref_norm], [hyp_norm])
        rows.append(
            {
                "id": uid,
                "ref": ref,
                "hyp": hyp,
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "word_errors": we,
                "word_len": wl,
                "char_errors": ce,
                "char_len": cl,
            }
        )
    return rows


def _words(rows: list[Row]) -> list[tuple[int, int]]:
    return [(r["word_errors"], r["word_len"]) for r in rows]


def _chars(rows: list[Row]) -> list[tuple[int, int]]:
    return [(r["char_errors"], r["char_len"]) for r in rows]


def is_loop(hyp_norm: str) -> bool:
    """Decoder stuck repeating (D-15d): over 20 words, under 30% of them distinct."""
    words = hyp_norm.split()
    return len(words) > 20 and len(set(words)) / len(words) < 0.3


def summarize(rows: list[Row]) -> dict[str, float]:
    w, c = _words(rows), _chars(rows)
    (wlo, whi), (clo, chi) = bootstrap_ci(w), bootstrap_ci(c)
    raw = word_counts([r["ref"] for r in rows], [r["hyp"] for r in rows])
    return {
        "wer": error_rate(w),
        "wer_ci_low": wlo,
        "wer_ci_high": whi,
        "cer": error_rate(c),
        "cer_ci_low": clo,
        "cer_ci_high": chi,
        "raw_wer": error_rate(raw),
        "looping": sum(is_loop(r["hyp_norm"]) for r in rows),
    }


def write_jsonl(path: Path, rows: list[Row]) -> None:
    """Write via ``.tmp`` + rename, so a killed run never leaves a half file that looks cached."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_run(models: list[str], out: Path, force: bool) -> list[str]:
    return [m for m in models if force or not (out / f"{m}.jsonl").exists()]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(x: float, lo: float, hi: float, sign: str = "") -> str:
    return f"{100 * x:{sign}.1f} [{100 * lo:{sign}.1f}, {100 * hi:{sign}.1f}]"


def render_table(results: dict[str, list[Row]], reference: str) -> str:
    """Markdown table sorted by WER; ΔWER is paired against ``reference`` (D-15a)."""
    ref_rows = results.get(reference)
    lines = [
        "| Model | WER % [95% CI] | CER % [95% CI] "
        f"| ΔWER vs {reference} [95% CI] | Raw WER % | Looping |",
        "|---|---|---|---|---|---|",
    ]
    summaries = {m: summarize(rows) for m, rows in results.items()}
    for model, s in sorted(summaries.items(), key=lambda kv: kv[1]["wer"]):
        delta = "—"
        if ref_rows is not None and model != reference:
            rows = results[model]
            assert [r["id"] for r in rows] == [r["id"] for r in ref_rows], "utterances differ"
            d = error_rate(_words(rows)) - error_rate(_words(ref_rows))
            delta = _ci(d, *paired_delta_ci(_words(rows), _words(ref_rows)), sign="+")
        lines.append(
            f"| {model} | {_ci(s['wer'], s['wer_ci_low'], s['wer_ci_high'])} "
            f"| {_ci(s['cer'], s['cer_ci_low'], s['cer_ci_high'])} | {delta} "
            f"| {_pct(s['raw_wer'])} | {s['looping']:.0f} |"
        )
    return "\n".join(lines)


def scoring_lexicon(cfg: BenchmarkConfig) -> tuple[dict[str, str], str]:
    """The configured lexicon and its sha256 (recorded with every score), or none."""
    if cfg.lexicon is None:
        return {}, "none"
    path = ROOT / cfg.lexicon
    return load_lexicon(path), sha256_file(path)


def rescore(rows: list[Row], lexicon: dict[str, str]) -> list[Row]:
    """Re-score cached predictions, so the table always uses the current lexicon."""
    return score_rows([(r["id"], r["ref"]) for r in rows], [r["hyp"] for r in rows], lexicon)


def readme_block(table: str, config: Path, n: int, lexicon_id: str) -> str:
    return (
        f"{table}\n\n"
        f"{n} utterances. Latin-script words in references and hypotheses are transliterated "
        f"with the D-10a lexicon (sha256 `{lexicon_id}`) before scoring (D-16c). "
        f"WER/CER after Arabic normalization (D-10), 95% bootstrap CIs over "
        "utterances (1000 resamples), ΔWER is a paired bootstrap (D-73). Raw WER is on "
        "unnormalized text. Beam 5, Arabic forced, no temperature fallback (D-26). "
        "Looping = utterances where the decoder got stuck repeating (hypothesis over 20 words, "
        "under 30% of them distinct; D-15d). They stay in every score: WER above 100% comes "
        "from the extra words they add.\n"
        f"Reproduce: `uv run python scripts/benchmark_asr.py --config {config}`"
    )


def run_model(
    key: str,
    cfg: BenchmarkConfig,
    utterances: list[Utterance],
    manifest: Manifest,
    out: Path,
    lexicon: tuple[dict[str, str], str],
) -> None:
    import torch  # heavy; only needed when a model actually runs

    from tuncall.asr.transcribe import load, transcribe

    ref = cfg.models[key]
    cuda = torch.cuda.is_available()
    model, processor = load(
        ref.id, ref.revision, "cuda" if cuda else "cpu", torch.float16 if cuda else torch.float32
    )
    hyps = transcribe(
        model,
        processor,
        [u.audio for u in utterances],
        batch_size=cfg.batch_size,
        language=cfg.language,
        task=cfg.task,
        num_beams=cfg.num_beams,
        max_new_tokens=cfg.max_new_tokens,
    )
    del model, processor
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    rows = score_rows([(u.id, u.text) for u in utterances], hyps, lexicon[0])
    path = out / f"{key}.jsonl"
    write_jsonl(path, rows)
    summary = summarize(rows)
    print(f"{key}: WER {_pct(summary['wer'])}  CER {_pct(summary['cer'])}", flush=True)
    with Tracker(
        cfg.experiment,
        config=cfg.model_dump(mode="json")
        | {
            "model": key,
            "model_id": ref.id,
            "model_revision": ref.revision,
            "utterances": len(rows),
            "lexicon_sha256": lexicon[1],
        },
        dataset=f"{manifest.repo}@{manifest.revision}",
        buffer_path=out / "tracking-buffer.jsonl",
        run_name=key,
    ) as tracker:
        tracker.log_metrics(summary, step=0)
        tracker.log_artifact(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", help="comma-separated subset of the configured models")
    parser.add_argument("--limit", type=int, help="first N utterances only (pre-flight)")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--force", action="store_true", help="rerun cached models")
    args = parser.parse_args()

    cfg = load_config(args.config, BenchmarkConfig)
    selected = args.models.split(",") if args.models else list(cfg.models)
    if unknown := set(selected) - set(cfg.models):
        parser.error(f"unknown models: {sorted(unknown)}")
    out: Path = args.out / f"limit-{args.limit}" if args.limit else args.out
    out.mkdir(parents=True, exist_ok=True)

    lexicon = scoring_lexicon(cfg)
    if todo := to_run(selected, out, args.force):
        manifest = load_config(ROOT / cfg.manifest, Manifest)
        utterances = load_test_set(manifest)[: args.limit]
        for key in todo:
            run_model(key, cfg, utterances, manifest, out, lexicon)

    cached = [m for m in cfg.models if (out / f"{m}.jsonl").exists()]
    results = {m: rescore(read_jsonl(out / f"{m}.jsonl"), lexicon[0]) for m in cached}
    table = render_table(results, cfg.reference_model)
    print(table)
    if args.limit or len(results) < len(cfg.models):
        print(f"README not updated: {len(cached)}/{len(cfg.models)} cached, limit={args.limit}")
        return
    n = len(next(iter(results.values())))
    config = args.config.resolve().relative_to(ROOT)
    block = readme_block(table, config, n, lexicon[1])
    text = replace_block(README.read_text(encoding="utf-8"), "baselines", block)
    README.write_text(text, encoding="utf-8")
    print(f"updated {README}")


if __name__ == "__main__":
    main()
