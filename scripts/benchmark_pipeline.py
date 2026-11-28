"""Phase 4 pipeline benchmark on the simulated conversations (D-70b, D-13b, D-13c).

For each channel (clean, telephone) and speaker count (known = 2, estimated), pyannote
diarizes every conversation once; every ASR model then transcribes those same turns. The
oracle row is the oracle model on the true turns, which separates ASR from diarization
errors. Metrics: DER (pyannote.metrics, 0.25 s collar, against the placed clips), cpWER
(best speaker matching) and plain WER/CER over the whole conversation, all with bootstrap
95% CIs over conversations (D-73). Everything is cached under ``<out>/<set>/``; a cached
step is skipped unless ``--force``. Tune on ``--set dev`` only; the frozen test set is run
once, and only a complete, unlimited test run writes the README block.

    uv run python scripts/benchmark_pipeline.py --config configs/phase4/benchmark.yaml --set dev
"""

import argparse
import gc
import itertools
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jiwer
from pydantic import Field

from tuncall.asr import ModelRef
from tuncall.config import StrictModel, load_config
from tuncall.data.conversations import Conversation, ConversationSet, load_conversations
from tuncall.data.lexicon import load_lexicon, transliterate
from tuncall.data.telephone import telephone
from tuncall.data.testsets import SAMPLE_RATE, sha256_file
from tuncall.diarize import Diarization, Span
from tuncall.eval.metrics import Counts, bootstrap_ci, char_counts, error_rate, word_counts
from tuncall.eval.normalize import normalize_arabic
from tuncall.eval.report import replace_block
from tuncall.record import CallRecord
from tuncall.tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CHANNELS = ("clean", "telephone")
SPEAKERS: dict[str, int | None] = {"known": 2, "estimated": None}
COLLAR = 0.25


class BenchmarkConfig(StrictModel):
    sets: dict[str, Path]  # name -> ConversationSet manifest
    diarization_revision: str
    lexicon: Path
    experiment: str
    readme_block: str
    readme_set: str  # the frozen set whose numbers go in the README
    oracle_model: str
    batch_size: int  # speed only: a turn's text never depends on its batch
    batch_size_by_model: dict[str, int] = Field(default_factory=dict)  # large-v3 on 8 GB
    language: str
    task: str
    num_beams: int
    max_new_tokens: int
    models: dict[str, ModelRef]


# --- scoring ----------------------------------------------------------------------------


def norm(text: str, lexicon: dict[str, str]) -> str:
    return normalize_arabic(transliterate(text, lexicon))


def _pair(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, reference words); an empty reference makes every hypothesis word an insertion."""
    if not ref.split():
        return len(hyp.split()), 0
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, len(ref.split())


def by_speaker(spans: list[tuple[str, str]]) -> list[str]:
    """Each speaker's words in time order, given (speaker, text) in time order."""
    texts: dict[str, list[str]] = {}
    for speaker, text in spans:
        texts.setdefault(speaker, []).append(text)
    return [" ".join(t) for t in texts.values()]


def cp_counts(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """cpWER counts: the speaker matching with the fewest errors (unmatched side = empty).
    shortcut: brute force over permutations; fine up to ~7 estimated speakers."""
    n = max(len(ref), len(hyp))
    ref, hyp = ref + [""] * (n - len(ref)), hyp + [""] * (n - len(hyp))
    best = min(
        sum(_pair(r, h)[0] for r, h in zip(ref, perm, strict=True))
        for perm in itertools.permutations(hyp)
    )
    return best, sum(len(r.split()) for r in ref)


def der_counts(conv: Conversation, regions: list[Span]) -> tuple[int, int]:
    """(confusion + missed + false alarm, reference speech), in ms, over the whole conversation."""
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref, hyp = Annotation(), Annotation()
    for i, c in enumerate(conv.clips):
        ref[Segment(c.start, c.end), i] = c.speaker
    for i, r in enumerate(regions):
        hyp[Segment(r.start, r.end), i] = r.speaker
    uem = Timeline([Segment(0, len(conv.audio) / SAMPLE_RATE)])
    d: Any = DiarizationErrorRate(collar=COLLAR)(ref, hyp, uem=uem, detailed=True)
    errors = d["confusion"] + d["missed detection"] + d["false alarm"]
    return round(errors * 1000), round(d["total"] * 1000)


def score(
    convs: list[Conversation],
    records: list[CallRecord],
    regions: list[list[Span]] | None,
    lexicon: dict[str, str],
) -> dict[str, Counts]:
    """Per-conversation counts for each metric; ``regions`` None = oracle (no DER)."""
    refs = [norm(" ".join(t.text for t in c.turns), lexicon) for c in convs]
    hyps = [" ".join(s.text_normalized for s in r.segments) for r in records]
    counts: dict[str, Counts] = {
        "cpwer": [
            cp_counts(
                [norm(t, lexicon) for t in by_speaker([(t.speaker, t.text) for t in c.turns])],
                by_speaker([(s.speaker, s.text_normalized) for s in r.segments]),
            )
            for c, r in zip(convs, records, strict=True)
        ],
        "wer": word_counts(refs, hyps),
        "cer": char_counts(refs, hyps),
    }
    if regions is not None:
        counts["der"] = [der_counts(c, r) for c, r in zip(convs, regions, strict=True)]
    return counts


def summarize(counts: dict[str, Counts]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric, c in counts.items():
        lo, hi = bootstrap_ci(c)
        out |= {metric: error_rate(c), f"{metric}_ci_low": lo, f"{metric}_ci_high": hi}
    return out


# --- running ----------------------------------------------------------------------------


def read_records(path: Path) -> list[CallRecord]:
    return [CallRecord.model_validate_json(line) for line in path.read_text().splitlines()]


def key_path(path: Path) -> Path:
    return path.with_suffix(".key.json")


def cached(path: Path, key: dict[str, Any]) -> bool:
    """A cache file counts only if it was written with the same inputs (``key``)."""
    kp = key_path(path)
    return path.exists() and kp.exists() and json.loads(kp.read_text()) == key


def write_atomic(path: Path, lines: list[str], key: dict[str, Any]) -> None:
    """``.tmp`` + rename, so a killed run never leaves a half file that looks cached.
    The key is written last: a crash in between leaves a file that does not count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key_path(path).unlink(missing_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    tmp.replace(path)
    key_path(path).write_text(json.dumps(key, sort_keys=True))


def diarization_key(cfg: "BenchmarkConfig", spk: str) -> dict[str, Any]:
    return {"revision": cfg.diarization_revision, "num_speakers": SPEAKERS[spk]}


def asr_key(cfg: "BenchmarkConfig", key: str, turns_file: Path | None) -> dict[str, Any]:
    """Model, decoding settings and the exact diarization the turns came from."""
    decoding = ("language", "task", "num_beams", "max_new_tokens")
    return {
        "model": cfg.models[key].model_dump(mode="json"),
        "decoding": {k: getattr(cfg, k) for k in decoding},
        "turns": sha256_file(turns_file) if turns_file else "oracle",
    }


def read_diarization(path: Path) -> list[Diarization]:
    return [
        Diarization([Span(**s) for s in d["regions"]], [Span(**s) for s in d["turns"]])
        for d in map(json.loads, path.read_text().splitlines())
    ]


def run_diarization(
    cfg: BenchmarkConfig, audio: dict[str, list[Any]], out: Path, force: bool
) -> None:
    todo = [
        (ch, spk)
        for ch in CHANNELS
        for spk in SPEAKERS
        if force or not cached(out / ch / f"diarization-{spk}.jsonl", diarization_key(cfg, spk))
    ]
    if not todo:
        return
    import torch

    from tuncall.diarize import diarize
    from tuncall.diarize import load as load_diarizer

    pipeline = load_diarizer(
        cfg.diarization_revision, "cuda" if torch.cuda.is_available() else "cpu"
    )
    for ch, spk in todo:
        lines = [json.dumps(asdict(diarize(pipeline, a, SPEAKERS[spk]))) for a in audio[ch]]
        write_atomic(out / ch / f"diarization-{spk}.jsonl", lines, diarization_key(cfg, spk))
        print(f"diarized {ch}/{spk}", flush=True)
    del pipeline
    gc.collect()


def run_asr(
    cfg: BenchmarkConfig,
    key: str,
    convs: list[Conversation],
    audio: dict[str, list[Any]],
    out: Path,
    force: bool,
) -> None:
    """Transcribe every condition this model is missing (oracle turns for the oracle model)."""
    conditions = [(ch, spk) for ch in CHANNELS for spk in SPEAKERS]
    if key == cfg.oracle_model:
        conditions += [(ch, "oracle") for ch in CHANNELS]

    def turns_file(ch: str, spk: str) -> Path | None:
        return None if spk == "oracle" else out / ch / f"diarization-{spk}.jsonl"

    def done(ch: str, spk: str) -> bool:
        return cached(out / ch / spk / f"{key}.jsonl", asr_key(cfg, key, turns_file(ch, spk)))

    todo = [(ch, spk) for ch, spk in conditions if force or not done(ch, spk)]
    if not todo:
        return
    import torch

    from tuncall.asr.transcribe import load
    from tuncall.pipeline import Models, PipelineConfig, build_record, timed, transcribe_turns

    cuda = torch.cuda.is_available()
    ref = cfg.models[key]
    whisper, processor = load(
        ref.id,
        ref.revision,
        "cuda" if cuda else "cpu",
        torch.float16 if cuda else torch.float32,
        ROOT / ref.adapter if ref.adapter else None,
    )
    pcfg = PipelineConfig(
        asr=ref,
        diarization_revision=cfg.diarization_revision,
        lexicon=cfg.lexicon,
        batch_size=cfg.batch_size_by_model.get(key, cfg.batch_size),
        language=cfg.language,
        task=cfg.task,
        num_beams=cfg.num_beams,
        max_new_tokens=cfg.max_new_tokens,
    )
    models = Models(pcfg, None, whisper, processor, load_lexicon(ROOT / cfg.lexicon))
    for ch, spk in todo:
        if spk == "oracle":
            turns = [[Span(t.start, t.end, t.speaker) for t in c.turns] for c in convs]
        else:
            turns = [d.turns for d in read_diarization(out / ch / f"diarization-{spk}.jsonl")]
        lines = []
        for c, a, ts in zip(convs, audio[ch], turns, strict=True):
            texts, asr_ms = timed(lambda a=a, ts=ts, m=models: transcribe_turns(m, a, ts))
            record = build_record(
                models,
                ts,
                texts,
                call_id=c.id,
                source=f"{c.id}/{ch}",
                duration_s=len(a) / SAMPLE_RATE,
                channels=1,
                processing_ms={"asr": asr_ms},
            )
            lines.append(record.model_dump_json())
        write_atomic(out / ch / spk / f"{key}.jsonl", lines, asr_key(cfg, key, turns_file(ch, spk)))
        print(f"transcribed {ch}/{spk}/{key}", flush=True)
    del whisper, processor, models
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


# --- report -----------------------------------------------------------------------------

Row = tuple[str, str, dict[str, float]]  # (system, speaker count, summary)


def _ci(s: dict[str, float], m: str) -> str:
    if m not in s:
        return "—"
    return f"{100 * s[m]:.1f} [{100 * s[m + '_ci_low']:.1f}, {100 * s[m + '_ci_high']:.1f}]"


def table(rows: list[Row]) -> str:
    lines = [
        "| System | Speakers | DER % [95% CI] | cpWER % [95% CI] "
        "| WER % [95% CI] | CER % [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {system} | {spk} | {_ci(s, 'der')} | {_ci(s, 'cpwer')} | {_ci(s, 'wer')} "
        f"| {_ci(s, 'cer')} |"
        for system, spk, s in rows
    ]
    return "\n".join(lines)


def readme_block(tables: dict[str, list[Row]], cfg: BenchmarkConfig, n: int, hours: float) -> str:
    parts = [
        f"**{ch}** ({n} simulated conversations, {hours} h, "
        f"`{cfg.sets[cfg.readme_set]}`{', telephone channel D-25a' if ch == 'telephone' else ''})"
        f"\n\n{table(rows)}\n"
        for ch, rows in tables.items()
    ]
    return (
        "\n".join(parts)
        + "\nSimulated two-speaker conversations stitched from held-out TEDxTN talks (D-13b): "
        "expect better numbers than on real calls. Diarization: pyannote "
        f"`speaker-diarization-community-1` @ `{cfg.diarization_revision[:7]}`, off the shelf; "
        "every ASR row transcribes the same turns. Speakers: known = told there are 2, "
        "estimated = pyannote picks the count. Oracle = "
        f"{cfg.oracle_model} on the true turns (no diarization errors). DER with a "
        f"{COLLAR} s collar, scored against the placed clips; cpWER = WER with each speaker's "
        "words matched to the best reference speaker; WER/CER ignore speakers. Lexicon + "
        "Arabic normalization as the ASR tables (D-16c); 95% bootstrap CIs over conversations "
        "(D-73).\n"
        "Reproduce: `uv run python scripts/benchmark_pipeline.py --config "
        "configs/phase4/benchmark.yaml --set test`"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4/benchmark.yaml"))
    parser.add_argument("--set", required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--models", help="comma-separated subset of the config's models")
    parser.add_argument("--limit", type=int, help="first N conversations only (pre-flight)")
    parser.add_argument("--force", action="store_true", help="redo cached steps")
    args = parser.parse_args()

    cfg = load_config(args.config, BenchmarkConfig)
    if args.set not in cfg.sets:
        parser.error(f"unknown set {args.set}; config has {sorted(cfg.sets)}")
    selected = args.models.split(",") if args.models else list(cfg.models)
    if unknown := set(selected) - set(cfg.models):
        parser.error(f"unknown models: {sorted(unknown)}")
    out = args.out / args.set / (f"limit-{args.limit}" if args.limit else "")
    lexicon = load_lexicon(ROOT / cfg.lexicon)

    manifest = load_config(ROOT / cfg.sets[args.set], ConversationSet)
    convs = load_conversations(manifest)[: args.limit]
    audio: dict[str, list[Any]] = {"clean": [c.audio for c in convs]}
    audio["telephone"] = [telephone(a) for a in audio["clean"]]

    run_diarization(cfg, audio, out, args.force)
    for key in selected:
        run_asr(cfg, key, convs, audio, out, args.force)

    tables: dict[str, list[Row]] = {}
    metrics: dict[str, float] = {}
    for ch in CHANNELS:
        rows: list[Row] = []
        for spk in SPEAKERS:
            regions = [d.regions for d in read_diarization(out / ch / f"diarization-{spk}.jsonl")]
            for key in cfg.models:
                path = out / ch / spk / f"{key}.jsonl"
                if path.exists():
                    rows.append(
                        (key, spk, summarize(score(convs, read_records(path), regions, lexicon)))
                    )
        oracle = out / ch / "oracle" / f"{cfg.oracle_model}.jsonl"
        if oracle.exists():
            s = summarize(score(convs, read_records(oracle), None, lexicon))
            rows.append((f"oracle ({cfg.oracle_model})", "true turns", s))
        rows.sort(key=lambda r: (r[1] != "known", r[2]["cpwer"]))
        tables[ch] = rows
        print(f"\n{args.set} / {ch} (simulated)\n{table(rows)}")
        metrics |= {
            re.sub(r"[^\w./-]+", "_", f"{ch}/{spk}/{sys}/{m}").strip("_"): v
            for sys, spk, s in rows
            for m, v in s.items()
        }

    with Tracker(
        cfg.experiment,
        config=cfg.model_dump(mode="json") | {"set": args.set, "limit": args.limit or 0},
        dataset=f"{manifest.name}@{manifest.audio_sha256[:12]}",
        buffer_path=out / "tracking-buffer.jsonl",
        run_name=f"{args.set}" + (f"/limit-{args.limit}" if args.limit else ""),
    ) as tracker:
        tracker.log_metrics(metrics, step=0)

    expected = len(CHANNELS) * (len(SPEAKERS) * len(cfg.models) + 1)
    complete = sum(len(r) for r in tables.values()) == expected
    if args.set != cfg.readme_set or args.limit or not complete:
        print(f"\nREADME not updated (set={args.set}, limit={args.limit}, complete={complete})")
        return
    block = readme_block(tables, cfg, len(convs), manifest.hours)
    README.write_text(replace_block(README.read_text(encoding="utf-8"), cfg.readme_block, block))
    print(f"updated {README}")


if __name__ == "__main__":
    main()
