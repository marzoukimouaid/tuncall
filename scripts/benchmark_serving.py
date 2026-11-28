"""Phase 6 serving benchmark (D-70e): drives the running stack end to end through the API.

For each channel (clean, telephone):
- **latency:** every conversation uploaded one at a time, each waited for. Processing seconds
  per minute of audio, by stage (from ``CallRecord.meta.processing_ms_by_stage``) and in
  total (worker time, claim to finish), and the cost per hour of audio from the GPU price
  in the config.
- **load:** N conversations uploaded at once, for each N in ``loads`` (N = 1 is the latency
  run): throughput (calls/hour), queue wait and p95 time to result. One worker, so
  throughput is capped; it is reported as it is.

Start the stack first (``docker compose up``), then
    uv run python scripts/benchmark_serving.py --config configs/phase6/benchmark.yaml --set dev
Tune on dev only; the frozen test set is run once, and only a complete, unlimited test run
writes the README block. ``--limit N`` (first N conversations, loads capped at N) is the
pre-flight. The benchmark's calls are deleted afterwards.
"""

import argparse
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from tuncall.config import StrictModel, load_config
from tuncall.data.conversations import ConversationSet, load_conversations
from tuncall.data.telephone import telephone
from tuncall.data.testsets import REPO_ROOT, SAMPLE_RATE
from tuncall.eval.report import replace_block
from tuncall.tracking import Tracker

README = REPO_ROOT / "README.md"
CHANNELS = ("clean", "telephone")
STAGES = ("diarize", "asr", "nlp")


class GpuPrice(StrictModel):
    """The hourly rental price of a GPU comparable to the reference machine (D-70e)."""

    gpu: str
    usd_per_hour: float
    source: str
    date: str


class BenchmarkConfig(StrictModel):
    sets: dict[str, Path]
    readme_set: str
    readme_block: str
    experiment: str
    api_url: str
    num_speakers: int
    loads: list[int]
    poll_interval_s: float
    timeout_s: float
    price: GpuPrice


# --- metrics (pure, tested) ---------------------------------------------------------------


def _t(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); exact on small samples."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def latency(calls: list[dict[str, Any]], usd_per_hour: float) -> dict[str, float]:
    """Seconds of processing per minute of audio, by stage and in total, and $ per audio hour.

    ``total`` is worker time (claim to finish), so it includes decoding and storing.
    """
    minutes = sum(c["record"]["duration_s"] for c in calls) / 60
    out = {
        f"{stage}_s_per_min": sum(
            c["record"]["meta"]["processing_ms_by_stage"].get(stage, 0.0) for c in calls
        )
        / 1000
        / minutes
        for stage in STAGES
    }
    worker_s = sum(_t(c["finished_at"]) - _t(c["claimed_at"]) for c in calls)
    out["total_s_per_min"] = worker_s / minutes
    out["usd_per_audio_hour"] = worker_s / 3600 / (minutes / 60) * usd_per_hour
    out["audio_minutes"] = minutes
    return out


def load(calls: list[dict[str, Any]]) -> dict[str, float]:
    """Throughput, queue wait and time to result for calls uploaded together."""
    created = [_t(c["created_at"]) for c in calls]
    finished = [_t(c["finished_at"]) for c in calls]
    waits = [_t(c["claimed_at"]) - t for c, t in zip(calls, created, strict=True)]
    results = [f - c for f, c in zip(finished, created, strict=True)]
    return {
        "calls": len(calls),
        "calls_per_hour": len(calls) / (max(finished) - min(created)) * 3600,
        "queue_wait_mean_s": sum(waits) / len(waits),
        "queue_wait_p95_s": percentile(waits, 95),
        "time_to_result_p50_s": percentile(results, 50),
        "time_to_result_p95_s": percentile(results, 95),
    }


# --- driving the API ----------------------------------------------------------------------


def wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def upload(client: httpx.Client, name: str, data: bytes, num_speakers: int) -> str:
    r = client.post(
        "/calls", files={"file": (f"{name}.wav", data)}, data={"num_speakers": str(num_speakers)}
    )
    r.raise_for_status()
    return r.json()["call_id"]


def wait(client: httpx.Client, ids: list[str], cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    deadline = time.monotonic() + cfg.timeout_s
    done: dict[str, dict[str, Any]] = {}
    while len(done) < len(ids):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{len(ids) - len(done)} calls not done after {cfg.timeout_s} s")
        for call_id in ids:
            if call_id not in done:
                call = client.get(f"/calls/{call_id}").raise_for_status().json()
                if call["status"] == "failed":
                    raise RuntimeError(f"call {call_id} failed: {call['error']}")
                if call["status"] == "done":
                    done[call_id] = call
        time.sleep(cfg.poll_interval_s)
    return [done[i] for i in ids]


def run_channel(
    client: httpx.Client, audios: list[tuple[str, bytes]], cfg: BenchmarkConfig
) -> dict[str, Any]:
    created: list[str] = []
    try:
        sequential = []
        for name, data in audios:
            call_id = upload(client, name, data, cfg.num_speakers)
            created.append(call_id)
            sequential += wait(client, [call_id], cfg)
            print(f"  {name}: {sequential[-1]['record']['duration_s']:.0f} s audio", flush=True)
        out: dict[str, Any] = {
            "latency": latency(sequential, cfg.price.usd_per_hour),
            "load": {"1": load(sequential[:1]) | {"note": "from the latency run"}},
        }
        for n in cfg.loads:
            if n == 1:
                continue
            batch = [audios[i % len(audios)] for i in range(n)]
            with ThreadPoolExecutor(n) as pool:
                ids = list(pool.map(lambda a: upload(client, a[0], a[1], cfg.num_speakers), batch))
            created += ids
            out["load"][str(n)] = load(wait(client, ids, cfg))
            print(f"  load {n}: {out['load'][str(n)]}", flush=True)
        return out
    finally:
        for call_id in created:
            client.delete(f"/calls/{call_id}")


# --- report -------------------------------------------------------------------------------


def table(results: dict[str, Any], manifest: ConversationSet, cfg: BenchmarkConfig) -> str:
    p = cfg.price
    lines = [
        f"{manifest.conversations} simulated conversations ({manifest.hours} h, "
        f"`{cfg.sets[cfg.readme_set]}`), uploaded through the API to one worker on the "
        "reference laptop (RTX 4070 Laptop, 8 GB).",
        "",
        "| Channel | Diarization | ASR | NLP | Total (worker) | $ per audio hour |",
        "|---|---|---|---|---|---|",
    ]
    for ch, r in results.items():
        lat = r["latency"]
        lines.append(
            f"| {ch} | {lat['diarize_s_per_min']:.1f} | {lat['asr_s_per_min']:.1f} | "
            f"{lat['nlp_s_per_min']:.1f} | {lat['total_s_per_min']:.1f} | "
            f"{lat['usd_per_audio_hour']:.3f} |"
        )
    lines += [
        "",
        "Seconds of processing per minute of audio. Cost = worker GPU-hours per audio hour x "
        f"${p.usd_per_hour}/h ({p.gpu}, {p.source}, {p.date}).",
        "",
        "| Channel | Simultaneous uploads | Calls/hour | Queue wait mean / p95 (s) | "
        "Time to result p50 / p95 (s) |",
        "|---|---|---|---|---|",
    ]
    for ch, r in results.items():
        for n, ld in r["load"].items():
            lines.append(
                f"| {ch} | {n} | {ld['calls_per_hour']:.0f} | "
                f"{ld['queue_wait_mean_s']:.0f} / {ld['queue_wait_p95_s']:.0f} | "
                f"{ld['time_to_result_p50_s']:.0f} / {ld['time_to_result_p95_s']:.0f} |"
            )
    lines += [
        "",
        "One GPU worker takes one call at a time (D-52a), so throughput is capped and queue "
        "wait grows with the number of simultaneous uploads.",
        "Reproduce: `docker compose up`, then `uv run python scripts/benchmark_serving.py "
        "--config configs/phase6/benchmark.yaml --set test`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6/benchmark.yaml"))
    parser.add_argument("--set", required=True)
    parser.add_argument("--limit", type=int, help="pre-flight: first N conversations only")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase6"))
    args = parser.parse_args()

    cfg = load_config(args.config, BenchmarkConfig)
    manifest = load_config(REPO_ROOT / cfg.sets[args.set], ConversationSet)
    conversations = load_conversations(manifest)[: args.limit]
    if args.limit:
        cfg = cfg.model_copy(update={"loads": [min(n, args.limit) for n in cfg.loads]})

    results: dict[str, Any] = {}
    with httpx.Client(base_url=cfg.api_url, timeout=120) as client:
        client.get("/health").raise_for_status()
        for ch in CHANNELS:
            print(f"{ch}:", flush=True)
            audios = [
                (c.id, wav(telephone(c.audio) if ch == "telephone" else c.audio))
                for c in conversations
            ]
            results[ch] = run_channel(client, audios, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"set": args.set, "limit": args.limit, "config": cfg.model_dump(mode="json")}
    path = args.out / f"benchmark-{args.set}{f'-limit{args.limit}' if args.limit else ''}.json"
    path.write_text(json.dumps(report | {"results": results}, indent=2) + "\n")
    print(f"wrote {path}")

    metrics = {
        f"{ch}/{k}": v
        for ch, r in results.items()
        for k, v in r["latency"].items()
        if isinstance(v, float)
    }
    with Tracker(
        cfg.experiment,
        config=cfg.model_dump(mode="json") | {"set": args.set, "limit": args.limit or 0},
        dataset=f"{manifest.name}@{manifest.audio_sha256[:12]}",
        buffer_path=args.out / "tracking-buffer.jsonl",
        run_name=f"{args.set}" + (f"/limit-{args.limit}" if args.limit else ""),
    ) as tracker:
        tracker.log_metrics(metrics, step=0)

    if args.set == cfg.readme_set and not args.limit:
        README.write_text(
            replace_block(README.read_text(), cfg.readme_block, table(results, manifest, cfg))
        )
        print(f"README {cfg.readme_block} block updated")


if __name__ == "__main__":
    main()
