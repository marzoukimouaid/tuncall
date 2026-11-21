"""MLflow logging that never blocks or breaks training (D-80d).

Calls only put work on an in-memory queue. A background thread sends it to the
tracking server (DagsHub, via MLFLOW_TRACKING_URI / credentials in the env) every
``flush_seconds``. A batch that fails to send is appended to a local JSON-lines
buffer and re-sent on the next successful flush. Every error is logged, never
raised into the caller.

    with Tracker("asr", config=cfg.model_dump(mode="json"), dataset="org/ds@v0.1",
                 buffer_path="outputs/tracking-buffer.jsonl") as tracker:
        tracker.log_metrics({"loss": loss}, step=step)
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

MAX_ARTIFACT_BYTES = 50 * 1024 * 1024  # N-01 / D-80c: nothing bigger leaves home
_METRICS_PER_CALL = 1000  # MLflow log_batch limits
_ITEMS_PER_CALL = 100


class _Client(Protocol):
    """The subset of MlflowClient this module uses (lets tests pass a fake)."""

    def get_experiment_by_name(self, name: str) -> Any: ...
    def create_experiment(self, name: str) -> str: ...
    def create_run(self, experiment_id: str, *, run_name: str | None = None) -> Any: ...
    def log_batch(
        self, run_id: str, *, metrics: Any = (), params: Any = (), tags: Any = ()
    ) -> Any: ...
    def log_artifact(self, run_id: str, local_path: str) -> None: ...
    def set_terminated(self, run_id: str, *, status: str | None = None) -> None: ...


def detect_machine() -> str:
    """Where this process runs: colab, kaggle, ci or laptop (§5b rule 1)."""
    if "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "GITHUB_ACTIONS" in os.environ:
        return "ci"
    return "laptop"


def git_commit() -> str:
    """Current commit hash, suffixed with ``-dirty`` if there are uncommitted changes."""
    try:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": "1"}``, for MLflow params."""
    out: dict[str, str] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = str(value)
    return out


def _empty_record() -> dict[str, Any]:
    return {"metrics": [], "params": {}, "tags": {}, "artifacts": []}


class Tracker:
    """One MLflow run, logged from a background thread. See the module docstring."""

    def __init__(
        self,
        experiment: str,
        *,
        config: dict[str, Any],
        dataset: str,
        buffer_path: str | Path,
        run_name: str | None = None,
        client: _Client | None = None,
        flush_seconds: float = 10.0,
    ) -> None:
        self._client: _Client = client or MlflowClient()
        self._experiment = experiment
        self._run_name = run_name
        self._buffer = Path(buffer_path)
        self._flush_seconds = flush_seconds
        self._run_id: str | None = None
        self._status = "FINISHED"
        self._queue: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._stop = threading.Event()

        self.log_params(flatten(config))
        self.set_tags({"git_commit": git_commit(), "dataset": dataset, "machine": detect_machine()})
        self._thread = threading.Thread(target=self._loop, name="tuncall-tracker", daemon=True)
        self._thread.start()

    # --- called from training code: never block, never raise -----------------

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        try:
            now = int(time.time() * 1000)
            for key, value in metrics.items():
                self._queue.put(("metrics", [key, float(value), now, step]))
        except Exception:
            log.warning("tracker: could not queue metrics", exc_info=True)

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self._queue.put(("params", (key, str(value))))

    def set_tags(self, tags: dict[str, Any]) -> None:
        for key, value in tags.items():
            self._queue.put(("tags", (key, str(value))))

    def log_artifact(self, path: str | Path) -> None:
        """Queue a small file for upload; files over 50 MB are refused (N-01)."""
        try:
            size = os.path.getsize(path)
        except OSError:
            log.warning("tracker: artifact %s not found", path)
            return
        if size > MAX_ARTIFACT_BYTES:
            log.warning("tracker: refusing %s (%d MB > 50 MB, N-01)", path, size // 2**20)
            return
        self._queue.put(("artifacts", str(path)))

    def close(self, status: str = "FINISHED", timeout: float = 60.0) -> None:
        """Flush what is queued and end the run; waits at most ``timeout`` seconds."""
        self._status = status
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("tracker: still sending after %.0fs; leaving it in the background", timeout)

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close("FAILED" if exc_type else "FINISHED")

    # --- background thread ----------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._flush_seconds):
            self._flush()
        self._flush()
        if self._run_id is not None:
            try:
                self._client.set_terminated(self._run_id, status=self._status)
            except Exception:
                # shortcut: a run that ends offline stays RUNNING on the server; mark it by hand.
                log.warning("tracker: could not end run %s", self._run_id, exc_info=True)

    def _flush(self) -> None:
        batch = self._drain()
        try:
            pending = [*self._read_buffer(), batch]
            run_id = self._ensure_run()
            for record in pending:
                self._send(run_id, record)
        except Exception:
            # shortcut: if replay fails halfway, records already sent are sent again next time
            # (duplicate metric points); fine for a rare offline case, dedupe if it becomes common.
            log.warning("tracker: send failed, buffering to %s", self._buffer, exc_info=True)
            if any(batch.values()):
                self._append_buffer(batch)
            return
        self._buffer.unlink(missing_ok=True)

    def _drain(self) -> dict[str, Any]:
        record = _empty_record()
        while True:
            try:
                kind, item = self._queue.get_nowait()
            except queue.Empty:
                return record
            if kind in ("params", "tags"):
                record[kind][item[0]] = item[1]
            else:
                record[kind].append(item)

    def _ensure_run(self) -> str:
        if self._run_id is None:
            experiment = self._client.get_experiment_by_name(self._experiment)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self._client.create_experiment(self._experiment)
            )
            run = self._client.create_run(experiment_id, run_name=self._run_name)
            self._run_id = run.info.run_id
        assert self._run_id is not None
        return self._run_id

    def _send(self, run_id: str, record: dict[str, Any]) -> None:
        metrics = [Metric(k, v, ts, step) for k, v, ts, step in record["metrics"]]
        params = [Param(k, v) for k, v in record["params"].items()]
        tags = [RunTag(k, v) for k, v in record["tags"].items()]
        for i in range(0, len(metrics), _METRICS_PER_CALL):
            self._client.log_batch(run_id, metrics=metrics[i : i + _METRICS_PER_CALL])
        for i in range(0, max(len(params), len(tags)), _ITEMS_PER_CALL):
            chunk = slice(i, i + _ITEMS_PER_CALL)
            self._client.log_batch(run_id, params=params[chunk], tags=tags[chunk])
        for path in record["artifacts"]:
            self._client.log_artifact(run_id, path)

    def _read_buffer(self) -> list[dict[str, Any]]:
        if not self._buffer.exists():
            return []
        records = []
        for line in self._buffer.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("tracker: skipping corrupt buffer line in %s", self._buffer)
        return records

    def _append_buffer(self, record: dict[str, Any]) -> None:
        try:
            self._buffer.parent.mkdir(parents=True, exist_ok=True)
            with self._buffer.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            log.error("tracker: could not write buffer %s; batch lost", self._buffer, exc_info=True)
