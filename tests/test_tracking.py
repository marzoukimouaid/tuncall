import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tuncall.tracking import MAX_ARTIFACT_BYTES, Tracker, flatten


class FakeClient:
    """In-memory stand-in for MlflowClient; ``down`` makes every call fail."""

    def __init__(self, down: bool = False) -> None:
        self.down = down
        self.metrics: list[tuple[str, float, int]] = []
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []
        self.status: str | None = None

    def _check(self) -> None:
        if self.down:
            raise ConnectionError("tracking server unreachable")

    def get_experiment_by_name(self, name: str) -> Any:
        self._check()
        return None

    def create_experiment(self, name: str) -> str:
        self._check()
        return "exp-1"

    def create_run(self, experiment_id: str, run_name: str | None = None) -> Any:
        self._check()
        return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    def log_batch(self, run_id: str, metrics: Any = (), params: Any = (), tags: Any = ()) -> None:
        self._check()
        self.metrics += [(m.key, m.value, m.step) for m in metrics]
        self.params |= {p.key: p.value for p in params}
        self.tags |= {t.key: t.value for t in tags}

    def log_artifact(self, run_id: str, local_path: str) -> None:
        self._check()
        self.artifacts.append(local_path)

    def set_terminated(self, run_id: str, status: str | None = None) -> None:
        self._check()
        self.status = status


def make(client: FakeClient, tmp_path: Path, flush_seconds: float = 0.01) -> Tracker:
    return Tracker(
        "test",
        config={"training": {"lr": 0.1}},
        dataset="org/ds@v0.1",
        buffer_path=tmp_path / "buffer.jsonl",
        client=client,
        flush_seconds=flush_seconds,
    )


def test_logs_metrics_params_and_run_metadata(tmp_path: Path) -> None:
    client = FakeClient()
    with make(client, tmp_path) as tracker:
        tracker.log_metrics({"loss": 0.5}, step=1)

    assert client.metrics == [("loss", 0.5, 1)]
    assert client.params == {"training.lr": "0.1"}
    assert {"git_commit", "dataset", "machine"} <= client.tags.keys()
    assert client.tags["dataset"] == "org/ds@v0.1"
    assert client.status == "FINISHED"


def test_unreachable_server_never_raises_and_buffers_to_disk(tmp_path: Path) -> None:
    client = FakeClient(down=True)
    tracker = make(client, tmp_path)
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.close()  # must not raise

    buffered = (tmp_path / "buffer.jsonl").read_text(encoding="utf-8")
    assert '"loss"' in buffered
    assert '"training.lr"' in buffered


def test_buffer_is_replayed_when_the_server_comes_back(tmp_path: Path) -> None:
    client = FakeClient(down=True)
    tracker = make(client, tmp_path, flush_seconds=3600)  # only the explicit flushes below run
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker._flush()
    assert (tmp_path / "buffer.jsonl").exists()

    client.down = False
    tracker.log_metrics({"loss": 0.4}, step=2)
    tracker.close()

    assert sorted(client.metrics) == [("loss", 0.4, 2), ("loss", 0.5, 1)]
    assert not (tmp_path / "buffer.jsonl").exists()


def test_exception_in_block_marks_run_failed(tmp_path: Path) -> None:
    client = FakeClient()
    try:
        with make(client, tmp_path):
            raise RuntimeError("training crashed")
    except RuntimeError:
        pass
    assert client.status == "FAILED"


def test_artifact_over_50_mb_is_refused(tmp_path: Path) -> None:
    big = tmp_path / "big.bin"
    with big.open("wb") as f:
        f.truncate(MAX_ARTIFACT_BYTES + 1)  # sparse file, no real disk use
    small = tmp_path / "small.txt"
    small.write_text("ok", encoding="utf-8")

    client = FakeClient()
    with make(client, tmp_path) as tracker:
        tracker.log_artifact(big)
        tracker.log_artifact(small)
        tracker.log_artifact(tmp_path / "missing.txt")

    assert client.artifacts == [os.fspath(small)]


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": [1, 2]}) == {
        "a.b": "1",
        "a.c.d": "x",
        "e": "[1, 2]",
    }
