"""Speech recognition: loading and running Whisper models (C2, D-22a)."""

from pathlib import Path

from tuncall.config import StrictModel


class ModelRef(StrictModel):
    """A Whisper model at a pinned Hub revision, optionally with a local LoRA adapter."""

    id: str
    revision: str
    # Local LoRA adapter dir, merged before decoding; its sha256 goes to MLflow.
    # shortcut: cached predictions are not keyed by the hash; rerun with --force after retraining.
    adapter: Path | None = None
