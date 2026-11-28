"""Phase 6 serving: a Postgres-backed call queue, the GPU worker and the API (ADR 0010).

The API side (``db``, ``app``) must never import torch (D-52a); only ``worker`` loads the
pipeline.
"""

import json
import logging
from pathlib import Path

from tuncall.config import StrictModel


class ServeConfig(StrictModel):
    """``configs/phase6/serve.yaml`` (D-87b). Secrets and ``DATABASE_URL`` come from the env."""

    pipeline_config: Path = Path("configs/phase5/pipeline.yaml")
    storage_dir: Path = Path("data/uploads")  # the audio volume (D-50d)
    max_upload_mb: int = 400  # a 30-minute 44.1 kHz stereo 16-bit WAV is ~320 MB
    max_duration_s: float = 1800  # D-50c: 30 minutes per call
    default_num_speakers: int = 2  # D-50b
    poll_interval_s: float = 2.0  # how often an idle worker looks for a queued call
    lease_s: float = 900  # D-55d: about 3x the expected time for a 30-minute call
    max_attempts: int = 2  # D-55d
    web_dir: Path = Path("web/dist")  # the built dashboard, served at / when present


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, message and any ``extra=`` fields (D-85a)."""

    _skip = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out |= {k: v for k, v in record.__dict__.items() if k not in self._skip}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
