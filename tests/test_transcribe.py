from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tuncall.asr.transcribe import SAMPLE_RATE, WINDOW, transcribe


class FakeModel:
    """Its 'transcript' of a clip is the clip's length in samples."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], bool]] = []

    def generate(self, features: torch.Tensor, **kw: Any) -> torch.Tensor:
        lengths = features[:, 0].long().tolist()
        self.calls.append((lengths, kw.get("return_timestamps", False)))
        return torch.tensor(lengths)[:, None]


class FakeProcessor:
    def feature_extractor(self, batch: list[np.ndarray], **kw: Any) -> Any:
        lengths = torch.tensor([[float(len(a))] for a in batch])
        return SimpleNamespace(input_features=lengths, attention_mask=torch.ones(len(batch), 1))

    def batch_decode(self, ids: torch.Tensor, **kw: Any) -> list[str]:
        return [f" {int(i)} " for i in ids[:, 0]]


def test_long_clips_are_decoded_alone_and_order_is_kept() -> None:
    lengths = [SAMPLE_RATE, WINDOW + 1, 2 * SAMPLE_RATE, WINDOW, 3 * WINDOW, 5]
    model = FakeModel()
    texts = transcribe(
        model,  # pyright: ignore[reportArgumentType]
        FakeProcessor(),  # pyright: ignore[reportArgumentType]
        [np.zeros(n, dtype=np.float32) for n in lengths],
        batch_size=2,
        language="ar",
        task="transcribe",
        num_beams=1,
        max_new_tokens=8,
    )
    assert texts == [str(n) for n in lengths]
    assert model.calls == [
        ([SAMPLE_RATE, 2 * SAMPLE_RATE], False),  # short clips batched, in order
        ([WINDOW, 5], False),  # exactly 30 s is still short
        ([WINDOW + 1], True),  # each long clip alone, with timestamps
        ([3 * WINDOW], True),
    ]
