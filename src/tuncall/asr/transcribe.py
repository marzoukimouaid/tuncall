"""Batched Whisper inference with HF Transformers (D-22a, D-26)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16_000


def load(
    model_id: str,
    revision: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    adapter: Path | None = None,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Model and processor at a pinned Hub revision, in eval mode on ``device``. A LoRA
    ``adapter`` directory is merged into the base weights, as serving will do."""
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
    # pyright: ignore below: transformers types `to` and `from_pretrained` loosely
    return model.to(device).eval(), processor  # pyright: ignore[reportArgumentType, reportReturnType]


@torch.inference_mode()
def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audios: Sequence[np.ndarray],
    *,
    batch_size: int,
    language: str,
    task: str,
    num_beams: int,
    max_new_tokens: int,
) -> list[str]:
    """Transcribe 16 kHz mono clips (≤ 30 s each), deterministically, in order."""
    texts: list[str] = []
    for i in range(0, len(audios), batch_size):
        features = processor.feature_extractor(  # pyright: ignore[reportAttributeAccessIssue]
            list(audios[i : i + batch_size]), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(model.device, model.dtype)
        ids = model.generate(
            features,
            language=language,
            task=task,
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        texts += processor.batch_decode(ids, skip_special_tokens=True)  # pyright: ignore[reportAttributeAccessIssue]
    return [t.strip() for t in texts]
