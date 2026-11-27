"""Batched Whisper inference with HF Transformers (D-22a, D-26)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16_000
WINDOW = 30 * SAMPLE_RATE  # Whisper's input window


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
    """Transcribe 16 kHz mono clips, deterministically, in order.

    Clips up to 30 s are batched (padded to 30 s, so the batch never changes a text).
    Each longer clip is decoded alone with Transformers' long-form (sequential) decoding
    (D-23a), so no clip's text depends on what it was batched with.
    """
    kwargs = {
        "language": language,
        "task": task,
        "num_beams": num_beams,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    short = [i for i, a in enumerate(audios) if len(a) <= WINDOW]
    texts: dict[int, str] = {}
    for b in range(0, len(short), batch_size):
        idx = short[b : b + batch_size]
        features = processor.feature_extractor(  # pyright: ignore[reportAttributeAccessIssue]
            [audios[i] for i in idx], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(model.device, model.dtype)
        ids = model.generate(features, **kwargs)
        texts |= dict(zip(idx, processor.batch_decode(ids, skip_special_tokens=True), strict=True))  # pyright: ignore[reportAttributeAccessIssue]
    for i in (i for i, a in enumerate(audios) if len(a) > WINDOW):
        inputs = processor.feature_extractor(  # pyright: ignore[reportAttributeAccessIssue]
            [audios[i]],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        ids = model.generate(
            inputs.input_features.to(model.device, model.dtype),
            attention_mask=inputs.attention_mask.to(model.device),
            return_timestamps=True,
            **kwargs,
        )
        texts[i] = processor.batch_decode(ids, skip_special_tokens=True)[0]  # pyright: ignore[reportAttributeAccessIssue]
    return [texts[i].strip() for i in range(len(audios))]
