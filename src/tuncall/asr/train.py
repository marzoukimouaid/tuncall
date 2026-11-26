"""LoRA fine-tuning of Whisper with the HF Seq2SeqTrainer (D-21b, D-21c).

The base model is frozen and LoRA adapters are trained (peft). The trainer handles batching,
gradient accumulation, mixed precision, checkpoints (best + latest, S-01) and resume; its logs
go to MLflow through ``Tracker`` (D-80d). The dev set is decoded like the benchmark (beam 5,
Arabic forced, D-26) and scored with the benchmark's normalization, so the best checkpoint is
the one with the lowest dev WER (D-17a).
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from tuncall.asr.cache import SAMPLE_RATE, Clips
from tuncall.config import StrictModel
from tuncall.data.lexicon import transliterate
from tuncall.eval.metrics import char_counts, error_rate, word_counts
from tuncall.eval.normalize import normalize_arabic
from tuncall.tracking import Tracker


class ModelRef(StrictModel):
    id: str
    revision: str


class LoraSettings(StrictModel):
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


class TrainConfig(StrictModel):
    experiment: str
    base_model: ModelRef
    mix: Path  # configs/data/train-mix.yaml (D-16); every weight must be 1.0
    lexicon: Path  # D-10a, applied to targets and when scoring
    dev_holdout: Path  # LinTO train clips held out as dev (D-17a)
    dev_splits: dict[str, str]  # source -> its own validation split, also dev (D-17a)
    cache: str  # cache root; $VARS expanded (D-21c)
    language: str
    task: str
    max_new_tokens: int
    lora: LoraSettings
    trainer: dict[str, Any]  # Seq2SeqTrainingArguments keyword arguments

    def cache_path(self) -> Path:
        return Path(os.path.expandvars(self.cache))


class Collator:
    """Pads log-mel features and target token ids; padding labels are -100 (ignored by loss)."""

    def __init__(
        self, processor: WhisperProcessor, decoder_start: int, max_labels: int, dtype: torch.dtype
    ) -> None:
        self.processor = processor
        self.dtype = dtype
        self.decoder_start = decoder_start
        self.max_labels = max_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        p: Any = self.processor
        features = p.feature_extractor(
            [b["audio"] for b in batch], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        # shortcut: over-long targets are cut to the decoder's window; none expected under 30 s.
        ids = [p.tokenizer(b["text"]).input_ids[: self.max_labels] for b in batch]
        labels = p.tokenizer.pad({"input_ids": ids}, return_tensors="pt")
        labels = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # The model prepends the start token itself when it shifts labels right.
        if (labels[:, 0] == self.decoder_start).all():
            labels = labels[:, 1:]
        return {"input_features": features.to(self.dtype), "labels": labels}


def scorer(processor: WhisperProcessor, lexicon: dict[str, str]) -> Any:
    """``compute_metrics`` for the trainer: dev WER/CER after the benchmark's normalization."""
    p: Any = processor

    def compute(pred: Any) -> dict[str, float]:
        label_ids = np.where(pred.label_ids == -100, p.tokenizer.pad_token_id, pred.label_ids)
        pred_ids = np.where(pred.predictions == -100, p.tokenizer.pad_token_id, pred.predictions)
        refs = p.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        hyps = p.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs, hyps = (
            [normalize_arabic(transliterate(t, lexicon)) for t in texts] for texts in (refs, hyps)
        )
        return {
            "wer": error_rate(word_counts(refs, hyps)),
            "cer": error_rate(char_counts(refs, hyps)),
        }

    return compute


class IntervalsFromConfig(TrainerCallback):
    """On resume the trainer restores logging/eval/save intervals from the checkpoint; use the
    config's instead, so changing them takes effect on the next run (D-21e)."""

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        state.compute_steps(args, state.max_steps)


class TrackerCallback(TrainerCallback):
    """Forwards every numeric trainer log line to MLflow (never blocks, never raises)."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
        numeric = {str(k): float(v) for k, v in (logs or {}).items() if isinstance(v, int | float)}
        self.tracker.log_metrics(numeric, step=state.global_step)


def load_lora(cfg: TrainConfig) -> tuple[Any, WhisperProcessor]:
    """Base model at its pinned revision with LoRA adapters attached, and its processor."""
    ref = cfg.base_model
    # Base weights in the training precision: eval (beam search) runs without autocast.
    dtype = torch.bfloat16 if cfg.trainer.get("bf16") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        ref.id, revision=ref.revision, dtype=dtype
    )
    processor: Any = WhisperProcessor.from_pretrained(ref.id, revision=ref.revision)
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task=cfg.task)
    # Feature window = the encoder's input (3000 frames = 30 s for real Whisper models).
    fe = processor.feature_extractor
    fe.nb_max_frames = 2 * model.config.max_source_positions
    fe.n_samples = fe.nb_max_frames * fe.hop_length
    gen: Any = model.generation_config  # Whisper adds language/task fields at runtime
    gen.language, gen.task, gen.forced_decoder_ids = cfg.language, cfg.task, None
    gen.max_new_tokens = cfg.max_new_tokens
    model.config.use_cache = False  # incompatible with activation checkpointing
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )
    return get_peft_model(model, lora), processor


def train(
    cfg: TrainConfig,
    train_clips: Clips,
    dev_clips: Clips,
    lexicon: dict[str, str],
    tracker: Tracker,
) -> dict[str, float]:
    """Train (resuming from the latest checkpoint in ``output_dir`` if any); return dev scores."""
    model, processor = load_lora(cfg)
    base = model.get_base_model()
    args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        report_to="none",
        remove_unused_columns=False,
        # non-reentrant checkpointing lets gradients reach LoRA layers behind frozen ones
        activation_checkpointing_kwargs={"use_reentrant": False},
        **cfg.trainer,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_clips,  # pyright: ignore[reportArgumentType]
        eval_dataset=dev_clips,  # pyright: ignore[reportArgumentType]
        data_collator=Collator(
            processor,
            base.config.decoder_start_token_id,
            base.config.max_target_positions,
            base.dtype,
        ),
        compute_metrics=scorer(processor, lexicon),
        callbacks=[IntervalsFromConfig(), TrackerCallback(tracker)],
    )
    resume = any(Path(args.output_dir or "").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    scores = trainer.evaluate()
    tracker.log_metrics(scores, step=trainer.state.global_step)
    trainer.save_model(str(Path(args.output_dir or "") / "best"))  # best adapter (load_best)
    return scores
