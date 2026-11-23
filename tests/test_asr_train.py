from pathlib import Path

import numpy as np
import pytest
from linto_dev_holdout import holdout

from test_tracking import FakeClient
from tuncall.asr.cache import Clips, build, target_text
from tuncall.data.sources.common import record
from tuncall.tracking import Tracker

WHISPER_TINY = "openai/whisper-tiny"
WHISPER_TINY_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised two-layer Whisper with whisper-tiny's tokenizer and generation
    settings (a few MB of config files, no weights downloaded)."""
    from transformers import (
        GenerationConfig,
        WhisperConfig,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    config = WhisperConfig.from_pretrained(WHISPER_TINY, revision=WHISPER_TINY_REVISION)
    config.update(
        {
            "d_model": 16,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "encoder_attention_heads": 2,
            "decoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "max_source_positions": 30,  # 60 feature frames = 0.6 s
        }
    )
    model = WhisperForConditionalGeneration(config)
    model.generation_config = GenerationConfig.from_pretrained(
        WHISPER_TINY, revision=WHISPER_TINY_REVISION
    )
    path = tmp_path_factory.mktemp("tiny-whisper")
    model.save_pretrained(path)
    WhisperProcessor.from_pretrained(WHISPER_TINY, revision=WHISPER_TINY_REVISION).save_pretrained(
        path
    )
    return path


def clip(key: str, seconds: float, text: str) -> object:
    rng = np.random.default_rng(len(key))
    return record("src", "train", key, (0.1 * rng.standard_normal(int(seconds * 16_000))), text)


def test_target_text_transliterates_and_strips_harakat() -> None:
    assert target_text("بَاهِي  mail", {"mail": "ميل"}) == "باهي ميل"


def test_build_drops_long_and_empty_clips_and_round_trips_audio(tmp_path: Path) -> None:
    records = [clip("a", 1.0, "باهي"), clip("long", 31.0, "باهي"), clip("e", 1.0, "  َ ")]
    counts = build(records, tmp_path / "c", {})  # pyright: ignore[reportArgumentType]
    assert counts == {"kept": 1, "too_long": 1, "empty": 1}
    clips = Clips(tmp_path / "c")
    assert len(clips) == 1
    item = clips[0]
    assert item["id"] == "src/train/a" and item["text"] == "باهي"
    original = np.clip(records[0].audio, -1, 1)  # pyright: ignore[reportAttributeAccessIssue]
    assert np.abs(item["audio"] - original).max() < 1e-4  # pyright: ignore[reportOperatorIssue]


def test_holdout_keeps_recordings_whole_and_is_deterministic() -> None:
    ids = [f"linto/train/S/rec{r}/{s:03d}" for r in range(50) for s in range(r % 4 + 1)]
    chosen = holdout(ids, 20)
    assert len(chosen) <= 20 and chosen == holdout(list(reversed(ids)), 20)
    recordings = {c.rsplit("/", 1)[0] for c in chosen}
    assert {i for i in ids if i.rsplit("/", 1)[0] in recordings} == set(chosen)


def test_lora_training_smoke_checkpoints_and_resumes(tmp_path: Path, tiny_whisper: Path) -> None:
    """D-83a CPU smoke test: train, evaluate with beam search, checkpoint, resume."""
    pytest.importorskip("peft")
    from tuncall.asr.train import TrainConfig, train

    texts = ["باهي برشا", "شنوة الحكاية", "يعيشك", "ميل"]
    records = [clip(f"k{i}", 1.0 + i / 4, t) for i, t in enumerate(texts)]
    build(records, tmp_path / "train", {})  # pyright: ignore[reportArgumentType]
    build(records[:2], tmp_path / "dev", {})  # pyright: ignore[reportArgumentType]

    def run(max_steps: int) -> dict[str, float]:
        cfg = TrainConfig.model_validate(
            {
                "experiment": "smoke",
                "base_model": {"id": str(tiny_whisper), "revision": "local"},
                "mix": "unused",
                "lexicon": "unused",
                "dev_holdout": "unused",
                "dev_splits": {},
                "cache": str(tmp_path),
                "language": "ar",
                "task": "transcribe",
                "max_new_tokens": 8,
                "lora": {
                    "r": 2,
                    "alpha": 4,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj", "fc1"],
                },
                "trainer": {
                    "output_dir": str(tmp_path / "out"),
                    "per_device_train_batch_size": 2,
                    "per_device_eval_batch_size": 2,
                    "generation_num_beams": 2,
                    "max_steps": max_steps,
                    "eval_strategy": "steps",
                    "eval_steps": 2,
                    "save_steps": 2,
                    "save_total_limit": 2,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "wer",
                    "greater_is_better": False,
                    "activation_checkpointing": True,
                    "use_cpu": True,
                    "logging_steps": 1,
                },
            }
        )
        client = FakeClient()
        with Tracker(
            "smoke", config={}, dataset="x", buffer_path=tmp_path / "b.jsonl", client=client
        ) as tracker:
            scores = train(cfg, Clips(tmp_path / "train"), Clips(tmp_path / "dev"), {}, tracker)
        assert any(key == "loss" for key, _, _ in client.metrics)
        return scores

    first = run(2)
    assert "eval_wer" in first and (tmp_path / "out/checkpoint-2").is_dir()
    run(4)  # resumes from checkpoint-2
    assert (tmp_path / "out/checkpoint-4").is_dir()
    assert (tmp_path / "out/best/adapter_config.json").is_file()
