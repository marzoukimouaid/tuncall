"""LoRA fine-tuning of Whisper large-v3-turbo (D-21b) from the caches built by
``scripts/prepare_asr_data.py``. Rerunning resumes from the latest checkpoint.

    uv run python scripts/train_asr.py --config configs/phase3/lora.yaml
"""

import argparse
from pathlib import Path

from tuncall.asr.cache import Clips
from tuncall.asr.train import TrainConfig, train
from tuncall.config import load_config
from tuncall.data.lexicon import load_lexicon
from tuncall.data.testsets import sha256_file
from tuncall.tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="use the --limit N caches (pre-flight)")
    args = parser.parse_args()
    cfg = load_config(args.config, TrainConfig)
    if args.limit:
        out = Path(cfg.trainer["output_dir"]) / f"limit-{args.limit}"
        cfg = cfg.model_copy(update={"trainer": cfg.trainer | {"output_dir": str(out)}})
    cache = cfg.cache_path() / (f"limit-{args.limit}" if args.limit else "")
    train_clips, dev_clips = Clips(cache / "train"), Clips(cache / "dev")
    data_id = f"train {sha256_file(cache / 'train/index.jsonl')[:12]}"
    data_id += f" dev {sha256_file(cache / 'dev/index.jsonl')[:12]}"
    out_dir = Path(cfg.trainer["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with Tracker(
        cfg.experiment,
        config=cfg.model_dump(mode="json")
        | {
            "train_clips": len(train_clips),
            "dev_clips": len(dev_clips),
            "lexicon_sha256": sha256_file(ROOT / cfg.lexicon),
        },
        dataset=data_id,
        buffer_path=out_dir / "tracking-buffer.jsonl",
        run_name=out_dir.name,
    ) as tracker:
        scores = train(cfg, train_clips, dev_clips, load_lexicon(ROOT / cfg.lexicon), tracker)
    print(scores)


if __name__ == "__main__":
    main()
