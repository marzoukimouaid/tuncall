"""Run the pipeline on audio files: one CallRecord JSON per file (D-24, D-55a).

uv run python scripts/run_pipeline.py call.wav [more.wav ...] --out outputs/records
"""

import argparse
from pathlib import Path

from tuncall.config import load_config
from tuncall.pipeline import PipelineConfig, load_models, run_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5/pipeline.yaml"))
    parser.add_argument("--out", type=Path, help="directory for <name>.json (default: stdout)")
    args = parser.parse_args()

    models = load_models(load_config(args.config, PipelineConfig))
    for path in args.audio:
        record = run_file(models, path)
        text = record.model_dump_json(indent=2)
        if args.out is None:
            print(text)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{path.stem}.json").write_text(text + "\n", encoding="utf-8")
            print(f"{path} -> {args.out / f'{path.stem}.json'}")


if __name__ == "__main__":
    main()
