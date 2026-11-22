import json
from pathlib import Path

import numpy as np
from source_stats import export, read_export, table

from tuncall.data.sources import Record


def rec(source: str, split: str, seconds: float, key: str = "x") -> Record:
    audio = np.zeros(round(seconds * 16_000), dtype=np.float32)
    return Record(f"{source}/{split}/{key}", source, split, audio, "عسلامة", "spk", seconds)


def test_table_sums_hours_and_utterances_per_source_and_split() -> None:
    records = [rec("linto", "train", 1800.0), rec("linto", "train", 1800.0)]
    records += [rec("linto", "test", 360.0), rec("tedxtn", "train", 360.0)]
    assert table(records) == (
        "| source | split | utterances | hours |\n"
        "|---|---|---:|---:|\n"
        "| linto | train | 2 | 1.000 |\n"
        "| linto | test | 1 | 0.100 |\n"
        "| tedxtn | train | 1 | 0.100 |\n"
        "| total | | 4 | 1.200 |"
    )


def test_export_writes_one_jsonl_per_source_and_split(tmp_path: Path) -> None:
    records = [rec("linto", "train", 1.5, "a"), rec("linto", "test", 2.0, "b")]
    assert list(export(records, tmp_path)) == records
    lines = (tmp_path / "linto-train.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"id": "linto/train/a", "text": "عسلامة", "duration": 1.5, "speaker": "spk"}
    ]
    assert (tmp_path / "linto-test.jsonl").exists()


def test_read_export_round_trips(tmp_path: Path) -> None:
    list(export([rec("slurp_tn", "test", 1.0, "a")], tmp_path))
    assert [(name, row["id"]) for name, row in read_export(tmp_path)] == [
        ("slurp_tn-test", "slurp_tn/test/a")
    ]
