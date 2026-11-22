import numpy as np
from source_stats import table

from tuncall.data.sources import Record


def rec(source: str, seconds: float) -> Record:
    audio = np.zeros(round(seconds * 16_000), dtype=np.float32)
    return Record(f"{source}/train/x", source, "train", audio, "", None, seconds)


def test_table_sums_hours_and_utterances_per_source() -> None:
    records = [rec("linto", 1800.0), rec("linto", 1800.0), rec("tedxtn", 360.0)]
    assert table(records) == (
        "| source | utterances | hours |\n"
        "|---|---:|---:|\n"
        "| linto | 2 | 1.000 |\n"
        "| tedxtn | 1 | 0.100 |\n"
        "| total | 3 | 1.100 |"
    )
