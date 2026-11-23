from typing import Any

from datasheet import collect, composition, mix


def row(i: str, seconds: float, text: str = "عسلامة", speaker: str | None = None) -> dict[str, Any]:
    return {"id": i, "text": text, "duration": seconds, "speaker": speaker}


ROWS = [
    ("linto-train", row("linto/train/a", 1800.0, "ابعث mail")),
    ("linto-train", row("linto/train/b", 31.0)),  # over 30 s, like the long ones
    ("linto-train", row("linto/train/c", 1769.0)),  # excluded (D-16e)
    ("linto-test", row("linto/test/d", 4.0)),
    ("omnilingual-train", row("omnilingual/train/e", 3600.0, speaker="spk01")),
]


def test_collect_counts_per_split() -> None:
    splits = collect(ROWS, frozenset({"linto/train/c"}))
    train = splits["linto-train"]
    assert (train.utterances, train.seconds, train.longest) == (3, 3600.0, 1800.0)
    assert (train.over_limit, train.latin, train.excluded) == (3, 1, 1)
    assert train.excluded_seconds == 1769.0
    assert splits["omnilingual-train"].speakers == {"spk01"}


def test_composition_lists_splits_then_pending_sources() -> None:
    table = composition(collect(ROWS, frozenset({"linto/train/c"}))).splitlines()
    assert (
        table[2] == "| linto | train | 3 | 1.00 | 1200.0 | 1800.0 | 3 | not given | 1 (33%) | 1 |"
    )
    assert table[3] == "| linto | test | 1 | 0.00 | 4.0 | 4.0 | 0 | not given | 0 (0%) | - |"
    assert table[4].startswith("| omnilingual | train | 1 | 1.00 |")
    assert table[4].endswith("| 1 | 1 | 0 (0%) | 0 |")
    assert table[5:] == [
        "| deep_confessions | - | pending #39 | | | | | | | |",
        "| tunswitch_weak | - | pending #39 | | | | | | | |",
    ]


def test_mix_subtracts_exclusions_and_flags_missing_exports() -> None:
    splits = collect(ROWS, frozenset({"linto/train/c"}))
    assert mix(splits, ["linto", "omnilingual", "tedxtn"]).splitlines()[2:] == [
        "| linto | 2 | 0.51 |",
        "| omnilingual | 1 | 1.00 |",
        "| tedxtn | no export | |",
        "| **total** | **3** | **1.51** |",
    ]
