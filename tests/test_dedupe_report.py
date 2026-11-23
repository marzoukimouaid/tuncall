from dedupe_report import exclusions, overlap, render

SETS = {
    "linto-train": [
        ("l1", "عسلامة شنحوالك لاباس عليك"),
        ("l2", "عسلامة شنحوالك لاباس عليك"),  # internal duplicate
        ("l3", "باهي"),  # under 4 words: never counted
    ],
    "linto-test": [("t1", "عسلامة، شنحوالك لاباس عليك")],  # same after normalization
    "slurp_tn-train": [("s1", "ابعث mail لصحبتي سلمى")],
    "slurp_tn-test": [("s2", "ابعث ميل لصحبتي سلمى")],  # same after the lexicon
}


def test_overlap_counts_eligible_matches_per_set_pair() -> None:
    eligible, matrix, matches = overlap(SETS, {"mail": "ميل"})
    assert eligible == {"linto-train": 2, "linto-test": 1, "slurp_tn-train": 1, "slurp_tn-test": 1}
    assert matrix["linto-train"] == {
        "linto-train": 2,
        "linto-test": 2,
        "slurp_tn-train": 0,
        "slurp_tn-test": 0,
    }
    assert matrix["slurp_tn-train"]["slurp_tn-test"] == 1
    assert matrix["linto-test"]["linto-test"] == 0
    assert matches == {
        "l1": ["linto-test"],
        "l2": ["linto-test"],
        "t1": ["linto-train"],
        "s1": ["slurp_tn-test"],
        "s2": ["slurp_tn-train"],
    }


def test_render_lists_rows_and_columns_in_order() -> None:
    eligible, matrix, _ = overlap({"a-train": [("x", "و ب ت ث")], "a-test": []}, {})
    assert render(eligible, matrix) == (
        "| set | eligible | a-train | a-test |\n"
        "|---|---:|---:|---:|\n"
        "| a-train | 1 | 0 | 0 |\n"
        "| a-test | 0 | 0 | 0 |"
    )


def test_exclusions_keep_train_clips_that_match_an_evaluation_split() -> None:
    matches = {
        "linto/train/a": ["linto-test"],  # test-general: dropped
        "slurp_tn/train/b": ["slurp_tn-validation"],  # validation: dropped
        "linto/train/c": ["tedxtn-train"],  # train-only duplicate: kept
        "tedxtn/train/d": ["linto-train", "tedxtn-test"],  # one eval match is enough
        "linto/test/e": ["linto-train"],  # test clips are never in the mix
    }
    assert exclusions(matches) == ["linto/train/a", "slurp_tn/train/b", "tedxtn/train/d"]
