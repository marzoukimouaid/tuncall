from pathlib import Path

import pytest
from benchmark_asr import (
    BenchmarkConfig,
    is_loop,
    read_jsonl,
    render_table,
    score_rows,
    summary_table,
    to_run,
    write_jsonl,
)

from tuncall.config import load_config
from tuncall.eval.report import replace_block


def _rows(refs: list[str], hyps: list[str], lexicon: dict[str, str] | None = None) -> list[dict]:
    return score_rows([(f"u{i}", r) for i, r in enumerate(refs)], hyps, lexicon or {})


def test_empty_hypothesis_scores_as_deletions() -> None:
    [row] = _rows(["باهي برشا"], [""])
    assert (row["word_errors"], row["word_len"]) == (2, 2)
    assert row["char_errors"] == row["char_len"] == len("باهي برشا")


def test_score_rows_normalizes_both_sides() -> None:
    [row] = _rows(["مدرسة"], ["مدرسه"])
    assert row["word_errors"] == 0


def test_score_rows_transliterates_both_sides_with_the_lexicon() -> None:
    [row] = _rows(["ابعث Mail توا"], ["ابعث ميل donc"], {"mail": "ميل", "donc": "دونك"})
    assert row["ref"] == "ابعث Mail توا"  # the frozen reference is kept as is
    assert (row["ref_norm"], row["hyp_norm"]) == ("ابعث ميل توا", "ابعث ميل دونك")
    assert row["word_errors"] == 1


def test_render_table_sorts_by_wer_and_pairs_against_reference() -> None:
    refs = ["a b c d"] * 4
    results = {
        "ref": _rows(refs, ["a b x y"] * 4),  # 50% WER
        "good": _rows(refs, ["a b c d"] * 4),  # 0%
    }
    lines = render_table(results, "ref").splitlines()
    assert lines[0].startswith("| Model |") and "ΔWER vs ref" in lines[0]
    assert lines[2] == "| good | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -50.0 [-50.0, -50.0] | 0.0 | 0 |"
    assert lines[3].startswith("| ref | 50.0 [50.0, 50.0] |") and "| — |" in lines[3]


def test_render_table_without_reference_has_no_deltas() -> None:
    lines = render_table({"m": _rows(["a"], ["a"])}, "missing").splitlines()
    assert "| — |" in lines[2]


def test_replace_block_rewrites_only_between_markers() -> None:
    text = "intro\n<!-- t:start -->\nold\n<!-- t:end -->\noutro\n"
    new = replace_block(text, "t", "new")
    assert new == "intro\n<!-- t:start -->\nnew\n<!-- t:end -->\noutro\n"


def test_replace_block_refuses_missing_markers() -> None:
    with pytest.raises(ValueError, match="t:start"):
        replace_block("no markers here", "t", "new")


def test_resume_skips_cached_models_unless_forced(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "a.jsonl", _rows(["x"], ["x"]))
    (tmp_path / "b.jsonl.tmp").write_text("partial")  # killed run: not a cache hit
    assert to_run(["a", "b"], tmp_path, force=False) == ["b"]
    assert to_run(["a", "b"], tmp_path, force=True) == ["a", "b"]
    assert read_jsonl(tmp_path / "a.jsonl")[0]["hyp"] == "x"


def test_is_loop_flags_runaway_repetition_only() -> None:
    assert is_loop(" ".join(["نحن"] * 147))
    assert not is_loop("ايه ايه ايه باهي")  # short repeats are normal speech
    assert not is_loop(" ".join(f"w{i}" for i in range(40)))  # long but varied


def test_summary_table_has_one_column_per_test_set_sorted_by_the_first() -> None:
    refs = ["a b c d"] * 2
    results = {
        "general": {"m1": _rows(refs, ["a b x y"] * 2), "m2": _rows(refs, ["a b c d"] * 2)},
        "telephone": {"m1": _rows(refs, ["x y z w"] * 2), "m2": _rows(refs, ["a b c x"] * 2)},
    }
    lines = summary_table(results).splitlines()
    assert lines[0] == "| Model | general WER / CER % | telephone WER / CER % |"
    assert lines[2] == "| m2 | 0.0 / 0.0 | 25.0 / 14.3 |"
    assert lines[3].startswith("| m1 | 50.0 / ")


def test_baseline_config_scores_three_test_sets() -> None:
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/phase1/baselines.yaml", BenchmarkConfig
    )
    assert {k: (str(v.manifest), v.channel) for k, v in cfg.test_sets.items()} == {
        "test-general": ("configs/data/test-general.yaml", "clean"),
        "test-domain": ("configs/data/test-domain.yaml", "clean"),
        "telephone": ("configs/data/test-general.yaml", "telephone"),
    }
