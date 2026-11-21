from pathlib import Path

import numpy as np
import pytest
from benchmark_asr import read_jsonl, render_table, score_rows, to_run, write_jsonl

from tuncall.data.testsets import Utterance
from tuncall.eval.report import replace_block

SILENCE = np.zeros(1, dtype=np.float32)


def _rows(refs: list[str], hyps: list[str]) -> list[dict]:
    utts = [Utterance(f"u{i}", SILENCE, r) for i, r in enumerate(refs)]
    return score_rows(utts, hyps)


def test_empty_hypothesis_scores_as_deletions() -> None:
    [row] = _rows(["باهي برشا"], [""])
    assert (row["word_errors"], row["word_len"]) == (2, 2)
    assert row["char_errors"] == row["char_len"] == len("باهي برشا")


def test_score_rows_normalizes_both_sides() -> None:
    [row] = _rows(["مدرسة"], ["مدرسه"])
    assert row["word_errors"] == 0


def test_render_table_sorts_by_wer_and_pairs_against_reference() -> None:
    refs = ["a b c d"] * 4
    results = {
        "ref": _rows(refs, ["a b x y"] * 4),  # 50% WER
        "good": _rows(refs, ["a b c d"] * 4),  # 0%
    }
    lines = render_table(results, "ref").splitlines()
    assert lines[0].startswith("| Model |") and "ΔWER vs ref" in lines[0]
    assert lines[2] == "| good | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -50.0 [-50.0, -50.0] | 0.0 |"
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
