from pathlib import Path

import pytest
from pydantic import ValidationError

from tuncall.config import StrictModel, deep_merge, load_config


class Training(StrictModel):
    lr: float
    batch_size: int


class RunConfig(StrictModel):
    name: str
    training: Training


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_keeps_unchanged_keys_and_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": 2}, "tags": ["one", "two"]}
    merged = deep_merge(base, {"a": {"y": 3}, "tags": ["three"]})
    assert merged == {"a": {"x": 1, "y": 3}, "tags": ["three"]}
    assert base == {"a": {"x": 1, "y": 2}, "tags": ["one", "two"]}


def test_load_config_resolves_base_chain(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", "name: base\ntraining:\n  lr: 3.0e-4\n  batch_size: 16\n")
    (tmp_path / "asr").mkdir()
    run = write(tmp_path / "asr" / "run.yaml", "base: ../base.yaml\ntraining:\n  lr: 1.0e-4\n")

    cfg = load_config(run, RunConfig)

    assert cfg.name == "base"
    assert cfg.training == Training(lr=1.0e-4, batch_size=16)
    assert cfg.model_dump(mode="json")["training"]["batch_size"] == 16


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    run = write(tmp_path / "run.yaml", "name: r\ntraining:\n  lr: 0.1\n  batch_size: 1\n  lrr: 2\n")
    with pytest.raises(ValidationError, match="lrr"):
        load_config(run, RunConfig)


def test_wrong_type_is_an_error(tmp_path: Path) -> None:
    run = write(tmp_path / "run.yaml", "name: r\ntraining:\n  lr: fast\n  batch_size: 1\n")
    with pytest.raises(ValidationError, match="lr"):
        load_config(run, RunConfig)


def test_inheritance_cycle_is_an_error(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "base: b.yaml\n")
    write(tmp_path / "b.yaml", "base: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml", RunConfig)


def test_non_mapping_file_is_an_error(tmp_path: Path) -> None:
    run = write(tmp_path / "run.yaml", "- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(run, RunConfig)
