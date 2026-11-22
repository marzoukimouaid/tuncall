import numpy as np
import pytest

from tuncall.data.telephone import alaw, telephone

SR = 16_000


def tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def gain(x: np.ndarray) -> float:
    core = slice(SR // 10, -SR // 10)  # ignore filter edges
    return float(np.std(telephone(x)[core]) / np.std(x[core]))


def test_same_input_same_output() -> None:
    x = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)
    assert np.array_equal(telephone(x), telephone(x.copy()))


def test_keeps_length_dtype_and_rate() -> None:
    y = telephone(tone(1000, 0.37))
    assert y.shape == (round(0.37 * SR),) and y.dtype == np.float32


@pytest.mark.parametrize(("hz", "kept"), [(100, False), (1000, True), (3000, True), (5000, False)])
def test_passes_only_the_phone_band(hz: float, kept: bool) -> None:
    assert (gain(tone(hz)) > 0.9) if kept else (gain(tone(hz)) < 0.05)


def test_alaw_has_256_levels_and_small_error() -> None:
    x = np.linspace(-1, 1, 100_001, dtype=np.float32)
    y = alaw(x)
    assert len(np.unique(y)) <= 256
    assert np.max(np.abs(y - x)) < 0.025
    assert np.max(np.abs(y - x)[np.abs(x) < 0.01]) < 0.0005  # fine steps for quiet speech
