"""Telephone-band channel simulation (D-25, D-25a).

A landline call: the 300-3400 Hz phone band, 8 kHz sampling and G.711 A-law 8-bit
companding, then back to 16 kHz for the models. No noise and no randomness, so the same
input always gives the same output and the telephone test set needs no frozen files.
"""

import numpy as np
import soxr

from tuncall.data.testsets import SAMPLE_RATE

PHONE_RATE = 8_000
BAND = (300.0, 3400.0)
A = 87.6


def alaw(x: np.ndarray) -> np.ndarray:
    """Compress with the A-law curve, quantize to 8 bits, expand back."""
    # shortcut: continuous A-law formula, not the bit-exact G.711 segment table; switch if a
    # real codec ever needs matching sample for sample.
    mag = np.minimum(np.abs(x.astype(np.float64)), 1.0)
    small = mag < 1 / A
    y = np.where(small, A * mag, 1 + np.log(np.maximum(A * mag, 1.0))) / (1 + np.log(A))
    y = np.round(y * 127) / 127
    t = y * (1 + np.log(A))
    out = np.where(t < 1, t / A, np.exp(t - 1) / A)
    return (np.sign(x) * out).astype(np.float32)


def _band(x: np.ndarray, sr: int) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    hz = np.fft.rfftfreq(len(x), 1 / sr)
    spectrum[(hz < BAND[0]) | (hz > BAND[1])] = 0
    return np.fft.irfft(spectrum, len(x))


def telephone(audio: np.ndarray) -> np.ndarray:
    """16 kHz mono float32 in, the same length out, as heard through a landline."""
    narrow = _band(soxr.resample(audio, SAMPLE_RATE, PHONE_RATE), PHONE_RATE)
    wide = soxr.resample(alaw(narrow), PHONE_RATE, SAMPLE_RATE)
    out = np.zeros(len(audio), dtype=np.float32)
    n = min(len(wide), len(audio))
    out[:n] = wide[:n]
    return out
