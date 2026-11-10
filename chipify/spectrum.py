# Copyright (c) 2026 Santiago Hofwimmer
"""
spectrum.py – Amplitude spectrum of a captured transient waveform.

The one FFT implementation behind every spectrum chipify draws: the Plots
tab's *Spectrum* view, the Multi-plot dashboard cell and a datasheet's
``type: spectrum`` report plot. Headless, no GUI deps.

The input is the transient CSV chipify already captures (``transient_signals:``),
so the analysis itself stays in the testbench: its ``.tran`` decides the time
window and the step.

What every caller can rely on:

* **Uniform sampling.** ngspice and VACASK choose their own time steps, dense
  around edges and sparse elsewhere. An FFT that treats such samples as evenly
  spaced puts energy at the wrong frequencies, so the samples are linearly
  interpolated onto an even grid with their mean step (like ngspice's
  ``linearize``). Content above that grid's Nyquist frequency aliases: a clean
  spectrum needs a bounded step in the testbench
  (``.tran <tstep> <tstop> <tstart> <tmax>``) and a ``tstart`` past the
  settling.
* **Whole periods are coherent.** The grid covers the simulated interval
  without its end point: a simulator writes both ``tstart`` and ``tstop``, and
  for a periodic waveform the end repeats the start. A whole number of periods
  then fills the FFT record exactly, and a uniformly stepped waveform is used
  sample for sample. Including the end point made the record one step too
  long: on a real ngspice run that left -60 dB of leakage beside the tone with
  the rectangular window, and a -90 dB third harmonic read -79.5 dB.
* **Peak amplitude.** A sine of amplitude A reads A at its frequency: the
  spectrum is single-sided and divided by the window's coherent gain. A tone
  that falls between two bins (non-coherent sampling) reads lower, least with
  ``blackman_harris`` and most with ``rectangular``.
* **DC is kept** unless ``remove_dc`` is set: bin 0 is the waveform's mean.
  The plot sets it. Its log axis has no 0 Hz, and the window spreads a bias
  point's DC over the lowest bins, where it hid a fundamental only a few bins
  up (a simulation of three or four periods).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Cosine-sum coefficients a_k of each window,
#: ``w[n] = sum_k (-1)^k a_k cos(2 pi k n / N)``, in the periodic (DFT-even)
#: form spectral analysis wants. Insertion order is the GUI order; the first
#: entry is the default.
_COSINE_TERMS: dict[str, tuple[float, ...]] = {
    "hann": (0.5, 0.5),
    # 4-term Blackman-Harris: -92 dB side lobes, for harmonics far below the
    # fundamental.
    "blackman_harris": (0.35875, 0.48829, 0.14128, 0.01168),
    # No window: exact bins for coherently sampled tones, heavy leakage else.
    "rectangular": (1.0,),
}

#: The ``window:`` vocabulary of a spectrum plot, default first.
WINDOWS: tuple[str, ...] = tuple(_COSINE_TERMS)
DEFAULT_WINDOW: str = WINDOWS[0]

#: Display names for the GUI selectors.
WINDOW_LABELS: dict[str, str] = {
    "hann": "Hann",
    "blackman_harris": "Blackman-Harris",
    "rectangular": "Rectangular",
}

#: Fewest distinct time points a spectrum is computed from.
MIN_SAMPLES = 4


def check_window(name: object) -> str:
    """The canonical name of window *name*, or ValueError naming the choices.

    Case-insensitive, so ``Hann`` in a datasheet is the ``hann`` window.
    """
    key = str(name or "").strip().lower()
    if key not in _COSINE_TERMS:
        raise ValueError(
            f"unknown window {name!r}; expected one of {', '.join(WINDOWS)}"
        )
    return key


def window_values(name: str, n: int) -> FloatArray:
    """The *n* coefficients of window *name*."""
    terms = _COSINE_TERMS[check_window(name)]
    phase = 2.0 * np.pi * np.arange(n) / n
    w = np.zeros(n, dtype=np.float64)
    for k, a in enumerate(terms):
        w += (-1) ** k * a * np.cos(k * phase)
    return w


def amplitude_spectrum(time: npt.ArrayLike, values: npt.ArrayLike,
                       window: str = DEFAULT_WINDOW, *,
                       remove_dc: bool = False) -> tuple[FloatArray, FloatArray]:
    """Single-sided peak amplitude spectrum of *values* sampled at *time*.

    Returns ``(frequency in Hz, amplitude in the signal's unit)``. Non-finite
    samples are dropped, and a time point written twice (simulators repeat
    breakpoints) keeps its first value. *remove_dc* subtracts the mean of the
    resampled waveform before windowing; the mean of the simulator's own
    samples would be weighted toward where its steps are dense. Raises
    ValueError for an unknown window or fewer than :data:`MIN_SAMPLES`
    distinct time points.
    """
    name = check_window(window)
    t = np.asarray(time, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError(
            f"time and values differ in length ({t.size} against {y.size})")
    keep = np.isfinite(t) & np.isfinite(y)
    # np.interp needs strictly increasing times; unique() also sorts.
    t, first = np.unique(t[keep], return_index=True)
    y = y[keep][first]
    n = int(t.size)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"a spectrum needs at least {MIN_SAMPLES} time points, got {n}")

    # n - 1 steps of the mean size over [t0, tstop): the end point is the
    # periodic repeat of the start (see the module docstring).
    m = n - 1
    dt = float(t[-1] - t[0]) / m
    uniform = np.interp(t[0] + dt * np.arange(m), t, y)
    if remove_dc:
        uniform = uniform - uniform.mean()
    w = window_values(name, m)
    amp = np.abs(np.fft.rfft(uniform * w)) / float(w.sum())
    # Single-sided: every bin but DC, and the Nyquist bin of an even length,
    # also carries its negative-frequency twin.
    amp[1:] *= 2.0
    if m % 2 == 0:
        amp[-1] /= 2.0
    freq = np.fft.rfftfreq(m, d=dt)
    return np.asarray(freq, dtype=np.float64), np.asarray(amp, dtype=np.float64)


def to_db(amplitude: npt.ArrayLike) -> FloatArray:
    """``20 log10 |amplitude|``. A zero amplitude becomes NaN, a gap in the curve.

    -inf would be dropped by matplotlib anyway; NaN says so explicitly, and the
    curve hover already treats NaN as a dropout.
    """
    amp = np.abs(np.asarray(amplitude, dtype=np.float64))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(amp)
    db[~np.isfinite(db)] = np.nan
    return np.asarray(db, dtype=np.float64)
