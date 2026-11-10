# Copyright (c) 2026 Santiago Hofwimmer
"""The FFT (chipify.spectrum) and the Spectrum overlay that draws it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from chipify import spectrum
from chipify.plot_manager import WAVEFORM_DRAWERS, PlotManager
from chipify.uikit.services import transient_loader as _tl

F0 = 1e3          # the test tone, Hz
N = 1000          # time steps, and so the FFT length
T = 10e-3         # simulated time: ten whole periods, so the tone is coherent
AMP = 0.5


def _uniform_time(steps: int = N, t_stop: float = T) -> np.ndarray:
    """Evenly stepped time as a simulator writes it, tstart and tstop included."""
    return np.linspace(0.0, t_stop, steps + 1)


def _tone(t: np.ndarray, amp: float = AMP, f0: float = F0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * f0 * t)


# ── chipify.spectrum ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("window", spectrum.WINDOWS)
def test_a_coherent_tone_reads_its_amplitude_at_its_frequency(window):
    t = _uniform_time()
    freq, amp = spectrum.amplitude_spectrum(t, _tone(t), window)
    k = int(np.argmax(amp))
    assert freq[k] == pytest.approx(F0)
    assert amp[k] == pytest.approx(AMP, rel=1e-9)


def test_the_frequency_axis_runs_from_dc_to_nyquist():
    t = _uniform_time()
    freq, amp = spectrum.amplitude_spectrum(t, _tone(t))
    assert freq.size == amp.size == N // 2 + 1
    assert freq[0] == 0.0
    assert freq[1] == pytest.approx(1 / T)                # resolution 1/record
    assert freq[-1] == pytest.approx(N / (2 * T))         # Nyquist


@pytest.mark.parametrize("window", spectrum.WINDOWS)
def test_dc_is_kept_as_the_mean(window):
    t = _uniform_time()
    freq, amp = spectrum.amplitude_spectrum(t, 0.9 + _tone(t), window)
    assert amp[0] == pytest.approx(0.9, rel=1e-9)


def test_remove_dc_takes_out_the_bias_point_and_its_window_leakage():
    """A bias point, smeared by the window, swamps the lowest bins."""
    t = _uniform_time()
    y = 0.9 + _tone(t)
    _, kept = spectrum.amplitude_spectrum(t, y, "blackman_harris")
    _, removed = spectrum.amplitude_spectrum(t, y, "blackman_harris", remove_dc=True)
    assert kept[1] > 1.0                          # DC leaking into bin 1
    assert removed[0] < 1e-12 and removed[1] < 1e-12
    assert removed[10] == pytest.approx(AMP, rel=1e-9)   # the tone is untouched


def test_remove_dc_takes_the_mean_of_the_resampled_waveform():
    """The mean of the simulator's own samples leans toward its dense steps."""
    t = T * np.linspace(0.0, 1.0, 4 * N) ** 2     # dense at the start
    y = 0.9 + _tone(t)
    assert abs(y.mean() - 0.9) > 0.03              # the samples' own mean is off
    _, amp = spectrum.amplitude_spectrum(t, y, remove_dc=True)
    assert amp[0] < 1e-5                           # what is left is interpolation


def test_the_nyquist_bin_is_not_doubled():
    """Only the bins with a negative-frequency twin carry twice the energy."""
    t = _uniform_time()
    alternating = AMP * np.cos(np.pi * np.arange(N + 1))  # a tone at exactly Nyquist
    _, amp = spectrum.amplitude_spectrum(t, alternating, "rectangular")
    assert amp[-1] == pytest.approx(AMP, rel=1e-9)


def test_whole_simulated_periods_fill_the_record_exactly():
    """A simulator writes tstart and tstop, and the end repeats the start.

    With it in the grid the record was one step too long, so even a whole
    number of periods leaked: on a real ngspice run, -60 dB beside the tone
    with the rectangular window and a -90 dB third harmonic read as -79.5 dB.
    """
    t = _uniform_time()
    h3 = 10 ** (-90 / 20)
    y = _tone(t) + h3 * np.sin(2 * np.pi * 3 * F0 * t)
    freq, amp = spectrum.amplitude_spectrum(t, y, "rectangular")
    k = int(np.argmax(amp))
    assert freq[k] == F0
    assert amp[k - 1] < 1e-12 and amp[k + 1] < 1e-12      # no leakage at all
    assert amp[3 * k] == pytest.approx(h3, rel=1e-6)


def test_simulator_time_steps_are_resampled_before_the_fft():
    """SPICE writes non-uniform steps. Treated as evenly spaced they smear the
    tone over many bins; resampled first, it reads its true amplitude."""
    u = np.linspace(0.0, 1.0, 4 * N)
    t = T * u ** 2                       # dense at the start, sparse at the end
    y = _tone(t)

    freq, amp = spectrum.amplitude_spectrum(t, y)
    k = int(np.argmax(amp))
    assert abs(freq[k] - F0) <= freq[1]
    assert amp[k] == pytest.approx(AMP, rel=0.02)

    w = spectrum.window_values("hann", y.size)
    naive = 2 * np.abs(np.fft.rfft(y * w)) / w.sum()
    assert naive.max() < 0.5 * AMP       # what skipping the resampling gives


def test_repeated_time_points_and_non_finite_samples_are_dropped():
    t = _uniform_time()
    y = _tone(t)
    clean = spectrum.amplitude_spectrum(t, y)

    # A breakpoint written twice, a NaN sample, out-of-order rows.
    t_dirty = np.concatenate([t[:5], t[4:5], [np.nan], t[5:]])[::-1]
    y_dirty = np.concatenate([y[:5], y[4:5], [1.0], y[5:]])[::-1]
    dirty = spectrum.amplitude_spectrum(t_dirty, y_dirty)
    np.testing.assert_allclose(dirty[0], clean[0])
    np.testing.assert_allclose(dirty[1], clean[1], atol=1e-12)


def test_windows_are_named_case_insensitively_and_unknown_ones_rejected():
    assert spectrum.check_window("Hann") == "hann"
    assert spectrum.check_window(" Blackman_Harris ") == "blackman_harris"
    assert spectrum.DEFAULT_WINDOW == spectrum.WINDOWS[0] == "hann"
    assert set(spectrum.WINDOW_LABELS) == set(spectrum.WINDOWS)
    with pytest.raises(ValueError, match="expected one of hann, blackman_harris"):
        spectrum.check_window("kaiser")
    with pytest.raises(ValueError, match="unknown window"):
        spectrum.amplitude_spectrum(_uniform_time(), _tone(_uniform_time()), "")


def test_too_few_time_points_raise():
    with pytest.raises(ValueError, match="at least 4 time points, got 3"):
        spectrum.amplitude_spectrum([0, 1, 2, 2, np.nan], [0, 1, 0, 5, 1])
    with pytest.raises(ValueError, match="differ in length"):
        spectrum.amplitude_spectrum([0, 1, 2, 3], [0, 1, 2])


def test_to_db_turns_a_zero_into_a_gap_not_minus_infinity():
    db = spectrum.to_db([1.0, 0.1, 0.0, -0.01])
    np.testing.assert_allclose(db[[0, 1, 3]], [0.0, -20.0, -40.0])
    assert np.isnan(db[2])


# ── The Spectrum overlay ──────────────────────────────────────────────────────

def _fig():
    fig = Figure(figsize=(6, 4))
    return fig, FigureCanvasAgg(fig)


def _write_runs(adir, frames) -> str:
    for run_id, frame in frames.items():
        frame.to_csv(adir / f"run_{run_id}__tb.csv", index=False)
    return str(adir)


def _two_runs(tmp_path) -> str:
    t = _uniform_time()
    return _write_runs(tmp_path, {
        "000000": pd.DataFrame({"time": t, "v(outp)": _tone(t), "v(outn)": -_tone(t)}),
        "000001": pd.DataFrame({"time": t, "v(outp)": _tone(t, 0.25),
                                "v(outn)": -_tone(t, 0.25)}),
    })


def test_the_spectrum_view_reads_the_transient_data():
    assert _tl.KIND_LABELS["Spectrum"] == "spectrum"
    assert _tl.kind_for_label("Spectrum") == "transient"
    assert _tl.view_for_label("Spectrum") == "spectrum"
    assert _tl.kind_for_label("Bode") == _tl.view_for_label("Bode") == "ac"
    # Every view the GUI offers has a plotter and an analysis kind.
    assert set(_tl.KIND_LABELS.values()) == set(WAVEFORM_DRAWERS) == set(_tl.VIEW_KINDS)
    assert WAVEFORM_DRAWERS["spectrum"] is PlotManager.draw_spectrum_plot


def test_the_overlay_draws_each_curve_in_db_over_frequency(tmp_path):
    adir = _two_runs(tmp_path)
    fig, canvas = _fig()
    line_map = PlotManager.draw_spectrum_plot(
        fig, canvas, adir, ["000000", "000001"], ["v(outp)", "v(outn)"])

    # The curve hover reads this map, so every drawn curve must be in it.
    assert set(line_map.values()) == {(r, s) for r in ("000000", "000001")
                                      for s in ("v(outp)", "v(outn)")}
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Amplitude (dB)"
    assert "Spectrum (Hann window)" in ax.get_title()
    # Log frequency, as on the Bode plot; a log axis has no 0 Hz, so the curve
    # starts at the first bin above DC.
    assert ax.get_xscale() == "log"
    assert all(line.get_xdata()[0] == pytest.approx(1 / T) for line in line_map)

    peaks = {line_map[line]: (line.get_xdata()[np.nanargmax(line.get_ydata())],
                              np.nanmax(line.get_ydata()))
             for line in line_map}
    assert peaks[("000000", "v(outp)")] == pytest.approx((F0, 20 * np.log10(0.5)))
    assert peaks[("000001", "v(outn)")] == pytest.approx((F0, 20 * np.log10(0.25)))


def test_the_window_is_a_parameter_and_an_unknown_one_raises(tmp_path):
    adir = _two_runs(tmp_path)
    fig, canvas = _fig()
    PlotManager.draw_spectrum_plot(fig, canvas, adir, ["000000"], ["v(outp)"],
                                   window="blackman_harris")
    assert "Blackman-Harris window" in fig.axes[0].get_title()

    # The window shapes the drawn curve, not just the title: Hann spreads a
    # coherent tone into the next bin at half its amplitude, no window does not.
    next_bin = {}
    for window in ("hann", "rectangular"):
        (line,) = PlotManager.draw_spectrum_plot(
            fig, canvas, adir, ["000000"], ["v(outp)"], window=window)
        k = int(np.nanargmax(line.get_ydata()))
        next_bin[window] = line.get_ydata()[k + 1]
    assert next_bin["hann"] == pytest.approx(20 * np.log10(AMP / 2))
    assert not next_bin["rectangular"] > -200          # numerically empty, or NaN

    with pytest.raises(ValueError, match="unknown window"):
        PlotManager.draw_spectrum_plot(fig, canvas, adir, ["000000"], ["v(outp)"],
                                       window="kaiser")


def test_the_overlay_hides_neither_tone_nor_floor_under_the_bias_point(tmp_path):
    adir = _write_runs(tmp_path, {
        "000000": pd.DataFrame({"time": _uniform_time(),
                                "v(out)": 0.9 + _tone(_uniform_time())})})
    fig, canvas = _fig()
    (line,) = PlotManager.draw_spectrum_plot(
        fig, canvas, adir, ["000000"], ["v(out)"], window="blackman_harris")
    y = line.get_ydata()
    assert np.nanmax(y) == pytest.approx(20 * np.log10(AMP))
    assert not y[0] > -200                         # bin 1: no DC leakage


def test_transient_equations_have_a_spectrum_too(tmp_path):
    """A differential output is often an equation, not a captured signal."""
    adir = _two_runs(tmp_path)
    fig, canvas = _fig()
    line_map = PlotManager.draw_spectrum_plot(
        fig, canvas, adir, ["000000"], ["vdiff"],
        equations=[{"name": "vdiff", "expr": "v(outp) - v(outn)"}])
    (line,) = line_map
    assert np.nanmax(line.get_ydata()) == pytest.approx(20 * np.log10(2 * AMP))


def test_a_waveform_too_short_for_an_fft_explains_itself(tmp_path):
    adir = _write_runs(tmp_path, {
        "000000": pd.DataFrame({"time": [0.0, 1e-6, 2e-6], "v(out)": [0, 1, 0]})})
    fig, canvas = _fig()
    assert PlotManager.draw_spectrum_plot(
        fig, canvas, adir, ["000000"], ["v(out)"]) == {}
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "Could not draw any curve" in texts and "at least 4 time points" in texts


def test_the_empty_state_points_at_transient_signals():
    fig, canvas = _fig()
    assert PlotManager.draw_spectrum_plot(fig, canvas, "", [], []) == {}
    assert "transient_signals" in fig.axes[0].texts[0].get_text()
