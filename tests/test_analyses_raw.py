# Copyright (c) 2026 Santiago Hofwimmer
"""Regression tests for analyses.write_tab_from_raw (VACASK .raw path)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
analyses = pytest.importorskip("chipify.analyses")


def test_ac_write_tab_freq_fallback_with_numpy_arrays(tmp_path) -> None:
    """A bucket without the __x__ sentinel falls back to the 'frequency' /
    'freq' columns. Regression: the fallback used `or`-chaining, whose truth
    test raises ValueError on numpy arrays longer than one element."""
    an = analyses.ACAnalysis(signals=["v(out)"])
    freq = np.array([1.0, 10.0, 100.0])
    sig = np.array([1 + 1j, 0.5 + 0.5j, 0.1 + 0.1j])

    out = tmp_path / "ac.tab"
    an.write_tab_from_raw({"frequency": freq, "v(out)": sig}, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3          # one row per frequency point
    assert len(lines[0].split()) == 3  # freq, mag, phase

    # And the 'freq' spelling works too.
    out2 = tmp_path / "ac2.tab"
    an.write_tab_from_raw({"freq": freq, "v(out)": sig}, str(out2))
    assert len(out2.read_text(encoding="utf-8").strip().splitlines()) == 3


@pytest.mark.parametrize("cls, x_key", [
    (analyses.TransientAnalysis, "time"),
    (analyses.DCSweepAnalysis, "__x__"),
    (analyses.ACAnalysis, "frequency"),
])
def test_missing_signal_raises_instead_of_writing_zeros(tmp_path, cls, x_key) -> None:
    """A bucket holding only the X vector used to become a tab of zeros for
    every declared signal, persisted as an apparently successful waveform."""
    an = cls(signals=["v(out)"])
    out = tmp_path / "x.tab"
    with pytest.raises(analyses.CaptureError, match=r"v\(out\)"):
        an.write_tab_from_raw({x_key: np.array([1.0, 2.0, 3.0])}, str(out))
    assert not out.exists()


def test_partially_missing_signals_name_the_missing_and_available(tmp_path) -> None:
    an = analyses.TransientAnalysis(signals=["v(out)", "v(typo)"])
    x = np.array([0.0, 1.0])
    out = tmp_path / "tran.tab"
    with pytest.raises(analyses.CaptureError) as info:
        an.write_tab_from_raw({"__x__": x, "time": x, "v(out)": x}, str(out))
    msg = str(info.value)
    assert "['v(typo)']" in msg            # the culprit
    assert "'v(out)'" in msg               # what the .raw does carry
    assert "__x__" not in msg              # internal sentinel stays internal
    assert not out.exists()
