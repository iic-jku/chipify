# Copyright (c) 2026 Santiago Hofwimmer
"""
bode.py – The phase curve of a Bode plot, shared by the figure and the TeX export.

Headless (numpy only), like ``spectrum``: ``plot_manager.draw_bode_plot`` and
``export_latex.generate_bode_latex_export`` both draw the captured
``<sig>_phase`` column through :func:`continuous_phase`, so the figure and the
.tex show the same curve.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def continuous_phase(phase_deg: Any) -> npt.NDArray[np.float64]:
    """*phase_deg* (degrees) without the jumps of ±360° at the ±180° boundary.

    The captured phase is wrapped: ngspice's ``ph()`` and ``numpy.angle`` on
    the VACASK path both return the principal value. An amplifier whose phase
    falls past -180° therefore jumped to +180° and fell again, a vertical line
    through the plot. Unwrapped from the first point, so the curve starts where
    the captured one starts; the stored CSVs stay as captured.

    Points that are not finite are skipped and kept as they are, so a gap does
    not shift the rest of the curve.
    """
    phase = np.array(phase_deg, dtype=float)
    finite = np.isfinite(phase)
    if finite.sum() > 1:
        phase[finite] = np.unwrap(phase[finite], period=360.0)
    return phase
