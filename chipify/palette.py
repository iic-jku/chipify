# Copyright (c) 2026 Santiago Hofwimmer
"""
palette.py – Colours for grouped plots, shared by the figures and the TeX export.

Headless on purpose: ``export_latex`` must give a group the colour the figure
gave it without importing matplotlib, so the palettes are hard-coded here and
``plot_manager`` reads them too, rather than each module keeping its own.
Values are matplotlib's: ``cm.tab10`` at its ten colours and ``cm.viridis``
sampled at 11 stops.
"""
from __future__ import annotations

RGB = tuple[float, float, float]

TAB10: list[RGB] = [
    (0.121569, 0.466667, 0.705882),
    (1.000000, 0.498039, 0.054902),
    (0.172549, 0.627451, 0.172549),
    (0.839216, 0.152941, 0.156863),
    (0.580392, 0.403922, 0.741176),
    (0.549020, 0.337255, 0.294118),
    (0.890196, 0.466667, 0.760784),
    (0.498039, 0.498039, 0.498039),
    (0.737255, 0.741176, 0.133333),
    (0.090196, 0.745098, 0.811765),
]

VIRIDIS_STOPS: list[RGB] = [
    (0.267004, 0.004874, 0.329415),  # 0.0
    (0.282327, 0.094955, 0.417331),  # 0.1
    (0.253935, 0.265254, 0.529983),  # 0.2
    (0.206756, 0.371758, 0.553117),  # 0.3
    (0.163625, 0.471133, 0.558148),  # 0.4
    (0.127568, 0.566949, 0.550556),  # 0.5
    (0.134692, 0.658636, 0.517649),  # 0.6
    (0.266941, 0.748751, 0.440573),  # 0.7
    (0.477504, 0.821444, 0.318195),  # 0.8
    (0.741388, 0.873449, 0.149561),  # 0.9
    (0.993248, 0.906157, 0.143936),  # 1.0
]


def viridis(t: float) -> RGB:
    """Linearly interpolated viridis sample, t in [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    pos = t * (len(VIRIDIS_STOPS) - 1)
    i = int(pos)
    if i >= len(VIRIDIS_STOPS) - 1:
        return VIRIDIS_STOPS[-1]
    frac = pos - i
    r0, g0, b0 = VIRIDIS_STOPS[i]
    r1, g1, b1 = VIRIDIS_STOPS[i + 1]
    return (r0 + frac * (r1 - r0),
            g0 + frac * (g1 - g0),
            b0 + frac * (b1 - b0))


def category_color(i: int, n: int) -> RGB:
    """Colour of category *i* of *n* (a group value, a signal).

    tab10 in order up to ten categories. Beyond that, viridis spaced evenly
    over 0.1 to 0.9, so every category keeps a colour of its own: the plots
    sampled tab10 at ``linspace(0, 1, n)`` and took ``[i % 10]``, which for 12
    temperatures gave 9 colours, the first two groups the same one. Groups are
    sorted, so a gradient also reads in their order.
    """
    if n <= len(TAB10):
        return TAB10[i % len(TAB10)]
    return viridis(0.1 + 0.8 * i / (n - 1))
