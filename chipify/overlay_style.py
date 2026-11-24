# Copyright (c) 2026 Santiago Hofwimmer
"""
overlay_style.py – Colour, line style and legend of a per-run waveform overlay.

One policy for the Transient / DC sweep / Spectrum / Bode figures
(``plot_manager._OverlayStyle``) and for their pgfplots export
(``export_latex``). The export kept its own copy of the colour rules, and when
the figures learned to group by a swept parameter the copy did not: a grouped
Bode plot exported its curves coloured by run index, with no group legend.

Headless on purpose: ``export_latex`` must not import matplotlib. Each colour
has a *key* (``("group", i)``, ``("signal", j)``, ``("run", k)``,
``("fail", 0)``, ``("neutral", 0)``) that :meth:`OverlayStyle.resolve` turns
into a colour; the TeX writer names its ``\\definecolor`` entries after the key.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from chipify import data_loader as _dl
from chipify.palette import category_color, viridis

#: Line styles (matplotlib notation) cycled per signal when colour is already
#: spoken for by a group.
GROUP_LINESTYLES = ("-", "--", "-.", ":")

#: A failing run, ungrouped.
FAIL_HEX = "#e74c3c"

#: A legend entry that stands for a line style, not a colour.
NEUTRAL = "gray"

ColorKey = tuple[str, int]


class OverlayStyle:
    """Colour / line-style / legend policy for a per-run waveform overlay.

    Ungrouped, colour encodes the signal, or the run index when a single
    signal is selected, and a failing run is drawn red. Grouped, colour encodes
    the *group value* so corners can be told apart, line style encodes the
    signal, and pass/fail highlighting is off: red would pull failing curves
    out of the very grouping that was asked for.

    *shade* maps a run's position in ``[0.1, 0.9]`` to its colour: the figure
    passes matplotlib's viridis, the TeX export the headless
    ``palette.viridis``.
    """

    def __init__(self, signals: Iterable[str], run_ids: Iterable[str],
                 group_map: Mapping[str, Any] | None = None,
                 group_label: str = "",
                 pass_map: Mapping[str, bool] | None = None,
                 *, shade: Callable[[float], Any] = viridis) -> None:
        run_ids = list(run_ids)
        self.signals = list(signals)
        self.group_label = group_label or ""
        self.group_map = dict(group_map or {})
        self.grouped = bool(self.group_map and self.group_label)
        # Ignored while grouping: see the class docstring.
        self.pass_map = {} if self.grouped else dict(pass_map or {})
        self.single_signal = len(self.signals) == 1
        self._shade = shade
        self._sig_index = {sig: i for i, sig in enumerate(self.signals)}

        self.group_values: list[Any] = []
        self.runs: list[str] = []
        if self.grouped:
            self.group_values = _dl.sort_group_values(
                {self.group_map[r] for r in run_ids if r in self.group_map}
            )
            self.sig_style = {sig: GROUP_LINESTYLES[i % len(GROUP_LINESTYLES)]
                              for i, sig in enumerate(self.signals)}
        else:
            self.sig_style = {sig: "-" for sig in self.signals}
            if self.single_signal:
                self.runs = sorted(set(run_ids))
        self._group_index = {v: i for i, v in enumerate(self.group_values)}
        self._run_index = {rid: i for i, rid in enumerate(self.runs)}
        self._run_shade = np.linspace(0.1, 0.9, max(1, len(self.runs)))

    def is_fail(self, run_id: str) -> bool:
        return self.pass_map.get(run_id, True) is False if self.pass_map else False

    def color_key(self, run_id: str, sig: str) -> ColorKey:
        """Which colour the curve of *sig* in *run_id* gets."""
        if self.grouped:
            i = self._group_index.get(self.group_map.get(run_id))
            return ("group", i) if i is not None else ("signal", self._sig_index[sig])
        if self.is_fail(run_id):
            return ("fail", 0)
        if self.single_signal:
            k = self._run_index.get(run_id)
            return ("run", k) if k is not None else ("signal", 0)
        return ("signal", self._sig_index[sig])

    def resolve(self, key: ColorKey) -> Any:
        """The colour of *key*: an RGB(A) tuple, or a matplotlib colour name."""
        role, i = key
        if role == "group":
            return category_color(i, len(self.group_values))
        if role == "signal":
            return category_color(i, len(self.signals))
        if role == "run":
            return self._shade(float(self._run_shade[i]))
        if role == "fail":
            return FAIL_HEX
        return NEUTRAL

    def color(self, run_id: str, sig: str) -> Any:
        return self.resolve(self.color_key(run_id, sig))

    def linestyle(self, sig: str) -> str:
        return self.sig_style.get(sig, "-")

    def legend_entries(self) -> list[tuple[ColorKey, str, str]]:
        """``(colour key, line style, label)`` per legend line, in order."""
        entries: list[tuple[ColorKey, str, str]] = []
        if self.grouped:
            for i, val in enumerate(self.group_values):
                entries.append((("group", i), "-", f"{self.group_label}={val}"))
            if len(self.signals) > 1:
                for sig in self.signals:
                    entries.append((("neutral", 0), self.sig_style[sig], sig))
            return entries

        if self.single_signal:
            # The first run's shade, viridis(0.1).
            entries.append((("run", 0), "-",
                            f"{self.signals[0]}  (color = run index)"))
        else:
            for j, sig in enumerate(self.signals):
                entries.append((("signal", j), "-", sig))
        if self.pass_map and any(v is False for v in self.pass_map.values()):
            entries.append((("fail", 0), "-", "Failing run"))
        return entries
