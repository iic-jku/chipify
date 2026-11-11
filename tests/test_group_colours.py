# Copyright (c) 2026 Santiago Hofwimmer
"""Group and signal colours: distinct beyond ten, sorted, figure equals TeX.

The plots sampled tab10 at ``linspace(0, 1, n)`` and took ``[i % 10]``: for 12
groups that gave 9 colours, the first two groups the same one. The histogram
also legended its groups in data order.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from chipify import export_latex, palette, plot_manager
from chipify.plot_manager import PlotManager, _OverlayStyle

TEMPS = [-40, -25, -10, 0, 10, 27, 40, 55, 70, 85, 100, 125]     # twelve groups


def _rgb(c):
    return tuple(round(float(v), 4) for v in tuple(c)[:3])


def test_up_to_ten_categories_keep_tab10_in_order():
    for n in (1, 3, 10):
        for i in range(n):
            assert palette.category_color(i, n) == pytest.approx(
                plt.cm.tab10(i)[:3], abs=1e-6)


@pytest.mark.parametrize("n", [11, 12, 30])
def test_more_than_ten_categories_all_get_their_own_colour(n):
    colours = {_rgb(palette.category_color(i, n)) for i in range(n)}
    assert len(colours) == n


def test_overlay_groups_and_signals_beyond_ten_are_distinct():
    runs = [f"{i:06d}" for i in range(len(TEMPS))]
    style = _OverlayStyle(["v(out)"], runs, group_map=dict(zip(runs, TEMPS)),
                          group_label="temp")
    assert len({_rgb(style.color(r, "v(out)")) for r in runs}) == 12
    signals = [f"s{i}" for i in range(12)]
    ungrouped = _OverlayStyle(signals, runs)
    assert len({_rgb(ungrouped.color(runs[0], s)) for s in signals}) == 12


def _sweep_frame():
    rows = [{"run_id": i, "temp": t, "gain": 10 + 0.01 * t + 0.001 * k,
             "pm": 50 + 0.05 * t + 0.01 * k}
            for i, (t, k) in enumerate((t, k) for k in range(3) for t in TEMPS)]
    df = pd.DataFrame(rows)
    df["sim_error"] = "None"
    df["tb__error"] = "None"
    df["tb_overall_pass"] = True
    return df


def _stim():
    val = lambda n: SimpleNamespace(name=n, vmin=None, vmax=None)  # noqa: E731
    return SimpleNamespace(
        tests=[SimpleNamespace(tb_path="tb", value_lst=[val("gain"), val("pm")],
                               measure={})],
        params={"temp": TEMPS})


def test_histogram_groups_are_sorted_and_distinct():
    df = _sweep_frame().sample(frac=1.0, random_state=3)   # data order shuffled
    fig = Figure()
    ax = fig.add_subplot(111)
    PlotManager.draw_histogram(fig, ax, FigureCanvasAgg(fig), df, _stim(), "gain",
                               "None", "temp", "Auto", False, "None")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [f"temp={t}" for t in TEMPS]
    # One colour per group: the first patch of each histogram.
    firsts = []
    for container in ax.containers:
        firsts.append(_rgb(container.patches[0].get_facecolor()))
    assert len(set(firsts)) == 12


def test_scatter_beyond_ten_groups_matches_its_tex(tmp_path):
    df = _sweep_frame()
    fig = Figure()
    sc, rows = PlotManager.draw_adv_plot(
        fig, None, FigureCanvasAgg(fig), df, _stim(), "Scatter Plot",
        "gain", "pm", "-", theme=plot_manager.PRINT_THEME, group_col="temp")
    faces = {_rgb(c) for c in sc.get_facecolors()}
    assert len(faces) == 12
    _csv, tex = export_latex.generate_scatter_latex_export(
        str(tmp_path), "sc", rows, "gain", "pm", "temp")
    text = tex.read_text(encoding="utf-8")
    for i in range(12):
        r, g, b = palette.category_color(i, 12)
        assert f"\\definecolor{{cfGrp{i}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}" in text
    exported = {tuple(round(v, 4) for v in palette.category_color(i, 12))
                for i in range(12)}
    assert faces == exported


def test_overlay_tex_signal_colours_follow_the_figure_beyond_ten(tmp_path):
    adir = tmp_path / "tran"
    adir.mkdir()
    signals = [f"s{i}" for i in range(12)]
    t = np.linspace(0, 1e-3, 5)
    pd.DataFrame({"time": t, **{s: t * i for i, s in enumerate(signals)}}).to_csv(
        adir / "run_000000__tb.csv", index=False)
    _csv, tex = export_latex.generate_transient_latex_export(
        str(tmp_path / "latex"), "tran", str(adir), ["000000"], signals)
    text = tex.read_text(encoding="utf-8")
    for i in (0, 11):
        r, g, b = palette.category_color(i, 12)
        assert f"\\definecolor{{cfSig{i}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}" in text
