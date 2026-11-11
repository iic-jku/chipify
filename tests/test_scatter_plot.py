# Copyright (c) 2026 Santiago Hofwimmer
"""The scatter (shmoo) plot: grouping, its pgfplots export, and the axis list.

The figure and the LaTeX file must agree point for point (same rows, same
verdicts, same group colours), because a user exports the plot they see.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from chipify import data_loader as dl
from chipify import export_latex
from chipify import plot_manager
from chipify.plot_manager import PlotManager


def _stim():
    val = lambda name: SimpleNamespace(name=name)  # noqa: E731
    return SimpleNamespace(
        tests=[SimpleNamespace(tb_path="tb", value_lst=[val("gain"), val("pm")],
                               measure={"slew": "x"})],
        params={"temp": [-40, 27, 100], "corner": ["tt", "ss"], "vdd": [1.8]},
    )


def _frame() -> pd.DataFrame:
    """Six runs over three temperatures; run 4 fails its testbench's spec."""
    return pd.DataFrame({
        "run_id": [0, 1, 2, 3, 4, 5],
        "temp": [27, -40, 100, 27, -40, 100],
        "corner": ["tt", "tt", "tt", "ss", "ss", "ss"],
        "vdd": [1.8] * 6,
        "sim_error": ["None"] * 6,
        "tb__error": ["None"] * 6,
        "tb_overall_pass": [True, True, True, True, False, True],
        "gain": [10.0, 10.2, 9.8, 10.1, 8.0, 9.9],
        "pm": [55.0, 60.0, 52.0, 58.0, 40.0, 57.0],
        "slew": [1.0, 1.1, 0.9, 1.0, 0.8, 1.05],
        "gain2": [20.0, 20.4, 19.6, 20.2, 16.0, 19.8],
    })


def _scatter(df, x="gain", y="pm", group=None):
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    sc, rows = PlotManager.draw_adv_plot(
        fig, None, canvas, df, _stim(), "Scatter Plot", x, y, "-",
        theme=plot_manager.PRINT_THEME, group_col=group)
    return fig, sc, rows


def _legend_labels(fig):
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


# ── Figure ────────────────────────────────────────────────────────────────────

def test_ungrouped_scatter_is_unchanged_green_and_red():
    fig, sc, rows = _scatter(_frame())
    faces = sc.get_facecolors()
    assert len(sc.get_paths()) == 1                       # one marker for all
    green, red = to_rgba(plot_manager._PASS_COLOR), to_rgba(plot_manager._FAIL_COLOR)
    assert faces[4][:3] == pytest.approx(red[:3])        # the failing run
    assert faces[0][:3] == pytest.approx(green[:3])
    assert _legend_labels(fig) == ["Pass", "Fail"]
    assert "grouped" not in fig.axes[0].get_title()


def test_grouped_scatter_colours_by_value_and_marks_failures_with_crosses():
    fig, sc, rows = _scatter(_frame(), group="temp")
    faces = sc.get_facecolors()
    # Sorted numerically (-40, 27, 100), tab10 in that order.
    order = {-40: 0, 27: 1, 100: 2}
    for i, temp in enumerate(rows["temp"]):
        assert faces[i][:3] == pytest.approx(plt.cm.tab10(order[temp])[:3], abs=1e-6)
    # One path per point, the failing one a different (cross) marker.
    paths = sc.get_paths()
    assert len(paths) == len(rows)
    verts = [p.vertices.tobytes() for p in paths]
    assert verts[4] != verts[0]
    assert len({v for i, v in enumerate(verts) if i != 4}) == 1
    assert _legend_labels(fig) == ["temp=-40", "temp=27", "temp=100", "Fail"]
    assert fig.axes[0].get_title().endswith("grouped by temp")


def test_grouped_scatter_without_failures_keeps_one_marker_and_no_fail_entry():
    df = _frame()
    df["tb_overall_pass"] = True
    fig, sc, _rows = _scatter(df, group="temp")
    assert len(sc.get_paths()) == 1
    assert _legend_labels(fig) == ["temp=-40", "temp=27", "temp=100"]


@pytest.mark.parametrize("group", [None, "", "None", "not_a_column"])
def test_no_usable_group_draws_the_ungrouped_scatter(group):
    fig, _sc, _rows = _scatter(_frame(), group=group)
    assert _legend_labels(fig) == ["Pass", "Fail"]


def test_hover_rows_stay_aligned_with_the_grouped_points():
    """The hover maps a point index to a row of the returned frame."""
    _fig, sc, rows = _scatter(_frame(), group="temp")
    offsets = sc.get_offsets()
    assert len(offsets) == len(rows)
    assert offsets[4][0] == pytest.approx(rows["gain"].iloc[4])


# ── LaTeX export ──────────────────────────────────────────────────────────────

def _export(tmp_path, rows, x="gain", y="pm", group=""):
    csv_path, tex_path = export_latex.generate_scatter_latex_export(
        str(tmp_path), "sc", rows, x, y, group)
    return pd.read_csv(csv_path), tex_path.read_text(encoding="utf-8")


def test_scatter_tex_colours_match_the_figure():
    assert export_latex.SCATTER_PASS_HEX == plot_manager._PASS_COLOR
    assert export_latex.SCATTER_FAIL_HEX == plot_manager._FAIL_COLOR
    from chipify import palette
    for i, rgb in enumerate(palette.TAB10):
        assert rgb == pytest.approx(plt.cm.tab10(i)[:3], abs=1e-6)


def test_ungrouped_export_splits_points_by_the_figures_verdict(tmp_path):
    _fig, _sc, rows = _scatter(_frame())
    table, tex = _export(tmp_path, rows)
    assert table["x_pass"].count() == 5 and table["x_fail"].count() == 1
    assert table["x_fail"].iloc[0] == pytest.approx(8.0)
    assert table["y_fail"].iloc[0] == pytest.approx(40.0)
    assert "\\addlegendentry{Pass}" in tex and "\\addlegendentry{Fail}" in tex
    assert "table [x=x_fail, y=y_fail]" in tex
    assert "forget plot" in tex                 # the legend comes from images only


def test_grouped_export_has_one_series_per_group_and_verdict(tmp_path):
    _fig, _sc, rows = _scatter(_frame(), group="temp")
    table, tex = _export(tmp_path, rows, group="temp")
    # -40 is group 0 (runs 1 and 4, run 4 failing), 27 group 1, 100 group 2.
    assert table["x_g0_pass"].count() == 1 and table["x_g0_fail"].count() == 1
    assert table["x_g1_pass"].count() == 2 and "x_g1_fail" not in table
    assert table["x_g2_pass"].count() == 2
    entries = [line.split("{", 1)[1].rstrip("}") for line in tex.splitlines()
               if line.strip().startswith("\\addlegendentry")]
    assert entries == ["temp=-40", "temp=27", "temp=100", "Fail"]
    r, g, b = plt.cm.tab10(0)[:3]
    assert f"\\definecolor{{cfGrp0}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}" in tex
    fail_plot = next(line for line in tex.splitlines() if "y=y_g0_fail" in line)
    assert "mark=x" in fail_plot and "color=cfGrp0" in fail_plot


def test_text_axis_becomes_named_ticks_at_the_figures_positions(tmp_path):
    """matplotlib puts categories at 0, 1, … in order of first appearance."""
    fig, sc, rows = _scatter(_frame(), x="corner", y="gain")
    table, tex = _export(tmp_path, rows, x="corner", y="gain")
    drawn_x = sorted(sc.get_offsets()[:, 0])
    exported_x = sorted(pd.concat([table["x_pass"], table["x_fail"]]).dropna())
    assert exported_x == pytest.approx(drawn_x)
    assert "xtick={0,1}, xticklabels={{tt},{ss}}" in tex


def test_names_are_escaped_for_latex(tmp_path):
    rows = _frame().rename(columns={"gain": "v(out)_max", "pm": "pm%"})
    _table, tex = _export(tmp_path, rows, x="v(out)_max", y="pm%")
    assert "xlabel={v(out)\\_max}" in tex and "ylabel={pm\\%}" in tex


@pytest.mark.parametrize("rows,x", [(None, "gain"), (pd.DataFrame(), "gain"),
                                    (_frame(), "nope")])
def test_export_without_points_raises(tmp_path, rows, x):
    with pytest.raises(ValueError):
        export_latex.generate_scatter_latex_export(str(tmp_path), "sc", rows, x, "pm")


# ── Shared helpers ────────────────────────────────────────────────────────────

def test_scatter_axes_offer_inputs_measurements_and_equation_results():
    cols = dl.scatter_axis_columns(_frame(), _stim(), ["gain2"])
    # Swept inputs first (vdd has one value, so it is no sweep), then the
    # measurements incl. measure: results, then the scalar-equation results.
    assert cols == ["temp", "corner", "gain", "pm", "slew", "gain2"]
    # A derived name the frame does not hold (an equation that failed) is
    # not offered.
    assert "lost" not in dl.scatter_axis_columns(_frame(), _stim(), ["lost"])


def test_group_values_sort_numerically_else_as_text():
    assert dl.sort_group_values([100, -40, 27]) == [-40, 27, 100]
    assert dl.sort_group_values(["tt", "ff", "ss"]) == ["ff", "ss", "tt"]


def test_scatter_pass_mask_counts_a_row_where_nothing_ran_as_failing():
    df = _frame()
    df.loc[2, "tb__error"] = "tb: CRASH"
    mask = dl.scatter_pass_mask(df)
    assert list(mask) == [True, True, False, True, False, True]
    assert mask.dtype == np.bool_
