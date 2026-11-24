# Copyright (c) 2026 Santiago Hofwimmer
"""The TeX export of a grouped plot is grouped, as the figure is.

The reported bug: a Bode plot grouped by temperature in the GUI exported its
curves coloured by run index, with no group legend. The overlay TeX kept its
own copy of the figure's colour rules, which never learned grouping; the
histogram's TeX ignored the grouping too. Both now share the figure's policy
(``overlay_style.OverlayStyle``, ``data_loader.histogram_groups``), and these
tests compare the .tex with the figure drawn from the same data.

Also the Bode phase: captured wrapped to ±180°, it jumped by 360° in the figure
and the .tex; both now draw ``bode.continuous_phase``.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from chipify import export_latex, palette  # noqa: E402
from chipify.bode import continuous_phase  # noqa: E402
from chipify.plot_manager import PRINT_THEME, PlotManager  # noqa: E402
from chipify.uikit.services import transient_loader as tl  # noqa: E402

TEMPS = [-40, 27, 125]
RUNS = [f"{i:06d}" for i in range(6)]
GROUP = dict(zip(RUNS, [t for t in TEMPS for _ in range(2)]))   # 2 runs per temp


def _rgb(c) -> tuple:
    return tuple(round(float(v), 4) for v in tuple(c)[:3])


def _tex_colours(tex: str) -> dict[str, tuple]:
    return {name: tuple(round(float(v), 4) for v in rgb.split(","))
            for name, rgb in re.findall(r"\\definecolor\{(\w+)\}\{rgb\}\{([^}]*)\}", tex)}


def _tex_curves(tex: str) -> dict[str, tuple[str, str]]:
    """``{y column: (colour name, options)}`` for every data curve."""
    return {y: (name, opts) for name, opts, y in re.findall(
        r"\\addplot\+ \[color=(\w+)([^\]]*)\] table \[x=\w+, y=(\w+)\]", tex)}


def _legend(tex: str) -> list[str]:
    return re.findall(r"\\addlegendentry\{(.*)\}", tex)


def _wrapped_phase_amp(f, pole):
    """Three poles: the phase falls past -180°, where the capture wraps it."""
    h = 1000 / ((1 + 1j * f / pole) * (1 + 1j * f / (40 * pole))
                * (1 + 1j * f / (200 * pole)))
    return np.abs(h), np.degrees(np.angle(h))


@pytest.fixture()
def ac_dir(tmp_path):
    adir = tmp_path / "ac"
    adir.mkdir()
    f = np.logspace(0, 10, 200)
    for k, rid in enumerate(RUNS):
        mag, ph = _wrapped_phase_amp(f, 5e6 * (1 + 0.05 * k))
        pd.DataFrame({"frequency": f, "out_mag": mag, "out_phase": ph}).to_csv(
            adir / f"run_{rid}__tb.csv", index=False)
    return adir


@pytest.fixture()
def tran_dir(tmp_path):
    adir = tmp_path / "tran"
    adir.mkdir()
    t = np.linspace(0, 1e-6, 50)
    for k, rid in enumerate(RUNS):
        pd.DataFrame({"time": t, "v(out)": t * 1e6 * (1 + 0.1 * k),
                      "v(in)": np.sin(t * 1e7) * (1 + 0.1 * k)}).to_csv(
            adir / f"run_{rid}__tb.csv", index=False)
    return adir


# ── Continuous phase ──────────────────────────────────────────────────────────

def test_continuous_phase_removes_the_wrap_and_keeps_the_start():
    wrapped = [0.0, -90.0, -170.0, 175.0, 90.0, -20.0]
    assert continuous_phase(wrapped) == pytest.approx(
        [0.0, -90.0, -170.0, -185.0, -270.0, -380.0])
    # An inverting stage starting at +180° stays there.
    assert continuous_phase([180.0, 170.0, -175.0]) == pytest.approx(
        [180.0, 170.0, 185.0])


def test_continuous_phase_steps_over_gaps():
    out = continuous_phase([0.0, -170.0, np.nan, 170.0, 100.0])
    assert np.isnan(out[2])
    assert out[[0, 1, 3, 4]] == pytest.approx([0.0, -170.0, -190.0, -260.0])
    assert continuous_phase([np.nan]).shape == (1,)
    assert continuous_phase([]).shape == (0,)


def test_bode_figure_and_tex_draw_the_same_continuous_phase(tmp_path, ac_dir):
    fig = Figure()
    line_map = PlotManager.draw_bode_plot(fig, FigureCanvasAgg(fig), str(ac_dir),
                                          RUNS, ["out"], theme=PRINT_THEME)
    csv, _tex = export_latex.generate_bode_latex_export(
        str(tmp_path / "latex"), "bode", str(ac_dir), RUNS, ["out"])
    table = pd.read_csv(csv)
    phase_lines = {rid: line for line, (rid, what) in line_map.items()
                   if what == "out (phase)"}
    assert set(phase_lines) == set(RUNS)
    for rid, line in phase_lines.items():
        drawn = np.asarray(line.get_ydata(), dtype=float)
        assert np.abs(np.diff(drawn)).max() < 180.0          # no wrap left
        assert drawn.min() < -200.0                          # it does go past -180
        exported = table[f"r_{rid}_out_phase"].dropna().to_numpy()
        assert exported == pytest.approx(drawn)


# ── Overlays: grouped as the figure ───────────────────────────────────────────

def test_grouped_bode_tex_matches_the_figure(tmp_path, ac_dir):
    """The reported bug: grouped in the GUI, coloured by run index in the .tex."""
    fig = Figure()
    line_map = PlotManager.draw_bode_plot(
        fig, FigureCanvasAgg(fig), str(ac_dir), RUNS, ["out"], theme=PRINT_THEME,
        group_map=GROUP, group_label="temp")
    _csv, tex_path = export_latex.generate_bode_latex_export(
        str(tmp_path / "latex"), "bode", str(ac_dir), RUNS, ["out"],
        group_map=GROUP, group_label="temp")
    tex = tex_path.read_text(encoding="utf-8")
    colours, curves = _tex_colours(tex), _tex_curves(tex)

    # Every curve, magnitude and phase, has the figure's colour.
    for line, (rid, what) in line_map.items():
        suffix = "mag" if "(mag)" in what else "phase"
        name, _opts = curves[f"r_{rid}_out_{suffix}"]
        assert colours[name] == _rgb(line.get_color()), (rid, what)
    # One colour per temperature, not per run.
    assert len({curves[f"r_{rid}_out_mag"][0] for rid in RUNS}) == len(TEMPS)

    fig_legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert fig_legend == ["temp=-40", "temp=27", "temp=125"]
    assert _legend(tex) == fig_legend
    assert "grouped by temp" in tex


def test_grouped_transient_tex_styles_signals_by_line_like_the_figure(
        tmp_path, tran_dir):
    signals = ["v(out)", "v(in)"]
    fig = Figure()
    line_map = PlotManager.draw_transient_plot(
        fig, FigureCanvasAgg(fig), str(tran_dir), RUNS, signals, theme=PRINT_THEME,
        group_map=GROUP, group_label="temp")
    _csv, tex_path = export_latex.generate_transient_latex_export(
        str(tmp_path / "latex"), "tran", str(tran_dir), RUNS, signals,
        group_map=GROUP, group_label="temp")
    tex = tex_path.read_text(encoding="utf-8")
    colours, curves = _tex_colours(tex), _tex_curves(tex)
    dash = {"-": None, "--": "dashed", "-.": "dashdotted", ":": "dotted"}
    tokens = {"v(out)": "v_out", "v(in)": "v_in"}
    for line, (rid, sig) in line_map.items():
        name, opts = curves[f"r_{rid}_{tokens[sig]}"]
        assert colours[name] == _rgb(line.get_color())
        pattern = re.search(r"\b(dashed|dashdotted|dotted)\b", opts)
        assert (pattern and pattern.group(1)) == dash[line.get_linestyle()], (rid, sig)
    assert _legend(tex) == [t.get_text() for t in
                            fig.axes[0].get_legend().get_texts()]
    assert "\\addlegendimage{color=gray, dashed, line width=1pt}" in tex


@pytest.mark.parametrize("generate,adir_fixture,signal", [
    ("generate_transient_latex_export", "tran_dir", "v(out)"),
    ("generate_dc_sweep_latex_export", None, "v(out)"),
    ("generate_bode_latex_export", "ac_dir", "out"),
])
def test_every_overlay_exporter_takes_the_grouping(request, tmp_path, generate,
                                                   adir_fixture, signal):
    if adir_fixture is None:            # a DC sweep: x column "sweep"
        adir = tmp_path / "dc"
        adir.mkdir()
        for k, rid in enumerate(RUNS):
            pd.DataFrame({"sweep": [0.0, 0.9, 1.8],
                          "v(out)": [0.0, 0.5 + k, 1.0 + k]}).to_csv(
                adir / f"run_{rid}__tb.csv", index=False)
    else:
        adir = request.getfixturevalue(adir_fixture)
    _csv, tex_path = getattr(export_latex, generate)(
        str(tmp_path / "latex"), "x", str(adir), RUNS, [signal],
        group_map=GROUP, group_label="temp")
    tex = tex_path.read_text(encoding="utf-8")
    assert _legend(tex) == ["temp=-40", "temp=27", "temp=125"]
    for i in range(len(TEMPS)):
        r, g, b = palette.category_color(i, len(TEMPS))
        assert f"\\definecolor{{cfGrp{i}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}" in tex
    assert "cfRun" not in tex


def test_ungrouped_overlay_tex_stays_ungrouped(tmp_path, tran_dir):
    _csv, tex_path = export_latex.generate_transient_latex_export(
        str(tmp_path / "latex"), "tran", str(tran_dir), RUNS, ["v(out)", "v(in)"])
    tex = tex_path.read_text(encoding="utf-8")
    assert _legend(tex) == ["v(out)", "v(in)"]
    assert "grouped by" not in tex and "cfGrp" not in tex and "dashed" not in tex
    # An empty or unknown group column is no grouping, as in the figure.
    _csv, tex_path = export_latex.generate_transient_latex_export(
        str(tmp_path / "latex"), "tran", str(tran_dir), RUNS, ["v(out)"],
        group_map={}, group_label="temp")
    assert "grouped by" not in tex_path.read_text(encoding="utf-8")


def test_single_signal_legend_swatch_is_the_first_runs_shade(tmp_path, tran_dir):
    """The figure's legend shows viridis(0.1), the first run's colour."""
    _csv, tex_path = export_latex.generate_transient_latex_export(
        str(tmp_path / "latex"), "tran", str(tran_dir), RUNS, ["v(out)"])
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\addlegendimage{color=cfRun0, line width=1pt}" in tex
    assert _tex_colours(tex)["cfRun0"] == _rgb(palette.viridis(0.1))


# ── Histogram ─────────────────────────────────────────────────────────────────

def _hist_stim():
    val = SimpleNamespace(name="gain", vmin=None, vmax=None)
    return SimpleNamespace(
        tests=[SimpleNamespace(tb_path="tb", value_lst=[val], measure={})],
        params={"temp": TEMPS})


def _hist_frame(missing_temp=None):
    rng = np.random.default_rng(7)
    rows = [{"run_id": i, "temp": t, "gain": rng.normal(60 + 0.02 * t, 0.3)}
            for i, t in enumerate(t for t in TEMPS for _ in range(30))]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=1)    # data order shuffled
    if missing_temp is not None:
        df.loc[df["temp"] == missing_temp, "gain"] = np.nan
    df["sim_error"] = "None"
    df["tb__error"] = "None"
    df["tb_overall_pass"] = True
    return df


def _hist_figure(df, fit="Gauss (Normal)"):
    fig = Figure()
    ax = fig.add_subplot(111)
    PlotManager.draw_histogram(fig, ax, FigureCanvasAgg(fig), df, _hist_stim(),
                               "gain", fit, "temp", "Auto", False, "None",
                               theme=PRINT_THEME)
    return ax


def _hist_tex(tmp_path, df, fit="Gauss (Normal)"):
    from chipify import data_loader as dl
    rows = dl.plot_rows(df, _hist_stim(), ["gain"])
    written = export_latex.generate_latex_export(
        "gain", rows["gain"].dropna(), fit, "auto", str(tmp_path),
        group_series=rows["temp"])
    csv, tex = written
    return pd.read_csv(csv), tex.read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", [None, 27])
def test_grouped_histogram_tex_matches_the_figure(tmp_path, missing):
    """Same groups, order, bar heights, colours and fits as the figure, also
    when every run of one temperature lacks the measurement."""
    df = _hist_frame(missing)
    ax = _hist_figure(df)
    table, tex = _hist_tex(tmp_path, df)
    colours = _tex_colours(tex)
    # (column key, colour name) of each histogram, in drawing order.
    bars = re.findall(r"fill=(\w+), fill opacity=0\.5[^\n]*forget plot\]\n"
                      r"\s+table \[x=x_hist_(g\d+)", tex)

    drawn = [t for t in TEMPS if t != missing]
    fig_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    tex_labels = _legend(tex)
    assert len(ax.containers) == len(tex_labels) == len(bars) == len(drawn)
    for k, temp in enumerate(drawn):
        cname, key = bars[k]
        patches = ax.containers[k].patches
        assert fig_labels[k].startswith(f"temp={temp} ")
        assert tex_labels[k].startswith(f"temp={temp} ($\\mu=")
        heights = [p.get_height() for p in patches]
        assert table[f"y_hist_{key}"].dropna().to_numpy()[:-1] == pytest.approx(heights)
        assert colours[cname] == _rgb(patches[0].get_facecolor())
        assert table[f"mu_{key}"].iloc[0] == pytest.approx(
            df.loc[df["temp"] == temp, "gain"].mean())
    assert "grouped by temp" in tex


def test_grouped_histogram_tex_without_fit_has_no_curves(tmp_path):
    table, tex = _hist_tex(tmp_path, _hist_frame(), fit="None")
    assert _legend(tex) == ["temp=-40", "temp=27", "temp=125"]
    assert not any(c.startswith("x_curve") for c in table.columns)
    assert "forget plot]\n        table [x=x_curve" not in tex


def test_ungrouped_histogram_tex_is_unchanged_without_a_group(tmp_path):
    df = _hist_frame()
    a = export_latex.generate_latex_export("gain", df["gain"], "Gauss (Normal)",
                                           "auto", str(tmp_path / "a"))
    b = export_latex.generate_latex_export("gain", df["gain"], "Gauss (Normal)",
                                           "auto", str(tmp_path / "b"),
                                           group_series=None)
    assert a[1].read_bytes() == b[1].read_bytes()
    assert "grouped by" not in a[1].read_text()


# ── The report path ───────────────────────────────────────────────────────────

def test_report_latex_of_grouped_bode_and_histogram_is_grouped(tmp_path, ac_dir):
    from chipify import report_service as rs
    from chipify.reports import PlotSpec

    df = _hist_frame()
    df = df[df["run_id"] < len(RUNS)].copy()          # the runs with AC data
    df["run_id"] = df["run_id"].astype(str)
    df["temp"] = df["run_id"].map(lambda r: GROUP[tl.pad_run_id(r)])
    df.attrs["analysis_dirs"] = {"ac": str(ac_dir)}
    stim = _hist_stim()
    stim.tests[0].analyses = [SimpleNamespace(kind="ac", signals=["out"])]

    bode = PlotSpec("bode", options={"signals": ["out"], "group": "temp"})
    _fig, latex = rs.render_spec(bode, df, stim, out_root=tmp_path)
    _csv, tex = latex(tmp_path / "rep", "bode_out_by_temp")
    assert _legend(tex.read_text(encoding="utf-8")) == [
        "temp=-40", "temp=27", "temp=125"]

    hist = PlotSpec("histogram", options={"param": "gain", "group": "temp"})
    _fig, latex = rs.render_spec(hist, df, stim, out_root=tmp_path)
    _csv, tex = latex(tmp_path / "rep", "histogram_gain_by_temp")
    labels = _legend(tex.read_text(encoding="utf-8"))
    assert [lab.split(" ")[0] for lab in labels] == [
        "temp=-40", "temp=27", "temp=125"]
