# Copyright (c) 2026 Santiago Hofwimmer
"""The datasheet's ``reports:`` block: validation and headless rendering."""
from __future__ import annotations

import pickle
import textwrap

import pandas as pd
import pytest

from chipify import report_service as rs
from chipify.reports import PLOT_TYPES, PlotSpec, ReportsConfig
from chipify.schema import SchemaError, validate_datasheet
from chipify.util import Stimuli

_BASE = """
parameters:
  temp: [-40, 27, 100]
  vdd: [1.8]
tests:
  tb:
    gain: {min: 9, max: 11, unit: dB}
    pm: {min: 45, unit: deg}
"""


def _datasheet(reports_block: str = "") -> dict:
    import yaml
    return yaml.safe_load(_BASE + textwrap.dedent(reports_block))


def _stim(reports_block: str = ""):
    return validate_datasheet(_datasheet(reports_block))


def _results() -> pd.DataFrame:
    return pd.DataFrame({
        "temp": [-40, 27, 100, 27],
        "vdd": [1.8] * 4,
        "run_id": ["0", "1", "2", "3"],
        "sim_error": ["None"] * 4,
        "tb__error": ["None"] * 4,
        "tb_overall_pass": [True] * 4,
        "gain": [10.0, 10.2, 9.8, 10.1],
        "gain_pass": [True] * 4,
        "pm": [55.0, 60.0, 52.0, 58.0],
        "pm_pass": [True] * 4,
    })


# ── Validation ────────────────────────────────────────────────────────────────

def test_absent_block_yields_an_empty_falsey_config():
    """Datasheets that declare nothing must be entirely unaffected."""
    cfg = _stim().reports
    assert isinstance(cfg, ReportsConfig)
    assert not cfg and cfg.plots == []


def test_full_block_parses():
    cfg = _stim("""
        reports:
          formats: [png, svg]
          pdf: true
          markdown: true
          plots:
            - {type: scatter, x: gain, y: pm, formats: [svg], name: g_vs_pm}
            - {type: histogram, param: gain, group: temp}
            - {type: bode, signals: [outp], group: temp, runs: all_valid}
    """).reports
    assert cfg.formats == ["png", "svg"] and cfg.pdf and cfg.markdown
    assert [p.type for p in cfg.plots] == ["scatter", "histogram", "bode"]
    assert cfg.plots[0].formats == ["svg"]           # per-plot override
    assert cfg.plots[1].resolved_formats(cfg.formats) == ["png", "svg"]
    assert cfg.plots[2].options["signals"] == ["outp"]
    assert cfg.plots_per_page == 1                   # default: one per page
    assert _stim("reports:\n  plots_per_page: 2").reports.plots_per_page == 2


@pytest.mark.parametrize("block,fragment", [
    ("reports: [1, 2]", "expected a mapping"),
    ("reports:\n  plots: {}", "expected a list"),
    ("reports:\n  plots: [{type: nope}]", "unknown plot type"),
    ("reports:\n  plots: [{param: gain}]", "missing required key 'type'"),
    ("reports:\n  plots: [{type: scatter, y: pm}]", "requires 'x'"),
    ("reports:\n  plots: [{type: histogram, param: gain, bogus: 1}]", "unknown key"),
    ("reports:\n  formats: [tiff_nope]", "unknown format"),
    ("reports:\n  plots_per_page: 3", "plots_per_page"),
    ("reports:\n  plots_per_page: true", "plots_per_page"),
    ("reports:\n  plots_per_page: '2'", "plots_per_page"),
    ("reports:\n  plots: [{type: spectrum, window: kaiser}]",
     "reports.plots[0].window: unknown window 'kaiser'"),
    ("reports:\n  plots: [{type: transient, window: hann}]", "unknown key 'window'"),
    ("reports:\n  plots: [{type: corner_yield, x: temp, y: vdd, group: temp}]",
     "unknown key 'group'"),
])
def test_invalid_blocks_name_the_offending_path(block, fragment):
    with pytest.raises(SchemaError) as exc:
        _stim("\n" + block)
    assert fragment in str(exc.value)


def test_stimuli_with_reports_still_pickles(tmp_path):
    """Stimuli is sent to worker processes, so it must stay picklable."""
    path = tmp_path / "d.yaml"
    path.write_text(_BASE + textwrap.dedent("""
        reports:
          pdf: true
          plots: [{type: scatter, x: gain, y: pm}]
    """), encoding="utf-8")
    stim = Stimuli(str(path))
    assert pickle.loads(pickle.dumps(stim)).reports.pdf is True


# ── Naming ────────────────────────────────────────────────────────────────────

def test_plot_names_come_from_the_spec_not_its_position():
    """Reordering the list must not rename every file and ruin run-to-run diffs."""
    assert rs.default_plot_name(
        PlotSpec("scatter", options={"x": "gain", "y": "pm"})
    ) == "scatter_gain_vs_pm"
    assert rs.default_plot_name(
        PlotSpec("histogram", options={"param": "gain", "group": "temp"})
    ) == "histogram_gain_by_temp"
    assert rs.default_plot_name(PlotSpec("scatter", name="my plot/name")) == "my_plot_name"


# ── Rendering ─────────────────────────────────────────────────────────────────

_TYPE_OPTIONS = {
    "scatter": {"x": "gain", "y": "pm"},
    "corner_yield": {"x": "temp", "y": "vdd"},
    "correlation": {},
    "tornado": {"target": "gain"},
    "fail_breakdown": {},
    "histogram": {"param": "gain"},
}


@pytest.mark.parametrize("ptype", sorted(_TYPE_OPTIONS))
def test_every_non_waveform_type_renders_headless(ptype, tmp_path):
    """Parametrised over the registry, so a new type cannot land uncovered."""
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["png", "svg"],
        plots=[PlotSpec(ptype, options=dict(_TYPE_OPTIONS[ptype]))],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert not result.warnings, result.warnings
    assert len(result.files) == 2
    for path in result.files:
        assert path.exists() and path.stat().st_size > 0


def test_registry_and_test_matrix_stay_in_step():
    """Every non-waveform plot type must appear in the render matrix above."""
    non_waveform = {n for n, t in PLOT_TYPES.items() if not t.is_waveform}
    assert non_waveform == set(_TYPE_OPTIONS)


def test_latex_on_an_unsupported_type_warns_but_keeps_the_other_formats(tmp_path):
    stim = _stim()
    stim.reports = ReportsConfig(
        plots=[PlotSpec("tornado", formats=["png", "latex"],
                        options={"target": "gain"})],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert [f.suffix for f in result.files] == [".png"]
    assert any("LaTeX" in w and "tornado" in w for w in result.warnings)


def test_grouped_scatter_writes_png_svg_and_latex(tmp_path):
    stim = _stim("""
        reports:
          formats: [png, svg, latex]
          plots: [{type: scatter, x: gain, y: pm, group: temp}]
    """)
    assert stim.reports.plots[0].options["group"] == "temp"
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert not result.warnings, result.warnings
    names = sorted(f.name for f in result.files)
    assert names == ["scatter_gain_vs_pm_by_temp.png",
                     "scatter_gain_vs_pm_by_temp.svg",
                     "scatter_gain_vs_pm_by_temp_plot.csv",
                     "scatter_gain_vs_pm_by_temp_plot.tex"]
    tex = next(f for f in result.files if f.suffix == ".tex").read_text(encoding="utf-8")
    assert "\\addlegendentry{temp=-40}" in tex and "grouped by temp" in tex
    # The images come from this figure; it must be grouped too, not only the TeX.
    fig, _latex = rs.render_spec(stim.reports.plots[0], _results(), stim,
                                 out_root=tmp_path)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend == ["temp=-40", "temp=27", "temp=100"]


def test_scatter_of_a_column_the_run_lacks_warns_instead_of_an_empty_image(tmp_path):
    """The scatter drew nothing for an unknown column, and the report wrote
    the empty figure without a warning."""
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["png", "latex"],
        plots=[PlotSpec("scatter", options={"x": "gain", "y": "gain_x2"})],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert result.files == []
    assert any("gain_x2" in w and "not in this run" in w for w in result.warnings)


def test_histogram_latex_writes_the_pgfplots_pair(tmp_path):
    stim = _stim()
    stim.reports = ReportsConfig(
        plots=[PlotSpec("histogram", formats=["latex"], options={"param": "gain"})],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert sorted(f.suffix for f in result.files) == [".csv", ".tex"]
    assert not result.warnings


def test_one_bad_spec_does_not_stop_the_others(tmp_path):
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["png"],
        plots=[
            PlotSpec("histogram", options={"param": "not_a_measurement"}),
            PlotSpec("scatter", options={"x": "gain", "y": "pm"}),
        ],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert [f.name for f in result.files] == ["scatter_gain_vs_pm.png"]
    assert any("not_a_measurement" in w for w in result.warnings)


def test_plots_sharing_a_file_name_are_both_written(tmp_path):
    """Two histograms of one measurement (different bins) reported two files
    but the second silently overwrote the first."""
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["png"],
        plots=[
            PlotSpec("histogram", options={"param": "gain", "bins": 10}),
            PlotSpec("histogram", options={"param": "gain", "bins": 40}),
            PlotSpec("scatter", name="Gain", options={"x": "gain", "y": "pm"}),
            PlotSpec("scatter", name="gain", options={"x": "pm", "y": "gain"}),
        ],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    names = [f.name for f in result.files]
    assert names == ["histogram_gain.png", "histogram_gain_2.png",
                     "Gain.png", "gain_2.png"]
    assert all(f.exists() for f in result.files)
    assert sum("distinct 'name:'" in w for w in result.warnings) == 2


def test_a_failing_exporter_costs_only_its_own_file(tmp_path, monkeypatch):
    """A plugin exporter that raised escaped generate_reports, so the later
    plots and the PDF were never written."""
    real = rs._exporters()

    class _Broken:
        extension = "svg"

        def export(self, fig, path, theme=None):
            raise RuntimeError("exporter bug")

    monkeypatch.setattr(rs, "_exporters", lambda: {**real, "svg": _Broken})
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["svg", "png"], pdf=True,
        plots=[PlotSpec("histogram", options={"param": "gain"}),
               PlotSpec("scatter", options={"x": "gain", "y": "pm"})],
    )
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert sorted(f.suffix for f in result.files) == [".pdf", ".png", ".png"]
    assert sum("svg export failed (exporter bug)" in w for w in result.warnings) == 2


def test_missing_waveform_directory_warns_rather_than_raising(tmp_path):
    stim = _stim()
    stim.reports = ReportsConfig(
        formats=["png"], plots=[PlotSpec("bode", options={"signals": ["outp"]})])
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    assert result.files == []
    assert len(result.warnings) == 1


def _with_transient_waveforms(tmp_path) -> pd.DataFrame:
    """The results plus a 1 kHz tone per run on disk, as a finished run has them."""
    import numpy as np
    adir = tmp_path / "analysis_data" / "transient" / "20261110_120000"
    adir.mkdir(parents=True)
    t = np.linspace(0.0, 10e-3, 1000, endpoint=False)
    df = _results()
    for rid in df["run_id"]:
        pd.DataFrame({"time": t, "v(out)": 0.5 * np.sin(2 * np.pi * 1e3 * t)}).to_csv(
            adir / f"run_{int(rid):06d}__tb.csv", index=False)
    df.attrs["analysis_dirs"] = {"transient": str(adir)}
    return df


def test_spectrum_plot_renders_from_the_transient_waveforms(tmp_path):
    stim = _stim("""
        reports:
          formats: [png, svg, latex]
          plots:
            - {type: spectrum, signals: [v(out)], window: Blackman_Harris, group: temp}
    """)
    assert stim.reports.plots[0].plot_type.kind == "transient"
    result = rs.generate_reports(_with_transient_waveforms(tmp_path), stim,
                                 "d.yaml", tmp_path)
    assert sorted(f.name for f in result.files) == [
        "spectrum_v_out_by_temp.png", "spectrum_v_out_by_temp.svg"]
    assert all(f.stat().st_size > 0 for f in result.files)
    # No pgfplots generator: the TeX overlay exporters would write the captured
    # time-domain CSVs under the spectrum's name.
    assert len(result.warnings) == 1, result.warnings
    assert "LaTeX" in result.warnings[0] and "'spectrum'" in result.warnings[0]


def test_spectrum_page_in_the_pdf_uses_the_datasheet_window(tmp_path):
    stim = _stim("""
        reports:
          pdf: true
          plots: [{type: spectrum, signals: [v(out)], window: rectangular}]
    """)
    df = _dl_prepared()
    df.attrs = _with_transient_waveforms(tmp_path).attrs
    (page,) = _pdf_pages(stim, df, tmp_path)
    assert any(t.startswith("Spectrum (Rectangular window)")
               for t in page["titles"]), page
    assert page["artists"] >= 4              # one curve per run


def test_documents_and_latest_pointer(tmp_path):
    stim = _stim()
    stim.reports = ReportsConfig(pdf=True, markdown=True)
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path,
                                 duration_s=1.25)
    assert not result.warnings, result.warnings
    assert {f.suffix for f in result.files} == {".pdf", ".md"}

    # Timestamped directory, with .latest naming it (the analysis_data pattern).
    assert result.out_dir.parent == rs.reports_dir(tmp_path)
    assert rs.latest_reports(tmp_path) == str(result.out_dir)


def test_empty_config_reports_that_there_is_nothing_to_do(tmp_path):
    result = rs.generate_reports(_results(), _stim(), "d.yaml", tmp_path)
    assert result.files == []
    assert "no 'reports:' block" in result.warnings[0]
    # Nothing was created for a datasheet that asked for nothing.
    assert not result.out_dir.exists()


# ── Config override (the unified PDF path) ────────────────────────────────────

def test_config_override_ignores_the_datasheet_block(tmp_path):
    """How the one-click PDF reuses this path instead of having its own."""
    stim = _stim("""
        reports:
          formats: [png]
          plots: [{type: scatter, x: gain, y: pm}]
    """)
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path,
                                 config=rs.pdf_only_config())
    assert [f.suffix for f in result.files] == [".pdf"]
    assert not result.warnings


def test_pdf_only_config_works_without_any_reports_block(tmp_path):
    """A datasheet that declares nothing can still produce a PDF."""
    result = rs.generate_reports(_results(), _stim(), "d.yaml", tmp_path,
                                 config=rs.pdf_only_config())
    assert [f.suffix for f in result.files] == [".pdf"]
    # And it lands in the timestamped folder like everything else, rather than
    # loose in out/reports/ as the old export_pdf path did.
    assert result.out_dir.parent == rs.reports_dir(tmp_path)


# ── runs: shares the GUI's vocabulary ─────────────────────────────────────────

@pytest.mark.parametrize("runs", ["all_valid", "failing", "first:5", "All Valid"])
def test_runs_accepts_the_canonical_tokens(runs):
    cfg = _stim(f"""
        reports:
          plots: [{{type: bode, runs: {runs}}}]
    """).reports
    assert cfg.plots[0].options["runs"] == runs


def test_unknown_runs_value_is_rejected():
    from chipify.schema import SchemaError
    with pytest.raises(SchemaError) as exc:
        _stim("""
            reports:
              plots: [{type: bode, runs: whenever}]
        """)
    assert "all_valid" in str(exc.value)


# ── One histogram, every format ───────────────────────────────────────────────

def test_histogram_options_come_from_one_place():
    """The PNG and the PDF page must resolve identical settings for a spec."""
    from chipify.reports import histogram_options, histogram_spec_for

    stim = _stim("""
        reports:
          plots: [{type: histogram, param: gain, group: temp, bins: '20'}]
    """)
    spec = histogram_spec_for(stim, "gain")
    assert spec is not None
    opts = histogram_options(spec)
    assert opts["group"] == "temp" and opts["bins"] == "20"
    # Unset keys fall back to the shared defaults, not to per-renderer ones.
    assert opts["fit"] == "Gauss (Normal)"
    assert opts["zoom"] is True


def test_report_histograms_zoom_to_the_data_by_default():
    """A spec far wider than the spread must not collapse into slivers.

    A +/-10 mV spec on a 60 uV distribution rendered every bar as a sliver
    because the axis spanned the spec, not the data.
    """
    from chipify.reports import histogram_options

    assert histogram_options(None)["zoom"] is True
    assert histogram_options(
        PlotSpec("histogram", options={"param": "gain", "zoom": False})
    )["zoom"] is False


def test_histogram_spec_for_only_matches_its_own_measurement():
    from chipify.reports import histogram_spec_for

    stim = _stim("""
        reports:
          plots:
            - {type: histogram, param: gain, group: temp}
            - {type: scatter, x: gain, y: pm}
    """)
    assert histogram_spec_for(stim, "gain").options["group"] == "temp"
    assert histogram_spec_for(stim, "pm") is None          # the scatter is not one
    assert histogram_spec_for(_stim(), "gain") is None     # no block at all


def test_pdf_page_and_standalone_png_render_the_same_grouping(tmp_path):
    """The reported bug: the PNG was grouped by temp and the PDF was not."""
    import matplotlib
    matplotlib.use("Agg")
    from chipify import pdf_export
    from chipify.reports import histogram_options, histogram_spec_for
    from chipify.uikit.services import measurements as meas

    stim = _stim("""
        reports:
          pdf: true
          formats: [png]
          plots: [{type: histogram, param: gain, group: temp}]
    """)
    df = _dl_prepared()
    captured = {}

    class _Stub:
        def savefig(self, fig):
            # The page carries header/title axes too; take the one that plotted.
            captured["ax"] = next(a for a in fig.axes if a.get_legend() is not None)

    pdf_export._add_histograms(_Stub(), df, stim, meas.measurement_rows(df, stim))
    labels = [t.get_text() for t in captured["ax"].get_legend().get_texts()]
    # One legend entry per temperature: the PDF honours the datasheet's grouping.
    assert any(l.startswith("temp=") for l in labels), labels

    # And it is zoomed to the data, not to the far-away spec limits.
    lo, hi = captured["ax"].get_xlim()
    assert hi - lo < 5.0, (lo, hi)
    assert histogram_options(histogram_spec_for(stim, "gain"))["group"] == "temp"


def _dl_prepared():
    from chipify import data_loader as _dl
    return _dl.prepare_results(_results())


def test_summary_page_draws_the_fail_pie_beside_the_table(caplog):
    """The pie call used a keyword draw_adv_plot does not have; the TypeError
    was swallowed and the page showed an empty axes. Drawing it must also
    keep the measurement table, which shares the figure."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Wedge

    from chipify import data_loader as _dl
    from chipify import pdf_export
    from chipify.uikit.services import measurements as meas

    df = _results()
    df["gain_pass"] = [True, True, True, False]       # one gain failure
    df["pm_pass"] = [False, False, False, True]       # three pm failures
    df["tb_overall_pass"] = df["gain_pass"] & df["pm_pass"]
    df = _dl.prepare_results(df)
    stim = _stim()
    pages = []

    class _Stub:
        def savefig(self, fig, **_kw):
            texts = [t.get_text() for a in fig.axes for t in a.texts]
            wedges = [p for a in fig.axes for p in a.patches if isinstance(p, Wedge)]
            pages.append((texts, wedges))

    with caplog.at_level("WARNING", logger="chipify.pdf"):
        pdf_export._add_table(_Stub(), meas.measurement_rows(df, stim), df, stim)
    assert not caplog.records, [r.getMessage() for r in caplog.records]
    texts, wedges = pages[0]
    assert "Parameter" in texts and "gain" in texts      # the table survived
    assert len(wedges) == 2                              # one slice per failing spec
    assert "25.0%" in texts and "75.0%" in texts


# ── The PDF contains exactly the configured plots ─────────────────────────────

def _pdf_pages(stim, df, out_root):
    """Render the report's plot pages and return one summary per page."""
    import matplotlib
    matplotlib.use("Agg")
    from chipify import pdf_export

    pages = []

    class _Stub:
        def savefig(self, fig, **_kw):
            # The banner sits at the top of the page; everything below is plot.
            header = [a for a in fig.axes if a.get_position().y0 > 0.9]
            plots = [a for a in fig.axes if a not in header]
            pages.append({
                "titles": [a.get_title() for a in plots if a.get_title()],
                "artists": sum(len(a.lines) + len(a.patches) + len(a.collections)
                               + len(a.images) for a in plots),
                "labels": [t.get_text() for a in header for t in a.texts]
                          + [t.get_text() for t in fig.texts],
                "facecolors": [a.get_facecolor() for a in plots],
                "header_facecolors": [a.get_facecolor() for a in header],
            })

    pdf_export._add_report_plots(_Stub(), df, stim, stim.reports.plots, out_root)
    return pages


def test_pdf_renders_every_configured_plot_type(tmp_path):
    """Only histograms used to reach the PDF; scatter/correlation were dropped."""
    stim = _stim("""
        reports:
          pdf: true
          plots:
            - {type: histogram, param: gain}
            - {type: scatter, x: gain, y: pm}
            - {type: correlation}
            - {type: tornado, target: gain}
    """)
    df = _dl_prepared()
    pages = _pdf_pages(stim, df, tmp_path)

    assert len(pages) == 4, "one page per configured plot"
    for page in pages:
        assert page["artists"] > 0, page          # not a blank page
    joined = " ".join(t for p in pages for t in p["titles"])
    assert "Distribution of" in joined and "Shmoo" in joined
    assert "Correlation" in joined


def test_pdf_plot_pages_are_labelled_in_order(tmp_path):
    stim = _stim("""
        reports:
          pdf: true
          plots:
            - {type: histogram, param: gain}
            - {type: scatter, x: gain, y: pm}
    """)
    pages = _pdf_pages(stim, _dl_prepared(), tmp_path)
    labels = [" ".join(p["labels"]) for p in pages]
    assert "histogram_gain" in labels[0] and "(1 / 2)" in labels[0]
    assert "scatter_gain_vs_pm" in labels[1] and "(2 / 2)" in labels[1]


def test_pdf_plot_pages_use_the_white_paper_palette(tmp_path):
    """Embedded pages must not carry the dark on-screen theme onto paper."""
    stim = _stim("""
        reports:
          pdf: true
          plots: [{type: scatter, x: gain, y: pm}]
    """)
    pages = _pdf_pages(stim, _dl_prepared(), tmp_path)
    # white_background() is active while the page is written.
    for fc in pages[0]["facecolors"]:
        assert tuple(round(c, 3) for c in fc[:3]) == (1.0, 1.0, 1.0), fc
    # ...but the header banner keeps its colour, or it would vanish into
    # white-on-white like it used to.
    for fc in pages[0]["header_facecolors"]:
        assert tuple(round(c, 3) for c in fc[:3]) != (1.0, 1.0, 1.0), fc


def test_configured_plots_replace_the_automatic_sections(tmp_path):
    """"Exactly the plots the datasheet asks for" — no extra auto figures."""
    stim = _stim("""
        reports:
          pdf: true
          plots: [{type: scatter, x: gain, y: pm}]
    """)
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    pdf = next(f for f in result.files if f.suffix == ".pdf")
    from pypdf import PdfReader
    pages = PdfReader(str(pdf)).pages
    # cover + measurement table + the single configured plot.
    assert len(pages) == 3, len(pages)


def test_no_plots_configured_keeps_the_default_report(tmp_path):
    """A pdf-only run still gets the automatic histograms and correlation."""
    stim = _stim()
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path,
                                 config=rs.pdf_only_config())
    pdf = next(f for f in result.files if f.suffix == ".pdf")
    from pypdf import PdfReader
    assert len(PdfReader(str(pdf)).pages) > 3


def test_pdf_pages_are_portrait(tmp_path):
    """Plot pages must match the rest of the report, not sit sideways."""
    from chipify import pdf_export

    stim = _stim("""
        reports:
          pdf: true
          plots:
            - {type: histogram, param: gain}
            - {type: scatter, x: gain, y: pm}
            - {type: correlation}
    """)
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    pdf = next(f for f in result.files if f.suffix == ".pdf")

    from pypdf import PdfReader
    for page in PdfReader(str(pdf)).pages:
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        assert h > w, f"page is landscape: {w} x {h}"
        assert round(w / 72, 1) == round(pdf_export.A4[0], 1)


def test_plot_axes_are_fitted_into_the_page_band(tmp_path):
    """A plot must not stretch down the whole sheet on a portrait page."""
    from chipify import pdf_export

    stim = _stim("""
        reports:
          pdf: true
          plots: [{type: scatter, x: gain, y: pm}]
    """)
    pages = _pdf_pages(stim, _dl_prepared(), tmp_path)
    assert len(pages) == 1
    # Recorded by _pdf_pages below via the axes positions.
    left, bottom, width, height = pdf_export._PLOT_BAND
    assert 0.0 < height < 0.7, "band should leave margins above and below"
    assert left + width <= 0.95, "band must leave room for colorbar labels"


_THREE_PLOTS = """
    reports:
      pdf: true
      plots_per_page: 2
      plots:
        - {type: histogram, param: gain, name: gain_hist}
        - {type: scatter, x: gain, y: pm, name: gain_vs_pm}
        - {type: tornado, target: gain, name: gain_tornado}
"""


def test_two_plots_per_page_stack_on_portrait_pages(tmp_path):
    from pypdf import PdfReader

    stim = _stim(_THREE_PLOTS)
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path)
    pages = PdfReader(str(result.pdf_report)).pages
    # cover + measurement table + ceil(3 / 2) plot pages.
    assert len(pages) == 4
    for page in pages:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        assert height > width, "every page stays portrait"
    first, second = pages[2].extract_text(), pages[3].extract_text()
    assert "Plots" in first and "(1 / 2)" in first
    # Both plots really are on the page, not just their captions.
    assert "gain_hist" in first and "Distribution of" in first
    assert "gain_vs_pm" in first and "Shmoo" in first
    assert "gain_tornado" in second and "Sensitivity" in second


def test_two_up_slots_reduce_the_plot_height():
    from chipify import pdf_export

    page_band_h = pdf_export._PLOT_BAND[3] * pdf_export.A4[1]
    slot_band_h = pdf_export._SLOT_BAND[3] * pdf_export._SLOT_H
    assert slot_band_h < 0.75 * page_band_h
    # Two slots fit between the banner and the bottom margin without overlap.
    top, bottom = pdf_export._SLOT_TOPS
    assert top <= 0.935 * pdf_export.A4[1] - 0.3
    assert bottom <= top - pdf_export._SLOT_H - 0.3
    assert bottom - pdf_export._SLOT_H >= 0.3


def test_two_up_without_pypdf_falls_back_to_one_plot_per_page(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", None)   # import raises ImportError
    result = rs.generate_reports(_results(), _stim(_THREE_PLOTS), "d.yaml", tmp_path)
    monkeypatch.undo()
    from pypdf import PdfReader
    assert len(PdfReader(str(result.pdf_report)).pages) == 2 + 3


def test_page_header_banner_is_actually_drawn():
    """axis("off") also hides the patch, which left the banner invisible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chipify import pdf_export

    fig = plt.figure(figsize=pdf_export.A4, facecolor="white")
    ax = pdf_export._page_header(fig, "Banner")
    try:
        assert ax.get_frame_on()
        assert ax.axison, "the axis must stay on or the background patch is skipped"
        assert not any(s.get_visible() for s in ax.spines.values())
        assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0
    finally:
        plt.close(fig)


# ── Datasheet description ─────────────────────────────────────────────────────

def test_description_is_kept_on_one_line():
    stim = _stim("""
        description: |
          loop gain, supply rejection
          and noise
    """)
    assert stim.description == "loop gain, supply rejection and noise"
    assert _stim().description == ""


def test_description_must_be_text():
    with pytest.raises(SchemaError, match="description"):
        _stim("description: {a: 1}")


def test_description_is_printed_on_the_cover(tmp_path):
    from pypdf import PdfReader

    stim = _stim("description: operating point of the bias chain")
    result = rs.generate_reports(_results(), stim, "d.yaml", tmp_path,
                                 config=rs.pdf_only_config())
    cover = PdfReader(str(result.pdf_report)).pages[0].extract_text()
    assert "operating point of the bias chain" in cover


# ── Batch report ──────────────────────────────────────────────────────────────

def _batch_entry(tmp_path, name, *, description="", pdf=True, df=None):
    stim = _stim(f"description: {description}" if description else "")
    out = tmp_path / name
    entry = rs.BatchEntry(tmp_path / f"{name}.yaml", out,
                          df=_results() if df is None else df, stim=stim,
                          duration_s=90.0)
    if pdf:
        entry.reports = rs.generate_reports(entry.df, stim, str(entry.yaml_path),
                                            out, config=rs.pdf_only_config())
    return entry


def _outline(reader):
    return [(item.title, reader.get_destination_page_number(item))
            for item in reader.outline]


def test_batch_report_is_a_summary_then_each_datasheet_report(tmp_path):
    from pypdf import PdfReader

    from chipify import data_loader as _dl

    entries = [_batch_entry(tmp_path, "op", description="operating point"),
               _batch_entry(tmp_path, "ac", description="loop gain and noise")]
    result = rs.generate_batch_report(entries, tmp_path, title="bandgap",
                                      subtitle="IHP SG13CMOS5L")
    assert not result.warnings, result.warnings
    assert result.out_dir.parent == rs.reports_dir(tmp_path)
    assert rs.latest_reports(tmp_path) == str(result.out_dir)

    section_pages = [len(PdfReader(str(e.reports.pdf_report)).pages) for e in entries]
    reader = PdfReader(str(result.pdf_report))
    assert len(reader.pages) == 1 + sum(section_pages)
    # One bookmark per datasheet, at the page the summary says it starts on.
    assert _outline(reader) == [("Summary", 0), ("op", 1),
                                ("ac", 1 + section_pages[0])]
    # The datasheet's own report is appended unchanged.
    first_cover = reader.pages[1].extract_text()
    assert first_cover == PdfReader(str(entries[0].reports.pdf_report)).pages[0].extract_text()

    summary = reader.pages[0].extract_text()
    s = _dl.result_summary(_dl.prepare_results(_results()))
    for text in ("bandgap", "IHP SG13CMOS5L", "operating point",
                 "loop gain and noise", f"{s.yield_pct:.1f} %", "1.5 min"):
        assert text in summary, text
    assert f"op {s.total} {s.valid} {s.passed}" in summary
    assert f"ac {s.total} {s.valid} {s.passed}" in summary
    # Start pages printed in the table: op after the summary, ac after op.
    assert summary.count(f"1.5 min {2 + section_pages[0]}") == 1


def test_batch_report_renders_a_report_for_a_datasheet_without_one(tmp_path):
    """A datasheet whose reports: block asks for no PDF still gets a section,
    but nothing is added to its own output folder."""
    from pypdf import PdfReader

    entry = _batch_entry(tmp_path, "op", pdf=False)
    result = rs.generate_batch_report([entry], tmp_path, title="t")
    assert not result.warnings, result.warnings
    reader = PdfReader(str(result.pdf_report))
    assert len(reader.pages) > 1
    assert _outline(reader) == [("Summary", 0), ("op", 1)]
    assert not (tmp_path / "op").exists()


def test_failed_datasheet_and_failing_measurements_are_on_the_summary(tmp_path):
    from pypdf import PdfReader

    failing = _results()
    failing.loc[3, ["gain", "gain_pass", "tb_overall_pass"]] = [12.5, False, False]
    entries = [_batch_entry(tmp_path, "op", df=failing),
               rs.BatchEntry(tmp_path / "crashed.yaml", tmp_path / "crashed",
                             error="no data")]
    result = rs.generate_batch_report(entries, tmp_path, title="t")
    reader = PdfReader(str(result.pdf_report))
    # The crashed datasheet is listed but has no pages or bookmark.
    assert [title for title, _ in _outline(reader)] == ["Summary", "op"]
    summary = reader.pages[0].extract_text()
    assert "75.0 %" in summary
    assert "FAIL: gain" in summary
    assert "no results (no data)" in summary


def test_batch_summary_continues_on_further_pages(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pypdf import PdfReader

    from chipify import pdf_export

    one_page = tmp_path / "one.pdf"
    with PdfPages(one_page) as pdf:
        fig = plt.figure(figsize=pdf_export.A4)
        pdf.savefig(fig)
        plt.close(fig)
    sections = [pdf_export.BatchSection(f"ds{i:02d}", pdf_path=str(one_page),
                                        total=4, valid=4, passed=4,
                                        yield_pct=100.0, duration_s=60.0)
                for i in range(40)]
    path = pdf_export.generate_batch_pdf(sections, tmp_path / "b.pdf", title="t")
    reader = PdfReader(path)
    summary_pages = len(reader.pages) - len(sections)
    assert summary_pages == 2
    assert "Contents (continued)" in reader.pages[1].extract_text()
    assert [s.start_page for s in sections] == list(range(3, 43))
    assert _outline(reader)[1:] == [(f"ds{i:02d}", 2 + i) for i in range(40)]
