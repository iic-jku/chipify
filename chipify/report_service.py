# Copyright (c) 2026 Santiago Hofwimmer
"""
report_service.py – Render the datasheet's ``reports:`` block to files.

Headless and toolkit-agnostic: the CLI (``--reports``) and the GUI's
*Generate Reports* button share this one implementation, so the two can never
produce different output for the same datasheet.

No new plotting code lives here. Figures come from the same
:class:`~chipify.plot_manager.PlotManager` entry points the GUI tabs use, drawn
onto a plain ``FigureCanvasAgg``; image files are written by the
``ExporterPlugin`` registry (so a user's own exporter is a usable format for
free), and the PDF / Markdown / LaTeX writers are the existing generators.

Every plot is rendered independently: a spec naming a measurement this run does
not have, or a waveform directory that was never produced, records a warning
and the remaining plots are still written. A report run must never fail as a
unit — the same reasoning that scoped simulation errors per testbench.

A batch run gets one more document, :func:`generate_batch_report`: a summary
page with one row per datasheet, followed by each datasheet's own PDF report.
"""
from __future__ import annotations

import datetime
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from chipify import data_loader as _dl
from chipify.reports import (LATEX_FORMAT, PLOT_TYPES, PlotSpec,
                             ReportsConfig, histogram_options,
                             histogram_spec_for)  # noqa: F401
from chipify.uikit.services import transient_loader as _tl

log = logging.getLogger("chipify.report_service")


@dataclass
class ReportResult:
    """What a report run produced, and what it could not."""

    out_dir: Path
    files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The PDF report among ``files`` (None when none was written); the batch
    #: report appends it rather than telling it apart from an exported plot.
    pdf_report: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.files) and not self.warnings


# ── Naming ────────────────────────────────────────────────────────────────────

def _slug(text: Any) -> str:
    """Filesystem-safe fragment of a signal or measurement name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "plot"


def default_plot_name(spec: PlotSpec) -> str:
    """Filename stem for a spec that did not name itself.

    Built from the options rather than the index, so a plot keeps its filename
    when the list is reordered and two runs stay diffable.
    """
    if spec.name:
        return _slug(spec.name)
    opts = spec.options
    parts: list[str] = [spec.type]
    if spec.type in ("scatter", "corner_yield"):
        parts += [_slug(opts.get("x")), "vs", _slug(opts.get("y"))]
    elif spec.type == "histogram":
        parts.append(_slug(opts.get("param")))
    elif spec.type == "tornado":
        parts.append(_slug(opts.get("target")))
    else:
        signals = opts.get("signals") or []
        parts += [_slug(s) for s in signals[:3]]
    if opts.get("group"):
        parts += ["by", _slug(opts["group"])]
    return "_".join(p for p in parts if p)


# ── Rendering ─────────────────────────────────────────────────────────────────

#: Default size of a standalone report figure, in inches.
FIGSIZE = (8.0, 5.0)


def _new_figure(figsize=None):
    """A figure plus an Agg canvas satisfying the PlotManager draw contract."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize or FIGSIZE, dpi=150)
    canvas = FigureCanvasAgg(fig)
    return fig, canvas


def _render_waveform(spec: PlotSpec, df, stim, out_root, theme, figsize=None):
    """Draw one transient / dc / bode / spectrum overlay.

    Returns ``(fig, latex_callable)``, the callable ``None`` for a type without
    a pgfplots generator.
    """
    from chipify.plot_manager import WAVEFORM_DRAWERS

    ptype = spec.plot_type
    opts = spec.options
    kind = ptype.kind
    adir = _tl.resolve_analysis_dir(df, str(out_root), kind)
    signals = list(opts.get("signals") or _tl.list_kind_signals(stim, kind))
    if not signals:
        raise ValueError(f"no {kind} signals declared in the datasheet")
    if not adir:
        raise ValueError(f"no {kind} waveform directory for this run")

    run_ids = _tl.select_run_ids(df, opts.get("runs", _tl.RUN_MODE_ALL))
    if not run_ids:
        raise ValueError("no runs matched the requested selection")

    group_col = str(opts.get("group") or "")
    equations = (_eq_service().transient_equations(stim)
                 if kind == "transient" else [])
    view = ptype.waveform_view
    draw_fn = WAVEFORM_DRAWERS[view]
    extra = {"window": str(opts.get("window") or "")} if view == "spectrum" else {}

    fig, canvas = _new_figure(figsize)
    draw_fn(fig, canvas, adir, run_ids, signals,
            pass_map=_tl.run_pass_map(df), equations=equations, theme=theme,
            group_map=_tl.run_group_map(df, group_col), group_label=group_col,
            **extra)

    if not ptype.supports_latex:
        return fig, None

    def _latex(dest: Path, stem: str):
        from chipify import export_latex
        gen = {
            "transient": export_latex.generate_transient_latex_export,
            "dc": export_latex.generate_dc_sweep_latex_export,
            "ac": export_latex.generate_bode_latex_export,
        }[kind]
        return gen(str(dest), stem, adir, run_ids, signals, equations)

    return fig, _latex


def _render_histogram(spec: PlotSpec, df, stim, theme, figsize=None):
    from chipify.plot_manager import PlotManager

    opts = spec.options
    param = str(opts["param"])
    if param not in df.columns:
        raise ValueError(f"measurement {param!r} is not in this run's results")

    hist = histogram_options(spec)
    fig, canvas = _new_figure(figsize)
    ax = fig.add_subplot(111)
    PlotManager.draw_histogram(
        fig, ax, canvas, df, stim, param,
        hist["fit"], hist["group"], hist["bins"], hist["zoom"],
        "None", theme=theme,
    )

    def _latex(dest: Path, stem: str):
        from chipify import export_latex
        data = _dl.plot_rows(df, stim, [param])[param]
        bins_text = str(hist["bins"])
        bins = "auto" if bins_text == "Auto" else int(bins_text)
        return export_latex.generate_latex_export(
            stem, data, str(hist["fit"]), bins, str(dest))

    return fig, _latex


def _render_advanced(spec: PlotSpec, df, stim, theme, figsize=None):
    from chipify.plot_manager import PlotManager

    opts = spec.options
    x_col, y_col = str(opts.get("x", "")), str(opts.get("y", ""))
    group_col = str(opts.get("group") or "")
    # A column the run lacks draws nothing (scatter, yield matrix) or a
    # "Select a target measurement first." placeholder (tornado), and the
    # report wrote that as a finished image without a warning.
    for key in ("x", "y", "target"):
        if key in spec.plot_type.required:
            col = str(opts.get(key, ""))
            if col not in df.columns:
                raise ValueError(f"{key}: column {col!r} is not in this run's results")
    fig, canvas = _new_figure(figsize)
    _artist, rows = PlotManager.draw_adv_plot(
        fig, None, canvas, df, stim, spec.plot_type.adv_mode,
        x_col, y_col, str(opts.get("target", "")), theme=theme,
        group_col=group_col,
    )
    if spec.type != "scatter":    # the one advanced type with a pgfplots generator
        return fig, None

    def _latex(dest: Path, stem: str):
        from chipify import export_latex
        # The rows the figure drew, so the file holds exactly its points.
        return export_latex.generate_scatter_latex_export(
            str(dest), stem, rows, x_col, y_col, group_col)

    return fig, _latex


def _eq_service():
    from chipify.uikit.services import equation_service
    return equation_service


def render_spec(spec: PlotSpec, df, stim, *, out_root, theme=None, figsize=None):
    """Render one plot spec. Returns ``(figure, latex_callable_or_None)``.

    Public because the PDF report renders the datasheet's configured plots
    through exactly this function — a plot must not look different depending on
    whether it was exported as an image or embedded in the report.

    *out_root* is the project's output directory (the one holding
    ``analysis_data/``), needed to locate waveform CSVs.
    """
    if spec.plot_type.is_waveform:
        return _render_waveform(spec, df, stim, out_root, theme, figsize)
    if spec.type == "histogram":
        return _render_histogram(spec, df, stim, theme, figsize)
    return _render_advanced(spec, df, stim, theme, figsize)


# ── Writing ───────────────────────────────────────────────────────────────────

def _exporters() -> dict[str, Any]:
    from chipify.plugin_loader import get_exporter_plugins
    return {e.extension.lstrip(".").lower(): e for e in get_exporter_plugins()}


def _unique_stems(specs: list[PlotSpec], warnings: list[str]) -> list[str]:
    """One output file stem per plot spec, never shared by two specs.

    Default stems leave out options such as bins, fit or the run selection,
    and explicit names can repeat or slug to the same text: two histograms of
    one measurement were reported as two files while the second overwrote the
    first. Compared case-insensitively, as on Windows and macOS file systems.
    """
    taken: set[str] = set()
    stems: list[str] = []
    for spec in specs:
        stem = default_plot_name(spec)
        if stem.casefold() in taken:
            n = 2
            while f"{stem}_{n}".casefold() in taken:
                n += 1
            warnings.append(
                f"{stem}: another plot is already saved under this name, so "
                f"this one is saved as {stem}_{n}. Give the plots distinct "
                f"'name:' values."
            )
            stem = f"{stem}_{n}"
        taken.add(stem.casefold())
        stems.append(stem)
    return stems


def _write_plot(spec: PlotSpec, cfg: ReportsConfig, df, stim,
                out_dir: Path, theme, result: ReportResult, stem: str) -> None:
    formats = spec.resolved_formats(cfg.formats)
    if not formats:
        result.warnings.append(f"{stem}: no output formats requested; skipped.")
        return

    try:
        fig, latex_fn = render_spec(spec, df, stim,
                                    out_root=out_dir.parent.parent,
                                    theme=theme)
    except Exception as exc:  # noqa: BLE001 — one bad spec must not stop the rest
        result.warnings.append(f"{stem}: could not render ({exc})")
        log.warning("Report plot %r failed to render: %s", stem, exc)
        return

    try:
        exporters = _exporters()
        for fmt in formats:
            if fmt == LATEX_FORMAT:
                if latex_fn is None:
                    result.warnings.append(
                        f"{stem}: LaTeX export is not available for plot type "
                        f"{spec.type!r}; other formats were still written."
                    )
                    continue
                try:
                    written = latex_fn(out_dir, stem)
                except Exception as exc:  # noqa: BLE001
                    result.warnings.append(f"{stem}: LaTeX export failed ({exc})")
                    continue
                for path in (written or ()):
                    if path:
                        result.files.append(Path(path))
                continue

            exporter = exporters.get(fmt)
            if exporter is None:
                result.warnings.append(f"{stem}: no exporter for format {fmt!r}.")
                continue
            dest = out_dir / f"{stem}.{fmt}"
            # Exporters can be user plugins: one that raises costs only its own
            # file. It used to escape generate_reports and take every later
            # plot, the PDF and the Markdown report with it.
            try:
                try:
                    exporter().export(fig, str(dest), theme=theme)
                except TypeError:      # exporter predating the theme kwarg
                    exporter().export(fig, str(dest))
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"{stem}: {fmt} export failed ({exc})")
                log.warning("Report plot %r: %s export failed: %s", stem, fmt, exc)
                continue
            result.files.append(dest)
    finally:
        # Figures are created outside pyplot, so nothing closes them for us.
        fig.clf()


def _write_documents(cfg: ReportsConfig, df, stim, yaml_path: str,
                     out_dir: Path, duration_s, result: ReportResult) -> None:
    if cfg.pdf:
        try:
            from chipify import pdf_export
            # Pass the effective config, not stim.reports: a PDF-only run
            # overrides it, and the report must reflect what was asked for.
            result.pdf_report = Path(pdf_export.generate_pdf_report(
                df, stim, yaml_path, str(out_dir), sim_duration_sec=duration_s,
                reports_config=cfg, out_root=out_dir.parent.parent))
            result.files.append(result.pdf_report)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"PDF report failed ({exc})")
            log.warning("PDF report failed: %s", exc)
    if cfg.markdown:
        try:
            from chipify import md_export
            result.files.append(Path(md_export.generate_md_report(
                df, stim, yaml_path, str(out_dir / "report.md"),
                sim_duration_sec=duration_s)))
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Markdown report failed ({exc})")
            log.warning("Markdown report failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def reports_dir(out_dir: str | Path) -> Path:
    """Root directory holding every report run."""
    return Path(out_dir) / "reports"


def latest_reports(out_dir: str | Path) -> str:
    """Path recorded in ``reports/.latest`` ("" when there is none)."""
    pointer = reports_dir(out_dir) / ".latest"
    try:
        return pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def pdf_only_config() -> ReportsConfig:
    """Config for "just give me the PDF", independent of the datasheet.

    The one-click PDF is this plus :func:`generate_reports`, rather than a
    second code path with its own output location and error handling.
    """
    return ReportsConfig(pdf=True)


def generate_reports(df: pd.DataFrame, stim: Any, yaml_path: str,
                     out_dir: str | Path, *, duration_s: float | None = None,
                     theme: dict | None = None,
                     config: ReportsConfig | None = None) -> ReportResult:
    """Render everything the datasheet's ``reports:`` block asks for.

    Writes into ``<out_dir>/reports/<timestamp>/`` and points
    ``<out_dir>/reports/.latest`` at it — the pointer convention
    ``simulator.write_analysis_pointers`` already uses for analysis data.

    *config* overrides the datasheet's block, which is how the PDF-only action
    reuses this single path (see :func:`pdf_only_config`).
    """
    cfg: ReportsConfig = config or getattr(stim, "reports", None) or ReportsConfig()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = reports_dir(out_dir) / stamp
    result = ReportResult(out_dir=dest)

    if not cfg:
        result.warnings.append(
            "The datasheet declares no 'reports:' block, so there is nothing "
            "to generate."
        )
        return result

    dest.mkdir(parents=True, exist_ok=True)
    prepared = _dl.prepare_results(df)

    stems = _unique_stems(cfg.plots, result.warnings)
    for spec, stem in zip(cfg.plots, stems):
        _write_plot(spec, cfg, prepared, stim, dest, theme, result, stem)
    _write_documents(cfg, prepared, stim, yaml_path, dest, duration_s, result)

    try:
        (reports_dir(out_dir) / ".latest").write_text(str(dest), encoding="utf-8")
    except OSError as exc:
        result.warnings.append(f"Could not write the .latest pointer ({exc})")

    log.info("Reports written to %s (%d file(s), %d warning(s))",
             dest, len(result.files), len(result.warnings))
    return result


# ── Batch report ──────────────────────────────────────────────────────────────

@dataclass
class BatchEntry:
    """One datasheet of a batch run, as the combined report needs it.

    *out_dir* is the datasheet's own output root (``history/``,
    ``analysis_data/``, ``reports/``). *df* is ``None`` when the datasheet
    produced no results; *error* then says why. *reports* is what
    :func:`generate_reports` wrote for it, if anything.
    """

    yaml_path: Path
    out_dir: Path
    df: pd.DataFrame | None = None
    stim: Any = None
    duration_s: float | None = None
    summary: dict = field(default_factory=dict)
    reports: ReportResult | None = None
    error: str = ""

    @property
    def label(self) -> str:
        return Path(self.yaml_path).stem


def _entry_description(entry: BatchEntry) -> str:
    if entry.stim is not None:
        return str(getattr(entry.stim, "description", "") or "")
    try:  # a run that failed still has a datasheet worth describing
        from chipify.util import Stimuli
        return Stimuli(entry.yaml_path).description
    except Exception:  # noqa: BLE001 — the summary row must not fail on it
        return ""


def _entry_section_pdf(entry: BatchEntry, prepared: pd.DataFrame,
                       scratch: Path, result: ReportResult) -> str | None:
    """The datasheet's report PDF, rendering one only if it has none yet."""
    if entry.reports is not None and entry.reports.pdf_report is not None \
            and entry.reports.pdf_report.exists():
        return str(entry.reports.pdf_report)
    # The datasheet's own reports did not include a PDF. Render one for the
    # combined report only, so its configured outputs stay exactly as declared.
    try:
        from chipify import pdf_export
        return pdf_export.generate_pdf_report(
            prepared, entry.stim, str(entry.yaml_path), str(scratch / entry.label),
            sim_duration_sec=entry.duration_s,
            reports_config=getattr(entry.stim, "reports", None),
            out_root=entry.out_dir)
    except Exception as exc:  # noqa: BLE001 — one datasheet must not sink the rest
        result.warnings.append(f"{entry.label}: report section failed ({exc})")
        log.warning("Batch report section %r failed: %s", entry.label, exc)
        return None


def _batch_section(entry: BatchEntry, scratch: Path, result: ReportResult) -> Any:
    from chipify import pdf_export
    from chipify.uikit.services import measurements as _meas

    section = pdf_export.BatchSection(label=entry.label,
                                      description=_entry_description(entry),
                                      duration_s=entry.duration_s)
    if entry.df is None:
        section.error = entry.error or "no data"
        return section
    prepared = _dl.prepare_results(entry.df)
    # The same numbers as the datasheet's own cover page.
    summary = _dl.result_summary(prepared)
    section.total, section.valid = summary.total, summary.valid
    section.passed, section.yield_pct = summary.passed, summary.yield_pct
    rows = _meas.measurement_rows(prepared, entry.stim)
    section.failing = [r.name for r in rows if r.status == _meas.STATUS_FAIL]
    section.erroring = [r.name for r in rows if r.status == _meas.STATUS_ERROR]
    section.pdf_path = _entry_section_pdf(entry, prepared, scratch, result)
    return section


def generate_batch_report(entries: list[BatchEntry], out_dir: str | Path, *,
                          title: str | None = None,
                          subtitle: str | None = None) -> ReportResult:
    """One PDF for a whole batch: a summary page, then every datasheet's report.

    Writes ``<out_dir>/reports/<timestamp>/batch_report_<timestamp>.pdf`` and
    points ``<out_dir>/reports/.latest`` at that folder, the convention
    :func:`generate_reports` uses for a single datasheet. Each datasheet's own
    PDF (``entry.reports``) is appended unchanged; one that has none is
    rendered for the combined report only.

    *title* defaults to ``name`` in ``project.yaml``, else the project folder's
    name; *subtitle* to ``description`` in ``project.yaml``, else
    "Chipify characterization report".
    """
    from chipify import pdf_export, project_config, settings

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = reports_dir(out_dir) / stamp
    result = ReportResult(out_dir=dest)
    if not entries:
        result.warnings.append("No datasheets, so there is no batch report.")
        return result

    project = project_config.load()
    if title is None:
        title = str(project.get("name") or Path(settings.PROJECT_ROOT).name)
    if subtitle is None:
        subtitle = str(project.get("description")
                       or "Chipify characterization report")

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chipify_batch_") as scratch:
        sections = [_batch_section(e, Path(scratch), result) for e in entries]
        try:
            path = pdf_export.generate_batch_pdf(
                sections, dest / f"batch_report_{stamp}.pdf",
                title=title, subtitle=subtitle)
            result.pdf_report = Path(path)
            result.files.append(result.pdf_report)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Batch PDF report failed ({exc})")
            log.warning("Batch PDF report failed: %s", exc)

    try:
        (reports_dir(out_dir) / ".latest").write_text(str(dest), encoding="utf-8")
    except OSError as exc:
        result.warnings.append(f"Could not write the .latest pointer ({exc})")
    return result
