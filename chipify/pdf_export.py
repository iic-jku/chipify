# Copyright (c) 2026 Santiago Hofwimmer
"""
pdf_export.py – Professional IC-datasheet-style PDF report for Chipify.

Page structure (strict A4 portrait throughout):
  1. Cover          – header bar, metadata block, yield badge, stats,
                      fail-breakdown pie chart (only when yield < 100 %)
  2. Measurements   – centred, IC-datasheet-styled table, alternating rows,
                      colour-coded Cpk + Status cells
  3. Histograms     – 2-up per page, white bg, Gauss fit + spec lines +
                      per-plot annotation box (μ, σ, Cpk, Status)
  4. Correlation    – white-background Pearson heatmap

The batch report (:func:`generate_batch_pdf`) puts a summary page with one row
per datasheet in front of the datasheets' own reports.
"""

import datetime
import io
import logging
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from chipify import data_loader as _dl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats

from chipify import app_config
from chipify.uikit.services import measurements as _meas
from chipify.plot_manager import PlotManager
from chipify.plot_manager import PRINT_THEME as PlotManager_PRINT_THEME
from chipify.exporters._white_bg import white_background
from chipify.reports import histogram_options, histogram_spec_for

# ── Style constants ────────────────────────────────────────────────────────────
BLUE   = "#1a5fa8"
GREEN  = "#1e7d3a"
LGREEN = "#d6f0de"
RED    = "#b71c1c"
LRED   = "#fde8e8"
AMBER  = "#c67c00"
LAMBER = "#fff3cd"
LGRAY  = "#f0f0f0"
MGRAY  = "#c8c8c8"
DGRAY  = "#555555"

A4     = (8.27, 11.69)   # inches

log = logging.getLogger("chipify.pdf_export")
ML     = 0.07            # left/right margin in figure-fraction units
MB     = 0.05            # bottom margin

# Applied via mpl.rc_context() inside generate_pdf_report — mutating the
# global rcParams here would permanently restyle the GUI's plots after the
# first PDF export.
_PDF_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
}


# ── Low-level helpers ──────────────────────────────────────────────────────────

#: Status cell colours. ERROR is amber, deliberately not a shade of red: an
#: un-takeable measurement is a different problem from an out-of-spec one.
_STATUS_BG = {
    _meas.STATUS_PASS: LGREEN,
    _meas.STATUS_FAIL: LRED,
    _meas.STATUS_ERROR: LAMBER,
}
_STATUS_FG = {
    _meas.STATUS_PASS: GREEN,
    _meas.STATUS_FAIL: RED,
    _meas.STATUS_ERROR: AMBER,
}


def _cpk_cell_bg(cpk):
    if math.isnan(cpk):
        return LGRAY
    if cpk >= 1.33:
        return LGREEN
    if cpk >= 1.0:
        return "#fff3cd"
    return LRED


# ── Metadata helpers (for cover page) ─────────────────────────────────────────

def _swept_params(stim, df: pd.DataFrame) -> list:
    if stim is not None and hasattr(stim, "params"):
        names = []
        for pname, pvalues in stim.params.items():
            try:
                if hasattr(pvalues, "__len__") and not isinstance(pvalues, str) and len(pvalues) > 1:
                    names.append(str(pname))
            except Exception:
                continue
        return names
    fallback = []
    for c in df.columns:
        if c.endswith("_pass") or c.endswith("_overall_pass") or c in ("global_pass", "sim_error"):
            continue
        try:
            n = df[c].nunique(dropna=True)
            if 1 < n <= 64:
                fallback.append(c)
        except Exception:
            continue
    return fallback


def _sim_duration_sec(df: pd.DataFrame):
    for col in (
        "simulation_duration_s_total",
        "sim_duration_s",
        "simulation_duration_s",
        "duration_s",
        "elapsed_s",
        "sim_time_s",
    ):
        if col in df.columns:
            try:
                return float(pd.to_numeric(df[col], errors="coerce").sum())
            except Exception:
                continue
    return None


class _DummyCanvas:
    def draw(self):
        pass


# ── Page header bar (shared across all pages) ─────────────────────────────────

def _page_header(fig, label: str):
    """Blue banner across the top of a report page.

    Ticks and spines are hidden individually rather than with ``axis("off")``:
    that turns the whole axis off, and matplotlib then skips the background
    patch as well — which left every page's banner as white text on white.
    """
    ax = fig.add_axes([0, 0.935, 1, 0.065])
    ax.set_facecolor(BLUE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, label, color="white", fontsize=14, weight="bold",
            ha="center", va="center", transform=ax.transAxes)
    return ax


# ── Section 1: Cover page ─────────────────────────────────────────────────────

def _add_cover(pdf: PdfPages, df: pd.DataFrame, yaml_path, rows, stim, sim_duration_sec=None):
    fig = plt.figure(figsize=A4, facecolor="white")
    _page_header(fig, "Chipify Statistical Report")

    ax = fig.add_axes([ML, MB, 1 - 2 * ML, 0.935 - MB])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Title block ──────────────────────────────────────────────────────────
    yaml_name = Path(yaml_path).name if yaml_path else "Unknown"
    ax.text(0.5, 0.955, yaml_name,
            ha="center", va="top", fontsize=12, weight="bold", color=DGRAY)
    description = str(getattr(stim, "description", "") or "")
    generated_y = 0.925
    if description:
        if len(description) > 95:
            description = description[:92] + "…"
        ax.text(0.5, 0.932, description,
                ha="center", va="top", fontsize=9.5, color=DGRAY, style="italic")
        generated_y = 0.908
    ax.text(0.5, generated_y,
            datetime.datetime.now().strftime("Generated  %Y-%m-%d  %H:%M:%S"),
            ha="center", va="top", fontsize=9, color=MGRAY)

    # ── Metadata grid ────────────────────────────────────────────────────────
    cfg = app_config.load_config()
    core_raw = cfg.get("num_cores")
    core_txt = str(core_raw) if core_raw else "auto"

    dur = sim_duration_sec if sim_duration_sec is not None else _sim_duration_sec(df)
    dur_txt = f"{dur:.1f} s" if dur is not None else "n/a"

    swept = _swept_params(stim, df)
    swept_txt = ", ".join(swept) if swept else "–"
    if len(swept_txt) > 72:
        swept_txt = swept_txt[:69] + "…"

    meta = [
        ("CPU Cores",           core_txt),
        ("Simulation Duration", dur_txt),
        ("Swept Parameters",    swept_txt),
    ]
    lx, rx = 0.04, 0.38
    for i, (lbl, val) in enumerate(meta):
        y = 0.885 - i * 0.030
        ax.text(lx, y, lbl + ":", ha="left", va="top", fontsize=8.5, color=DGRAY)
        ax.text(rx, y, val,         ha="left", va="top", fontsize=8.5,
                color="black", weight="bold")

    ax.axhline(0.772, xmin=0.0, xmax=1.0, color=MGRAY, linewidth=0.7)

    # ── Statistics ───────────────────────────────────────────────────────────
    s = _dl.result_summary(df)
    total, bad, valid, passed = s.total, s.crashes, s.valid, s.passed
    yld    = s.yield_pct
    yld_color = GREEN if yld >= 99 else AMBER if yld >= 80 else RED

    # Yield badge (left side)
    badge = mpatches.FancyBboxPatch(
        (0.01, 0.62), 0.44, 0.135,
        boxstyle="round,pad=0.01", linewidth=1.4,
        edgecolor=yld_color, facecolor="white",
        transform=ax.transData, clip_on=False,
    )
    ax.add_patch(badge)
    ax.text(0.23, 0.710, f"{yld:.1f}%",
            ha="center", va="center", fontsize=28, weight="bold", color=yld_color)
    ax.text(0.23, 0.626, "Global Yield",
            ha="center", va="bottom", fontsize=9.5, color=DGRAY)

    # Run counts (right side of badge row)
    stats_items = [
        ("Total Iterations",  f"{total}"),
        ("Simulator Crashes", f"{bad}"),
        ("Valid Runs",        f"{valid}"),
        ("Passing Runs",      f"{passed}"),
    ]
    for i, (lbl, val) in enumerate(stats_items):
        y = 0.748 - i * 0.034
        ax.text(0.50, y, lbl + ":", ha="left", va="center", fontsize=9.5, color=DGRAY)
        ax.text(0.88, y, val,         ha="right", va="center", fontsize=9.5,
                weight="bold", color="black")

    ax.axhline(0.606, xmin=0.0, xmax=1.0, color=MGRAY, linewidth=0.7)

    # ── Measurement summary text ──────────────────────────────────────────────
    n_pass = sum(1 for r in rows if r.status == _meas.STATUS_PASS)
    n_err  = sum(1 for r in rows if r.status == _meas.STATUS_ERROR)
    n_fail = len(rows) - n_pass - n_err

    ax.text(0.0, 0.588, "Measurement Results Summary",
            ha="left", va="top", fontsize=13, weight="bold", color=BLUE)
    ax.text(0.0, 0.549, f"Measurements evaluated: {len(rows)}",
            ha="left", va="top", fontsize=9.5, color=DGRAY)
    ax.text(0.0, 0.517, f"PASS: {n_pass}",
            ha="left", va="top", fontsize=9.5, weight="bold", color=GREEN)
    ax.text(0.145, 0.517, f"FAIL: {n_fail}",
            ha="left", va="top", fontsize=9.5, weight="bold",
            color=RED if n_fail > 0 else DGRAY)
    ax.text(0.290, 0.517, f"ERROR: {n_err}",
            ha="left", va="top", fontsize=9.5, weight="bold",
            color=AMBER if n_err > 0 else DGRAY)

    y_note = 0.488
    if n_fail > 0:
        failing = [r.name for r in rows if r.status == _meas.STATUS_FAIL]
        wrapped = textwrap.fill("Failing: " + ", ".join(failing), width=78)
        ax.text(0.0, y_note, wrapped,
                ha="left", va="top", fontsize=8.5, color=RED)
        y_note -= 0.028 * (wrapped.count(chr(10)) + 1)

    # A measurement that could never be taken must not be summarised away: the
    # cover used to count it as a pass and say nothing at all about it.
    if n_err > 0:
        errored = [r.name for r in rows if r.status == _meas.STATUS_ERROR]
        wrapped = textwrap.fill("Errored: " + ", ".join(errored), width=78)
        ax.text(0.0, y_note, wrapped,
                ha="left", va="top", fontsize=8.5, color=AMBER)

    pdf.savefig(fig)
    plt.close(fig)


# ── Section 2: Measurements table ─────────────────────────────────────────────

def _add_table(pdf: PdfPages, rows: list, valid_df: pd.DataFrame, stim):
    if not rows:
        return

    HEADERS   = ["Parameter", "Sim Min", "Sim Typ", "Sim Max",
                 "Spec Min", "Spec Max", "Cpk", "Status"]
    # Fractional column widths (sum ≈ 0.84 → centred with margins)
    COL_W     = [0.20, 0.09, 0.09, 0.09, 0.09, 0.09, 0.08, 0.09]
    TABLE_W   = sum(COL_W)                       # ~0.83
    X0        = (1.0 - TABLE_W) / 2.0            # left edge of table
    ROWS_PP   = 16
    ROW_H     = 0.82 / (ROWS_PP + 1.5)
    pages     = math.ceil(len(rows) / ROWS_PP)

    for pg in range(pages):
        chunk = rows[pg * ROWS_PP: (pg + 1) * ROWS_PP]
        fig = plt.figure(figsize=A4, facecolor="white")
        _page_header(fig, f"Measurements  ({pg + 1} / {pages})")

        ax = fig.add_axes([ML, MB, 1 - 2 * ML, 0.885])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.text(0.5, 0.922, "Measurement Results Summary",
                 ha="center", va="center", fontsize=13, weight="bold", color=DGRAY)

        hdr_y = 0.94

        # Header row
        x = X0
        for txt, w in zip(HEADERS, COL_W):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, hdr_y - ROW_H * 0.5), w - 0.002, ROW_H,
                boxstyle="square,pad=0", facecolor=BLUE,
                edgecolor="white", linewidth=0.6,
                clip_on=False, transform=ax.transData,
            ))
            ax.text(x + w / 2, hdr_y, txt,
                    ha="center", va="center", fontsize=8.5,
                    color="white", weight="bold")
            x += w

        # Data rows
        for ri, row in enumerate(chunk):
            y     = hdr_y - ROW_H * (ri + 1)
            even  = ri % 2 == 0

            cells = [
                row.name,
                _meas.fmt_eng(row.sim_min), _meas.fmt_eng(row.sim_typ), _meas.fmt_eng(row.sim_max),
                _meas.fmt_eng(row.spec_min), _meas.fmt_eng(row.spec_max),
                _meas.fmt_value(row.cpk),   # dimensionless: no unit suffix
                row.status,
            ]
            row_bg = LGRAY if even else "white"

            x = X0
            for ci, (txt, w) in enumerate(zip(cells, COL_W)):
                if ci == 6:
                    fc = _cpk_cell_bg(row.cpk)
                elif ci == 7:
                    fc = _STATUS_BG.get(row.status, LRED)
                else:
                    fc = row_bg

                ax.add_patch(mpatches.FancyBboxPatch(
                    (x, y - ROW_H * 0.5), w - 0.002, ROW_H,
                    boxstyle="square,pad=0", facecolor=fc,
                    edgecolor=MGRAY, linewidth=0.4,
                    clip_on=False, transform=ax.transData,
                ))

                tc     = DGRAY
                weight = "normal"
                ha_txt = "center"
                tx     = x + w / 2

                if ci == 0:
                    ha_txt = "left"
                    tx     = x + 0.006
                    weight = "bold"
                elif ci == 7:
                    tc     = _STATUS_FG.get(row.status, RED)
                    weight = "bold"

                ax.text(tx, y, txt, ha=ha_txt, va="center",
                        fontsize=8, color=tc, weight=weight)
                x += w

        if pg == 0:
            total_valid = len(valid_df)
            passed_valid = int(valid_df["global_pass"].sum()) if total_valid else 0
            if total_valid > 0 and passed_valid < total_valid:
                # Center pie chart between table bottom and page end.
                table_bottom_ax = hdr_y - ROW_H * len(chunk) - ROW_H * 0.5
                table_bottom_fig = MB + max(0.0, table_bottom_ax) * 0.885
                area_bottom = 0.03
                area_height = max(0.12, table_bottom_fig - area_bottom)
                pie_h = min(0.24, area_height * 0.86)
                pie_y = area_bottom + (area_height - pie_h) / 2.0
                try:
                    pie_ax = fig.add_axes([0.24, pie_y, 0.52, pie_h])
                    # PDF export always renders against a white page — pin a print-friendly
                    # palette regardless of which GUI theme is active.
                    _pdf_theme = {
                        "bg": "white", "fg": "#222222",
                        "grid": "#cccccc", "spine": "#888888",
                        "legend_bg": "white", "legend_edge": "#888888",
                        "legend_text": "#222222", "accent": "#3484F0",
                    }
                    PlotManager.draw_adv_plot(
                        fig=fig,
                        ax_dummy=pie_ax,
                        canvas=_DummyCanvas(),
                        df=valid_df,
                        current_stim=stim,
                        mode="Fail Breakdown (Pie Chart)",
                        x_col="-",
                        y_col="-",
                        target="-",
                        bg_color="white",
                        theme=_pdf_theme,
                    )
                    pie_ax.set_title("")
                except Exception:  # noqa: BLE001 — the table page must still print
                    # Never silent: a wrong keyword here raised a TypeError on
                    # every report, and the bare pass left an empty axes where
                    # the pie belongs for as long as nobody looked.
                    logging.getLogger("chipify.pdf").warning(
                        "Fail-breakdown pie could not be drawn.", exc_info=True)
            else:
                fig.text(0.5, 0.12, "All runs passed specifications.",
                         ha="center", va="center", fontsize=12, color=GREEN, weight="bold")

        pdf.savefig(fig)
        plt.close(fig)


# ── Section 2b: Simulation errors ─────────────────────────────────────────────

def _add_errors(pdf: PdfPages, errors: list):
    """One page listing every distinct simulator failure.

    Without it a broken testbench left no trace in the report at all: its runs
    were filtered out before the tables were built.
    """
    if not errors:
        return

    # Each entry costs at most ~0.142 of the axis (title + count + 3 wrapped
    # message lines + conditions + spacer), so 6 is what fits under y=0.95.
    PER_PAGE = 6
    pages = math.ceil(len(errors) / PER_PAGE)
    for pg in range(pages):
        chunk = errors[pg * PER_PAGE: (pg + 1) * PER_PAGE]
        fig = plt.figure(figsize=A4, facecolor="white")
        _page_header(fig, f"Simulation Errors  ({pg + 1} / {pages})")

        ax = fig.add_axes([ML, MB, 1 - 2 * ML, 0.885])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.text(0.5, 0.922, "Measurements that could not be taken",
                 ha="center", va="center", fontsize=13, weight="bold", color=DGRAY)

        y = 0.95
        for e in chunk:
            ax.text(0.0, y, f"{e.tb_path}  [{e.kind}]",
                    ha="left", va="top", fontsize=10, weight="bold", color=AMBER)
            y -= 0.026
            ax.text(0.02, y, f"{e.run_n} of {e.total_n} run(s) affected",
                    ha="left", va="top", fontsize=8.5, color=DGRAY)
            y -= 0.022
            for line in textwrap.wrap(e.message, width=95)[:3]:
                ax.text(0.02, y, line, ha="left", va="top", fontsize=8,
                        color=DGRAY, family="monospace")
                y -= 0.020
            if e.conditions:
                conds = ", ".join(f"{k}={v}" for k, v in e.conditions.items())
                ax.text(0.02, y, "first seen at: " + textwrap.shorten(conds, 90),
                        ha="left", va="top", fontsize=8, color=DGRAY)
                y -= 0.020
            y -= 0.014

        pdf.savefig(fig)
        plt.close(fig)


# ── Section 3: Histograms (2-up) ──────────────────────────────────────────────

def _finish_pdf_hist(ax, data, cpk, passed):
    """Apply the report's page styling over a shared-renderer histogram.

    The renderer places its legend with ``loc="best"``, which lands top-right —
    where the badge goes. Pin it left, shrink the type to page scale, then draw
    the badge, so the paper layout matches what it has always been.
    """
    ax.set_xlabel(ax.get_xlabel(), fontsize=9, color=DGRAY)
    ax.set_ylabel("Density", fontsize=9, color=DGRAY)
    ax.tick_params(colors=DGRAY, labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, fontsize=7.5, framealpha=0.9, loc="upper left")

    mu = sigma = float("nan")
    if len(data):
        try:
            mu, sigma = stats.norm.fit(data)
        except Exception:  # noqa: BLE001
            mu, sigma = data.mean(), data.std()

    status_clr = GREEN if passed else RED
    info = (f"μ = {_meas.fmt_eng(mu)}\nσ = {_meas.fmt_eng(sigma)}\n"
            f"Cpk = {_meas.fmt_value(cpk)}\n{'PASS' if passed else 'FAIL'}")
    ax.text(0.975, 0.975, info,
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, family="monospace", color=status_clr, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=status_clr, linewidth=1.2, alpha=0.93),
            zorder=5)


#: Where a plot sits on its portrait page: (left, bottom, width, height) in
#: figure coordinates. Roughly 6.5 x 5.6 inches on A4 — letting a plot fill
#: the full page height would stretch it into a tall, hard-to-read strip.
#: The right margin leaves room for a colorbar's tick labels.
_PLOT_BAND = (0.11, 0.30, 0.79, 0.48)


def _fit_axes_to_band(fig, band=_PLOT_BAND) -> None:
    """Map every axes of *fig* into *band*, preserving their relative layout.

    The renderers lay themselves out for a figure of their own proportions, so
    on a portrait page their axes would span the whole sheet. Rescaling the
    union of the axes into a band keeps a readable aspect ratio, and doing it
    proportionally keeps stacked plots (the Bode magnitude/phase pair) and
    colorbars in the right places relative to each other.
    """
    axes = list(fig.axes)
    if not axes:
        return
    boxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0

    bx, by, bw, bh = band
    for ax, box in zip(axes, boxes):
        ax.set_position([
            bx + (box.x0 - x0) / span_x * bw,
            by + (box.y0 - y0) / span_y * bh,
            box.width / span_x * bw,
            box.height / span_y * bh,
        ])


#: Two plots per page (``plots_per_page: 2``). Every renderer owns its whole
#: figure (``fig.clf()``, ``tight_layout()``), so two cannot share one. Each is
#: drawn on its own A4-wide figure of _SLOT_H inches, saved as a one-page PDF,
#: and placed onto the report page afterwards (:func:`_place_slots`) — vector
#: all the way, no rasterising.
_SLOT_H = 4.8
#: Top edge of each slot, in inches from the bottom of the page. The header
#: banner ends at 10.93 in; each slot has a caption line above it.
_SLOT_TOPS = (10.58, 5.38)
#: Plot band inside a slot figure: the single-plot page's left/right margins,
#: a reduced height (about 3.7 in instead of 5.6 in), room for title and labels.
_SLOT_BAND = (0.11, 0.12, 0.79, 0.78)


def _render_slot_pdf(spec, df, stim, out_root, name) -> bytes:
    """One plot on a slot-sized figure, as the bytes of a one-page PDF."""
    from chipify import report_service

    size = (A4[0], _SLOT_H)
    try:
        fig, _ = report_service.render_spec(
            spec, df, stim, out_root=out_root,
            theme=PlotManager_PRINT_THEME, figsize=size,
        )
        _fit_axes_to_band(fig, _SLOT_BAND)
    except Exception as exc:  # noqa: BLE001 — one bad spec must not kill the report
        log.warning("Report plot %r could not be drawn into the PDF: %s", name, exc)
        fig = plt.figure(figsize=size, facecolor="white")
        fig.text(0.5, 0.5, f"{name}\ncould not be rendered:\n{exc}",
                 ha="center", va="center", color=RED, fontsize=11, wrap=True)
    buf = io.BytesIO()
    with white_background(fig):
        fig.savefig(buf, format="pdf", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _add_report_plot_pairs(pdf: PdfPages, df, stim, plots: list, out_root) -> list:
    """Pages of two stacked plots; returns where the plots go (see _place_slots).

    The pages are written with their header and a caption per slot; the plots
    themselves are placed once the PDF is closed.
    """
    from chipify import report_service

    pairs = [plots[i:i + 2] for i in range(0, len(plots), 2)]
    placements = []
    for page_no, pair in enumerate(pairs, start=1):
        fig = plt.figure(figsize=A4, facecolor="white")
        _page_header(fig, f"Plots   ({page_no} / {len(pairs)})")
        slots = []
        for top, spec in zip(_SLOT_TOPS, pair):
            name = report_service.default_plot_name(spec)
            fig.text(_SLOT_BAND[0], (top + 0.17) / A4[1], name, ha="left",
                     va="center", fontsize=11, weight="bold", color=DGRAY)
            slots.append((_render_slot_pdf(spec, df, stim, out_root, name),
                          top - _SLOT_H))
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        placements.append((pdf.get_pagecount() - 1, slots))
    return placements


def _place_slots(pdf_path, placements) -> None:
    """Overlay the two-up plots onto their pages of the finished PDF.

    *placements* is ``[(page_index, [(pdf_bytes, bottom_inches), ...]), ...]``.
    The slot PDFs are as wide as the page, so each only needs moving up.
    """
    from pypdf import PdfReader, PdfWriter, Transformation

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append(reader)
    for index, slots in placements:
        page = writer.pages[index]
        for data, bottom in slots:
            page.merge_transformed_page(PdfReader(io.BytesIO(data)).pages[0],
                                        Transformation().translate(0, bottom * 72))
    if reader.metadata:
        writer.add_metadata(reader.metadata)
    with open(pdf_path, "wb") as fh:
        writer.write(fh)


def _add_report_plots(pdf: PdfPages, df, stim, plots: list, out_root,
                      per_page: int = 1) -> list:
    """The plots declared in the datasheet's ``reports:`` block.

    Renders through ``report_service.render_spec`` — the same call that writes
    the standalone PNG/SVG — so the report contains exactly the configured
    plots and each one looks the same in both places. Previously the PDF only
    ever showed its own per-measurement histograms, so a configured scatter,
    transient or DC sweep appeared as an image but was missing from the report.

    Each renderer owns its whole figure (they all call ``fig.clf()``), so with
    *per_page* 1 every plot gets its own page. With 2, two plots are stacked
    per page (:func:`_add_report_plot_pairs`); the returned placements must
    then be applied with :func:`_place_slots` once the PDF is closed. Returns
    ``[]`` for one plot per page.
    """
    from chipify import report_service

    if per_page == 2:
        try:
            import pypdf  # noqa: F401 — places the plots onto their pages
        except ImportError:
            log.warning("plots_per_page: 2 needs pypdf (pip install pypdf); "
                        "using one plot per page.")
        else:
            return _add_report_plot_pairs(pdf, df, stim, plots, out_root)

    total = len(plots)
    for i, spec in enumerate(plots, start=1):
        name = report_service.default_plot_name(spec)
        try:
            fig, _ = report_service.render_spec(
                spec, df, stim, out_root=out_root,
                theme=PlotManager_PRINT_THEME, figsize=A4,
            )
            _fit_axes_to_band(fig)
        except Exception as exc:  # noqa: BLE001 — one bad spec must not kill the report
            log.warning("Report plot %r could not be drawn into the PDF: %s", name, exc)
            fig = plt.figure(figsize=A4, facecolor="white")
            fig.text(0.5, 0.5, f"{name}\ncould not be rendered:\n{exc}",
                     ha="center", va="center", color=RED, fontsize=11, wrap=True)

        # Snapshot before the banner is added: white_background must whiten the
        # plot but leave the header's colour alone.
        plot_axes = list(fig.axes)
        _page_header(fig, f"{name}   ({i} / {total})")
        # The same re-skin the PNG/SVG exporters apply. Without it the report
        # embedded the dark on-screen palette while the exported image of the
        # very same plot came out light.
        with white_background(fig, axes=plot_axes):
            pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    return []

def _add_histograms(pdf: PdfPages, df: pd.DataFrame, stim, rows_meta: list):
    """Distribution pages, one measurement per slot.

    Takes the full prepared frame: each measurement is scoped to the runs where
    its own testbench produced data, so one crashed testbench costs only its own
    histograms instead of blanking every page.
    """

    meta = {r.name: r for r in rows_meta}
    params = [r.name for r in rows_meta if r.name in df.columns]
    if not params:
        return

    pairs = [params[i:i + 2] for i in range(0, len(params), 2)]
    total = len(pairs)

    for pi, pair in enumerate(pairs):
        fig = plt.figure(figsize=A4, facecolor="white")
        _page_header(fig, f"Histograms  ({pi + 1} / {total})")
        fig.text(0.5, 0.925, "Measurement Distributions",
                 ha="center", va="center", fontsize=13, color=DGRAY, weight="bold")

        for si, p in enumerate(pair):
            m = meta.get(p)
            if m is None:
                continue
            data = _dl.plot_rows(df, stim, [p])[p].dropna()
            if data.empty:
                continue
            # Two balanced slots that avoid clipping and keep labels/legend readable.
            bottom = 0.56 - si * 0.43
            ax = fig.add_axes([ML + 0.015, bottom, 1 - 2 * (ML + 0.015), 0.29])
            # Same renderer, same options as the standalone figure: a report
            # page must not show a measurement differently from its PNG.
            hist = histogram_options(histogram_spec_for(stim, p))
            PlotManager.draw_histogram(
                fig, ax, fig.canvas, df, stim, p,
                hist["fit"], hist["group"], hist["bins"], hist["zoom"],
                "None", theme=PlotManager_PRINT_THEME, tight=False,
            )
            _finish_pdf_hist(ax, data, m.cpk, m.status == _meas.STATUS_PASS)

        pdf.savefig(fig)
        plt.close(fig)


# ── Section 4: Correlation heatmap ────────────────────────────────────────────

def _add_correlation(pdf: PdfPages, df: pd.DataFrame):
    import copy as _copy
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # run_id / duration are bookkeeping, not data (matches the GUI heatmap).
    non_data = {"run_id", "simulation_duration_s_total"}
    cols = [c for c in numeric
            if not c.endswith("_pass") and c != "global_pass"
            and c not in non_data
            and df[c].nunique() > 1]
    if len(cols) < 2:
        return

    corr = df[cols].corr()
    n = len(cols)

    fig = plt.figure(figsize=A4, facecolor="white")
    fig.patch.set_facecolor("white")
    # Centered plotting block on the page.
    ax = fig.add_axes([0.20, 0.24, 0.60, 0.60])
    ax.set_facecolor("white")

    # Mask the self-correlation diagonal (always 1.0 → deep red); grey reads
    # as "not informative" — same treatment as the GUI heatmap.
    masked = np.ma.masked_where(np.eye(n, dtype=bool), corr.values)
    cmap = _copy.copy(plt.cm.coolwarm)
    cmap.set_bad("#bfbfbf")
    cax  = ax.matshow(masked, cmap=cmap, vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Pearson r", fontsize=9)

    fs = max(6, 10 - n // 3)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # masked grey diagonal carries no value
            v  = corr.iloc[i, j]
            fc = "black" if abs(v) < 0.6 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs, color=fc)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(cols, rotation=45, ha="right", rotation_mode="anchor", fontsize=9, color=DGRAY)
    ax.set_yticklabels(cols, fontsize=9, color=DGRAY)
    ax.tick_params(axis="x", labelbottom=True, bottom=True, labeltop=False, top=False, pad=4)
    ax.tick_params(axis="y", labelleft=True, left=True, labelright=False, right=False, pad=4)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_title("Parameter Correlation Matrix", fontsize=13, pad=14, weight="bold")
    fig.text(0.5, 0.925, "Correlation Matrix",
             ha="center", va="center", fontsize=13, color=DGRAY, weight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(MGRAY)

    pdf.savefig(fig)
    plt.close(fig)


# ── Section 5: ReportPlugin pages ─────────────────────────────────────────────

def _add_plugin_pages(pdf: PdfPages, valid_df: pd.DataFrame, stim):
    """Invoke ``render_pdf`` on every installed ReportPlugin (PLUGINS.md)."""
    try:
        from chipify.plugin_loader import get_report_plugins
        plugins = get_report_plugins()
    except Exception:
        return
    for cls in plugins:
        try:
            cls().render_pdf(pdf, valid_df, stim)
        except Exception:
            import logging
            logging.getLogger("chipify.pdf").warning(
                "ReportPlugin %r render_pdf failed.",
                getattr(cls, "name", cls.__name__), exc_info=True,
            )


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf_report(df, stim, yaml_path, out_dir, sim_duration_sec=None,
                        reports_config=None, out_root=None):
    """Build the PDF report.

    When the datasheet (or *reports_config*) declares ``plots:``, the report's
    figure pages are exactly those plots. With nothing declared it falls back
    to the automatic per-measurement histograms and correlation page.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = reports_config if reports_config is not None else getattr(stim, "reports", None)
    cfg_plots = list(getattr(cfg, "plots", None) or [])
    # Waveform plots resolve their CSVs under the project output directory;
    # out_dir is the run's report folder inside it.
    out_root = Path(out_root) if out_root else out_dir.parent.parent
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = out_dir / f"report_{ts}.pdf"

    prepared = _dl.prepare_results(df)
    valid_df = prepared[prepared["sim_error"] == "None"]
    # The service scopes errors per testbench, so it takes the full frame: a
    # row-filtered one drops every run in which any testbench crashed, which is
    # what made a wholly failed sweep render as a clean PASS.
    rows     = _meas.measurement_rows(prepared, stim)
    errors   = _meas.error_rows(prepared, stim)
    placements = []

    with mpl.rc_context(_PDF_RCPARAMS), PdfPages(pdf_path) as pdf:
        _add_cover(pdf, prepared, yaml_path, rows, stim, sim_duration_sec=sim_duration_sec)
        _add_table(pdf, rows, valid_df, stim)
        _add_errors(pdf, errors)
        if cfg_plots:
            # "Exactly the plots the datasheet asks for": the automatic
            # per-measurement histograms and the correlation page would be
            # extra figures nobody requested.
            placements = _add_report_plots(
                pdf, prepared, stim, cfg_plots, out_root,
                per_page=getattr(cfg, "plots_per_page", 1))
        else:
            _add_histograms(pdf, prepared, stim, rows)
            # corr() is pairwise and the nunique() > 1 filter already drops
            # all-NaN columns, so the full frame is safe.
            _add_correlation(pdf, prepared)
        _add_plugin_pages(pdf, valid_df, stim)

    if placements:
        _place_slots(pdf_path, placements)
    return str(pdf_path)


# ── Batch report: summary page + the datasheets' own reports ──────────────────

NAVY = "#1f4e89"

#: Summary-page geometry, in inches from the top-left corner of the sheet.
_SUM_LEFT = 0.8
_SUM_RIGHT = A4[0] - 0.8
_SUM_TOP = 0.9
_SUM_BOTTOM = A4[1] - 0.8
_SUM_HEADER_H = 0.32
_SUM_ROW_H = 0.30
_SUM_LINE_H = 0.22
#: (heading, width in inches); the widths add up to the text width.
_SUM_COLUMNS = (("datasheet", 2.17), ("runs", 0.7), ("valid", 0.7),
                ("passed", 0.7), ("yield", 0.85), ("time", 0.85), ("page", 0.7))


@dataclass
class BatchSection:
    """One datasheet of a batch report: its summary row and its own report.

    ``pdf_path`` is the datasheet's finished report, appended unchanged after
    the summary. ``None`` lists the datasheet without pages of its own (a run
    that crashed). ``error`` marks a datasheet that produced no results.
    """

    label: str
    pdf_path: str | None = None
    description: str = ""
    total: int | None = None
    valid: int | None = None
    passed: int | None = None
    yield_pct: float | None = None
    duration_s: float | None = None
    failing: list[str] = field(default_factory=list)
    erroring: list[str] = field(default_factory=list)
    error: str = ""
    #: First page of the section in the combined PDF; set by generate_batch_pdf.
    start_page: int | None = None


class _SummaryLayout:
    """Flow layout for the summary page(s): planned first, drawn second.

    Planning fixes the page count before anything is drawn. The table prints
    each section's start page, and those depend on how many summary pages
    come first.
    """

    def __init__(self):
        self.pages = []
        self.y = _SUM_TOP
        self.new_page()

    def new_page(self):
        self.pages.append([])
        self.y = _SUM_TOP

    def fits(self, height):
        return self.y + height <= _SUM_BOTTOM

    def add(self, height, draw):
        """Reserve *height* inches and queue ``draw(ax, y)`` at the current y."""
        y = self.y
        self.pages[-1].append(lambda ax: draw(ax, y))
        self.y += height

    def space(self, height):
        """Vertical gap; dropped at a page break, where it is not needed."""
        if self.fits(height):
            self.y += height


def _sum_cell(ax, x, y, w, h, text, *, bg="white", fg="black", bold=False):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=bg,
                                    edgecolor="black", linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9.5, color=fg, weight="bold" if bold else "normal")


def _sum_row(ax, y, values, *, yield_bg="white", header=False, bold=False):
    x = (A4[0] - sum(w for _, w in _SUM_COLUMNS)) / 2
    height = _SUM_HEADER_H if header else _SUM_ROW_H
    for (name, w), text in zip(_SUM_COLUMNS, values):
        if header:
            _sum_cell(ax, x, y, w, height, text, bg=NAVY, fg="white", bold=True)
        else:
            _sum_cell(ax, x, y, w, height, text,
                      bg=yield_bg if name == "yield" else "white", bold=bold)
        x += w


def _yield_bg(yield_pct):
    """Yield cell colour, on the same thresholds as the cover's yield badge."""
    if yield_pct is None:
        return LRED
    return LGREEN if yield_pct >= 99 else LAMBER if yield_pct >= 80 else LRED


def _fmt_minutes(seconds):
    return "—" if seconds is None else f"{seconds / 60:.1f} min"


def _section_values(s):
    label = s.label if len(s.label) <= 30 else s.label[:29] + "…"
    page = str(s.start_page) if s.start_page else "—"
    if s.error or s.total is None:
        return [label, "—", "—", "—", "no data", _fmt_minutes(s.duration_s), page]
    return [label, str(s.total), str(s.valid), str(s.passed),
            f"{s.yield_pct:.1f} %", _fmt_minutes(s.duration_s), page]


def _plan_batch_summary(sections, title, subtitle, generated):
    lay = _SummaryLayout()
    left, mid = _SUM_LEFT, A4[0] / 2
    title_size = 24 if len(title) <= 34 else max(14, 24 * 34 / len(title))

    def heading(text, size=16):
        def draw(ax, y):
            ax.text(left, y + 0.22, text, ha="left", va="center",
                    fontsize=size, weight="bold", color=NAVY)
        return draw

    def title_block(ax, y):
        ax.text(mid, y + 0.35, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=NAVY)
        if subtitle:
            ax.text(mid, y + 0.85, subtitle, ha="center", va="center", fontsize=12)
        ax.text(mid, y + 1.15, generated, ha="center", va="center",
                fontsize=9.5, color="grey")

    def note(ax, y):
        ax.text(left, y + 0.12, "Each datasheet below follows as its own Chipify "
                "report (cover, measurement table, plots).",
                ha="left", va="center", fontsize=9.5, color=DGRAY)

    def header(ax, y):
        _sum_row(ax, y, [name for name, _ in _SUM_COLUMNS], header=True)

    def continue_table():
        lay.new_page()
        lay.add(0.45, heading("Contents (continued)"))
        lay.add(_SUM_HEADER_H, header)

    lay.add(1.6, title_block)
    lay.add(0.45, heading("Contents"))
    lay.add(0.4, note)
    lay.add(_SUM_HEADER_H, header)
    for s in sections:
        if not lay.fits(_SUM_ROW_H):
            continue_table()
        lay.add(_SUM_ROW_H, lambda ax, y, s=s: _sum_row(
            ax, y, _section_values(s),
            yield_bg=_yield_bg(None if s.error else s.yield_pct)))

    measured = [s for s in sections if not s.error and s.total is not None]
    if len(measured) > 1:
        total = sum(s.total for s in measured)
        passed = sum(s.passed for s in measured)
        yield_pct = passed / total * 100.0 if total else 0.0
        durations = [s.duration_s for s in measured if s.duration_s is not None]
        values = ["total", str(total), str(sum(s.valid for s in measured)),
                  str(passed), f"{yield_pct:.1f} %",
                  _fmt_minutes(sum(durations) if durations else None), ""]
        if not lay.fits(_SUM_ROW_H):
            continue_table()
        lay.add(_SUM_ROW_H, lambda ax, y: _sum_row(
            ax, y, values, yield_bg=_yield_bg(yield_pct), bold=True))

    def text_section(name, entries):
        """Bold label column + wrapped text; *entries* = (label, [(text, colour)])."""
        if not entries:
            return
        label_w = max(len(label) + 1 for label, _ in entries) * 9 * 0.62 / 72 + 0.3
        x_text = left + min(max(label_w, 1.5), 3.5)
        chars = max(20, int((_SUM_RIGHT - x_text) * 72 / (9 * 0.52)))
        lay.space(0.45)
        if not lay.fits(0.45 + _SUM_LINE_H):
            lay.new_page()
        lay.add(0.45, heading(name, size=13))
        for label, parts in entries:
            lines = [(line, colour) for text, colour in parts
                     for line in (textwrap.wrap(text, chars) or [""])]
            height = len(lines) * _SUM_LINE_H
            if not lay.fits(height):
                lay.new_page()
                lay.add(0.45, heading(f"{name} (continued)", size=13))

            def draw(ax, y, label=label, lines=lines):
                ax.text(left, y + _SUM_LINE_H / 2, f"{label}:", ha="left",
                        va="center", fontsize=9, weight="bold")
                for i, (line, colour) in enumerate(lines):
                    ax.text(x_text, y + (i + 0.5) * _SUM_LINE_H, line,
                            ha="left", va="center", fontsize=9, color=colour)
            lay.add(height, draw)

    text_section("What each datasheet covers",
                 [(s.label, [(s.description, "black")])
                  for s in sections if s.description])

    problems = []
    for s in sections:
        parts = []
        if s.error:
            parts.append((f"no results ({s.error})", RED))
        if s.failing:
            parts.append(("FAIL: " + ", ".join(s.failing), RED))
        if s.erroring:
            parts.append(("ERROR: " + ", ".join(s.erroring), AMBER))
        if parts:
            problems.append((s.label, parts))
    text_section("Failing measurements", problems)
    return lay


def generate_batch_pdf(sections, out_path, *, title, subtitle="", generated=None):
    """Write the combined batch report and return its path.

    Summary page(s) first, then every section's own report PDF, appended as it
    is: nothing is rendered twice. Each section's page count is read up front
    so the summary can print its start page, and each gets a bookmark. Sets
    ``start_page`` on every section that has pages.
    """
    import chipify
    from pypdf import PdfReader, PdfWriter

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = [len(PdfReader(s.pdf_path).pages) if s.pdf_path else 0
              for s in sections]
    generated = generated or datetime.datetime.now()
    stamp = f"generated {generated:%Y-%m-%d %H:%M}  ·  chipify {chipify.__version__}"

    lay = _plan_batch_summary(sections, title, subtitle, stamp)
    page = len(lay.pages) + 1
    for s, n in zip(sections, counts):
        s.start_page = page if n else None
        page += n

    buf = io.BytesIO()
    with mpl.rc_context(_PDF_RCPARAMS), PdfPages(buf) as pdf:
        for ops in lay.pages:
            fig = plt.figure(figsize=A4, facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, A4[0])
            ax.set_ylim(A4[1], 0)  # y grows downwards, in inches
            ax.axis("off")
            for op in ops:
                op(ax)
            pdf.savefig(fig)
            plt.close(fig)
    buf.seek(0)

    writer = PdfWriter()
    writer.append(buf, import_outline=False)
    writer.add_outline_item("Summary", 0)
    for s, n in zip(sections, counts):
        if n:
            writer.append(s.pdf_path, import_outline=False)
            writer.add_outline_item(s.label, s.start_page - 1)
    writer.add_metadata({"/Title": title})
    writer.page_mode = "/UseOutlines"
    with open(out_path, "wb") as fh:
        writer.write(fh)
    return str(out_path)
