# Copyright (c) 2026 Santiago Hofwimmer
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chipify.bode import continuous_phase
from chipify.overlay_style import OverlayStyle
from chipify.palette import category_color
from chipify.palette import viridis as _viridis


_RE_NON_ID = re.compile(r"[^A-Za-z0-9]+")


def _safe_col(text: str) -> str:
    """Sanitise a string for use as a CSV column name / LaTeX \\foreach token."""
    cleaned = _RE_NON_ID.sub("_", str(text)).strip("_")
    return cleaned or "x"


def _collect_run_files(adir: str, run_ids: list[str]) -> list[tuple[str, str]]:
    """Return ``[(run_id, csv_path), …]`` for the given run IDs."""
    if not adir or not Path(adir).is_dir() or not run_ids:
        return []
    run_id_set = set(run_ids)
    matched: list[tuple[str, str]] = []
    for fname in Path(adir).glob("run_*.csv"):
        rid = fname.name[4:].split("__", 1)[0]
        if rid in run_id_set:
            matched.append((rid, str(fname)))
    matched.sort(key=lambda rf: rf[0])
    return matched


def _apply_equations(df: pd.DataFrame, equations: list | None) -> pd.DataFrame:
    if not equations:
        return df
    from chipify.expression import default_evaluator
    for eq in equations:
        name = (eq.get("name") or "").strip()
        expr = (eq.get("expr") or "").strip()
        if name and expr:
            try:
                df = default_evaluator.evaluate_dataframe_column(df, name, expr)
            except Exception:
                pass
    return df


def _time_autoscale(x: np.ndarray) -> tuple[float, str]:
    """Mirror PlotManager._draw_xy_overlay's time-axis autoscaling.

    Returns ``(scale, unit_suffix)`` — divide-by factor not used; data is
    multiplied by *scale* before going to the CSV, and the axis label gets
    ``"Time ({unit_suffix})"``.
    """
    t_max = float(np.nanmax(np.abs(x))) if x.size else 0.0
    if t_max >= 1.0:
        return 1.0, "s"
    if t_max >= 1e-3:
        return 1e3, "ms"
    if t_max >= 1e-6:
        return 1e6, r"\textmu s"
    return 1e9, "ns"


def _build_xy_table(
    run_files: list[tuple[str, str]],
    signals: list[str],
    x_col: str,
    *,
    y_columns_for_signal,
    equations: list | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]], dict[str, str]]:
    """
    Load each per-run CSV and assemble a wide LaTeX-ready DataFrame.

    *y_columns_for_signal(df, sig)* returns a list of ``(suffix, series)`` to
    emit for that signal — transient/DC supply one (``("", df[sig])``), Bode
    supplies two (``("mag", db_series), ("phase", phase_series)``).

    Every source file gets its own X column: transient runs have per-run
    adaptive timesteps, so plotting all runs against one shared time vector
    would silently distort every run but the first.

    Returns ``(wide_df, series_keys, series_x)``:
      series_keys – ``(run_id, base_signal, suffix, column_name)`` per curve.
                    The base signal is the name the user picked (without
                    mag/phase) so chipify-style colouring can group mag+phase
                    of the same (rid, sig) under one colour.
      series_x    – ``{column_name: x_column_name}`` for the matching X data.
    """
    cols: dict[str, np.ndarray] = {}
    series_keys: list[tuple[str, str, str, str]] = []
    series_x: dict[str, str] = {}

    for rid, fpath in run_files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if x_col not in df.columns:
            continue
        df = _apply_equations(df, equations)

        rid_tok = _safe_col(rid)
        x_name = f"x_{rid_tok}"
        n = 2
        while x_name in cols:  # same run id can appear once per testbench
            x_name = f"x_{rid_tok}_{n}"
            n += 1
        cols[x_name] = df[x_col].to_numpy()

        wrote_any = False
        for sig in signals:
            for suffix, series in y_columns_for_signal(df, sig):
                if series is None:
                    continue
                sig_tok = _safe_col(sig)
                suffix_tok = f"_{suffix}" if suffix else ""
                col_name = f"r_{rid_tok}_{sig_tok}{suffix_tok}"
                n = 2
                while col_name in cols:
                    col_name = f"r_{rid_tok}_{sig_tok}{suffix_tok}_{n}"
                    n += 1
                cols[col_name] = np.asarray(series, dtype=float)
                series_keys.append((rid, sig, suffix, col_name))
                series_x[col_name] = x_name
                wrote_any = True
        if not wrote_any:
            del cols[x_name]

    if not series_keys:
        return pd.DataFrame(), [], {}

    # NaN-pad to the longest column so all series fit in one CSV.
    max_len = max(len(v) for v in cols.values())
    padded: dict[str, np.ndarray] = {}
    for name, vec in cols.items():
        if len(vec) < max_len:
            pad = np.full(max_len - len(vec), np.nan)
            padded[name] = np.concatenate([vec, pad])
        else:
            padded[name] = vec
    return pd.DataFrame(padded), series_keys, series_x


# ── chipify-styled colour / alpha plumbing ────────────────────────────────────
# Colours, line styles and the legend come from overlay_style.OverlayStyle, the
# policy the figures draw with, so a group or signal gets the same colour in the
# figure and in the .tex, and a grouped figure exports grouped.

#: matplotlib line style -> pgfplots dash pattern (solid is pgfplots' default).
_TEX_DASH = {"--": "dashed", "-.": "dashdotted", ":": "dotted"}

#: Colour key role (``OverlayStyle.color_key``) -> ``\definecolor`` name prefix.
_TEX_COLOR_PREFIX = {"group": "cfGrp", "signal": "cfSig", "run": "cfRun"}


def _rgb_tex(color) -> str:
    """An RGB(A) tuple or a hex string as ``"r,g,b"`` for ``\\definecolor``."""
    if isinstance(color, str):
        return _hex_rgb(color)
    r, g, b = (float(c) for c in tuple(color)[:3])
    return f"{r:.4f},{g:.4f},{b:.4f}"


@dataclass
class _CurveStyles:
    """What the overlay writers need from the shared ``OverlayStyle``."""

    #: ``[(name, "r,g,b"), …]`` to emit as ``\definecolor`` blocks.
    color_defs: list[tuple[str, str]]
    #: ``{(rid, base_sig): "color=<name>[, dashed]"}`` for each curve.
    curve_opts: dict[tuple[str, str], str]
    #: Curve opacity: 1.0 up to 50 curves, else max(0.05, 50/N).
    alpha: float
    #: The ``\addlegendimage`` / ``\addlegendentry`` lines, in legend order.
    legend_lines: list[str]
    #: The column the curves are grouped by, ``""`` when ungrouped.
    group: str


def _chipify_curve_styles(
    series_keys: list[tuple[str, str, str, str]],
    signals: list[str],
    run_ids: list[str],
    group_map: dict | None = None,
    group_label: str = "",
) -> _CurveStyles:
    """Per-curve pgfplots options and legend, as the figure draws them.

    *signals* and *run_ids* are what the figure gets: the selected signals and
    the runs whose files were found. *group_map* (``run_id -> value``) and
    *group_label* group the curves exactly as ``PlotManager`` does: colour by
    group value, line style by signal, a legend entry per group value.
    """
    style = OverlayStyle(signals, run_ids, group_map=group_map,
                         group_label=group_label, shade=_viridis)
    names: dict = {}
    color_defs: list[tuple[str, str]] = []

    def _opts(key, linestyle: str) -> str:
        if key not in names:
            role, i = key
            if role == "neutral":
                names[key] = "gray"
            else:
                names[key] = ("cfFail" if role == "fail"
                              else f"{_TEX_COLOR_PREFIX[role]}{i}")
                color_defs.append((names[key], _rgb_tex(style.resolve(key))))
        dash = _TEX_DASH.get(linestyle)
        return f"color={names[key]}" + (f", {dash}" if dash else "")

    legend_lines: list[str] = []
    for key, linestyle, label in style.legend_entries():
        legend_lines.append(
            f"    \\addlegendimage{{{_opts(key, linestyle)}, line width=1pt}}")
        legend_lines.append(f"    \\addlegendentry{{{_tex_escape(label)}}}")

    # A "curve" is one (rid, base_sig) pair — mag+phase share it for Bode.
    curve_opts: dict[tuple[str, str], str] = {}
    for rid, sig, _suffix, _col in series_keys:
        if (rid, sig) not in curve_opts:
            curve_opts[(rid, sig)] = _opts(style.color_key(rid, sig),
                                           style.linestyle(sig))
    n_curves = len(curve_opts)
    alpha = 1.0 if n_curves <= 50 else max(0.05, 50.0 / n_curves)
    return _CurveStyles(color_defs, curve_opts, alpha, legend_lines,
                        style.group_label if style.grouped else "")


def _overlay_title(prefix: str, series_keys, group: str) -> str:
    n_runs = len({rid for rid, _s, _u, _c in series_keys})
    n_sigs = len({sig for _r, sig, _u, _c in series_keys})
    title = f"{prefix} \\textemdash{{}} {n_runs} run(s) {n_sigs} signal(s)"
    if n_sigs > 1:
        title = (f"{prefix} \\textemdash{{}} {n_runs} run(s) "
                 f"$\\times$ {n_sigs} signal(s)")
    if group:
        title += f", grouped by {_tex_escape(group)}"
    return title


def _emit_color_defs(color_defs: list[tuple[str, str]]) -> str:
    """Format \\definecolor blocks for the colour table at the top of the .tex."""
    if not color_defs:
        return ""
    return "\n".join(
        f"\\definecolor{{{name}}}{{rgb}}{{{rgb}}}" for name, rgb in color_defs
    ) + "\n"


def _write_overlay_tex(
    output_dir: str,
    name: str,
    csv_filename: str,
    series_keys: list[tuple[str, str, str, str]],
    series_x: dict[str, str],
    styles: _CurveStyles,
    *,
    xlabel: str,
    ylabel: str,
    xmode: str = "normal",
    title_prefix: str = "Overlay",
) -> str:
    """Emit a single-axis pgfplots overlay styled like chipify's GUI plot."""
    tex_filename = f"{name}_plot.tex"
    title = _overlay_title(title_prefix, series_keys, styles.group)

    addplots: list[str] = []
    for rid, sig, _suffix, col in series_keys:
        opts = styles.curve_opts.get((rid, sig), "color=black")
        x_col = series_x.get(col, col)
        addplots.append(
            f"    \\addplot+ [{opts}, no marks,"
            f" line width=0.6pt, opacity={styles.alpha:.3f}]"
            f" table [x={x_col}, y={col}] {{\\datatable}};"
        )

    plot_block = "\n".join(styles.legend_lines + addplots)
    xmode_line = f"    xmode={xmode},\n" if xmode and xmode != "normal" else ""
    color_block = _emit_color_defs(styles.color_defs)

    tex_content = (
        f"% {title_prefix}\n"
        f"\\pgfplotstableread[col sep=comma]{{{csv_filename}}}\\datatable\n\n"
        f"{color_block}"
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"    title={{{title}}},\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        f"{xmode_line}"
        "    scale only axis, width=10cm, height=6cm,\n"
        "    grid=major, grid style={solid, gray!20},\n"
        "    scaled x ticks=false, scaled y ticks=false,\n"
        "    x tick label style={/pgf/number format/.cd, fixed, precision=3},\n"
        "    y tick label style={/pgf/number format/.cd, fixed, precision=4},\n"
        "    legend cell align={left},\n"
        "    legend style={\n"
        "        align=left, fill=white, draw=black!30, rounded corners=1pt,\n"
        "        at={(0.97,0.97)}, anchor=north east, font=\\small\n"
        "    },\n"
        "    axis line style={thick}, tick style={thick, black}\n"
        "]\n"
        f"{plot_block}\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}\n"
    )

    out_path = Path(output_dir) / tex_filename
    out_path.write_text(tex_content, encoding="utf-8")
    return out_path


# ── Analysis-overlay LaTeX export entry points ────────────────────────────────

def generate_transient_latex_export(
    out_dir: str,
    name: str,
    tran_dir: str,
    run_ids: list[str],
    signals: list[str],
    equations: list | None = None,
    group_map: dict | None = None,
    group_label: str = "",
) -> tuple[str, str]:
    """Export a transient overlay (time vs signal) as CSV + pgfplots .tex.

    *group_map* (``run_id -> value``, ``transient_loader.run_group_map``) and
    *group_label* group the curves as the figure does.

    Returns ``(csv_path, tex_path)``. Raises ``ValueError`` if no data could be
    collected (no matching runs, missing 'time' column, etc.).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_files = _collect_run_files(tran_dir, run_ids)

    def y_for(df, sig):
        if sig in df.columns:
            return [("", df[sig])]
        return []

    table, series_keys, series_x = _build_xy_table(
        run_files, signals, x_col="time",
        y_columns_for_signal=y_for, equations=equations,
    )
    if table.empty or not series_keys:
        raise ValueError("No transient data available for the current selection.")

    # Auto-scale time to a readable unit (s / ms / µs / ns), matching the GUI.
    # Every run has its own time column (adaptive timesteps) — scale them all.
    x_cols = sorted(set(series_x.values()))
    all_x = np.concatenate([table[c].to_numpy() for c in x_cols])
    t_scale, t_unit = _time_autoscale(all_x)
    if t_scale != 1.0:
        for c in x_cols:
            table[c] = table[c] * t_scale

    csv_filename = f"{name}_plot.csv"
    csv_path = Path(out_dir) / csv_filename
    table.to_csv(csv_path, index=False)
    styles = _chipify_curve_styles(series_keys, signals,
                                   [rid for rid, _ in run_files],
                                   group_map, group_label)
    tex_path = _write_overlay_tex(
        out_dir, name, csv_filename, series_keys, series_x, styles,
        xlabel=f"Time ({t_unit})", ylabel="Signal Value",
        title_prefix="Transient Overlay",
    )
    return csv_path, tex_path


def generate_dc_sweep_latex_export(
    out_dir: str,
    name: str,
    dc_dir: str,
    run_ids: list[str],
    signals: list[str],
    equations: list | None = None,
    group_map: dict | None = None,
    group_label: str = "",
) -> tuple[str, str]:
    """Export a DC sweep overlay (sweep vs signal) as CSV + pgfplots .tex,
    grouped as in :func:`generate_transient_latex_export`."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_files = _collect_run_files(dc_dir, run_ids)

    def y_for(df, sig):
        if sig in df.columns:
            return [("", df[sig])]
        return []

    table, series_keys, series_x = _build_xy_table(
        run_files, signals, x_col="sweep",
        y_columns_for_signal=y_for, equations=equations,
    )
    if table.empty or not series_keys:
        raise ValueError("No DC sweep data available for the current selection.")

    csv_filename = f"{name}_plot.csv"
    csv_path = Path(out_dir) / csv_filename
    table.to_csv(csv_path, index=False)
    styles = _chipify_curve_styles(series_keys, signals,
                                   [rid for rid, _ in run_files],
                                   group_map, group_label)
    tex_path = _write_overlay_tex(
        out_dir, name, csv_filename, series_keys, series_x, styles,
        xlabel="Sweep", ylabel="Signal Value",
        title_prefix="DC Sweep Overlay",
    )
    return csv_path, tex_path


def generate_bode_latex_export(
    out_dir: str,
    name: str,
    ac_dir: str,
    run_ids: list[str],
    signals: list[str],
    equations: list | None = None,
    group_map: dict | None = None,
    group_label: str = "",
) -> tuple[str, str]:
    """Export a Bode overlay (frequency vs mag-in-dB and phase) as CSV + .tex.

    Uses pgfplots' ``groupplots`` library to stack magnitude (dB) above phase
    (degrees) with a shared logarithmic frequency axis. The phase is the
    figure's continuous one (``bode.continuous_phase``), and the curves are
    grouped as in :func:`generate_transient_latex_export`.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_files = _collect_run_files(ac_dir, run_ids)

    def y_for(df, sig):
        mag_col = f"{sig}_mag"
        ph_col = f"{sig}_phase"
        out: list[tuple[str, pd.Series]] = []
        if mag_col in df.columns:
            mag = np.maximum(np.abs(df[mag_col].to_numpy()), 1e-30)
            out.append(("mag", pd.Series(20.0 * np.log10(mag))))
        if ph_col in df.columns:
            out.append(("phase", pd.Series(continuous_phase(df[ph_col]))))
        return out

    table, series_keys, series_x = _build_xy_table(
        run_files, signals, x_col="frequency",
        y_columns_for_signal=y_for, equations=equations,
    )
    if table.empty or not series_keys:
        raise ValueError("No AC/Bode data available for the current selection.")

    csv_filename = f"{name}_plot.csv"
    csv_path = Path(out_dir) / csv_filename
    table.to_csv(csv_path, index=False)

    styles = _chipify_curve_styles(series_keys, signals,
                                   [rid for rid, _ in run_files],
                                   group_map, group_label)

    mag_keys = [k for k in series_keys if k[2] == "mag"]
    phase_keys = [k for k in series_keys if k[2] == "phase"]

    def _plots_block(keys: list[tuple[str, str, str, str]]) -> str:
        rows: list[str] = []
        for rid, sig, _suffix, col in keys:
            opts = styles.curve_opts.get((rid, sig), "color=black")
            x_col = series_x.get(col, col)
            rows.append(
                f"        \\addplot+ [{opts}, no marks,"
                f" line width=0.5pt, opacity={styles.alpha:.3f}]"
                f" table [x={x_col}, y={col}] {{\\datatable}};"
            )
        return "\n".join(rows) if rows else ""

    # Show the legend on the magnitude subplot only (it's the same colour
    # mapping for the phase plot below).
    legend_block = "\n".join("    " + ln.lstrip() for ln in styles.legend_lines)

    title = _overlay_title("Bode Plot", series_keys, styles.group)

    tex_filename = f"{name}_plot.tex"
    color_block = _emit_color_defs(styles.color_defs)
    tex_content = (
        f"% Bode plot for {name}\n"
        "% Requires: \\usepgfplotslibrary{groupplots}\n"
        f"\\pgfplotstableread[col sep=comma]{{{csv_filename}}}\\datatable\n\n"
        f"{color_block}"
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        "    group style={\n"
        "        group size=1 by 2, vertical sep=12pt,\n"
        "        xticklabels at=edge bottom,\n"
        "    },\n"
        "    scale only axis, width=10cm, height=4cm,\n"
        "    xmode=log, grid=both, grid style={solid, gray!20},\n"
        "    scaled y ticks=false,\n"
        "    y tick label style={/pgf/number format/.cd, fixed, precision=2},\n"
        "    legend cell align={left},\n"
        "    legend style={\n"
        "        align=left, fill=white, draw=black!30, rounded corners=1pt,\n"
        "        at={(0.97,0.97)}, anchor=north east, font=\\small\n"
        "    },\n"
        "    axis line style={thick}, tick style={thick, black}\n"
        "]\n"
        f"    \\nextgroupplot[title={{{title}}}, ylabel={{Magnitude (dB)}}]\n"
        f"{legend_block}\n"
        f"{_plots_block(mag_keys)}\n"
        "    \\nextgroupplot[ylabel={Phase ($^\\circ$)}, xlabel={Frequency (Hz)}]\n"
        f"{_plots_block(phase_keys)}\n"
        "\\end{groupplot}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}\n"
    )
    tex_path = Path(out_dir) / tex_filename
    tex_path.write_text(tex_content, encoding="utf-8")
    return csv_path, tex_path


# ── Scatter (shmoo) plot ──────────────────────────────────────────────────────

#: The scatter's pass / fail colours, as ``plot_manager`` draws them
#: (``_PASS_COLOR`` / ``_FAIL_COLOR``); a test keeps the two in step.
SCATTER_PASS_HEX = "#2ecc71"
SCATTER_FAIL_HEX = "#e74c3c"

_TEX_SPECIALS = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
    "&": r"\&", "#": r"\#", "^": r"\^{}", "_": r"\_", "%": r"\%",
    "~": r"\textasciitilde{}",
}


def _tex_escape(text) -> str:
    """*text* as literal LaTeX: measurement names carry ``_`` and SPICE syntax."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in str(text))


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(f"{int(h[i:i + 2], 16) / 255:.4f}" for i in (0, 2, 4))


def _axis_coords(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Numeric coordinates for one scatter axis, plus tick labels for text.

    matplotlib draws a column of strings (a corner sweep: ``tt``, ``ss``) as
    categories at 0, 1, … in order of first appearance. A pgfplots axis needs
    numbers, so the export puts the categories at the same positions and
    names the ticks.
    """
    values = series.dropna()
    if len(values) and all(isinstance(v, str) for v in values):
        labels = list(dict.fromkeys(values))
        pos = {lab: float(i) for i, lab in enumerate(labels)}
        return np.array([pos.get(v, np.nan) for v in series], dtype=float), labels
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float), []


def _tick_options(axis: str, labels: list[str]) -> str:
    if not labels:
        return ""
    ticks = ",".join(str(i) for i in range(len(labels)))
    names = ",".join("{" + _tex_escape(lab) + "}" for lab in labels)
    return f"    {axis}tick={{{ticks}}}, {axis}ticklabels={{{names}}},\n"


def generate_scatter_latex_export(
    out_dir: str,
    name: str,
    rows: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "",
) -> tuple[Path, Path]:
    """Export a scatter (shmoo) plot as CSV + pgfplots .tex.

    *rows* are the rows the scatter drew, the frame ``draw_adv_plot`` returns
    with its artist, so the file holds exactly the plotted points. Verdicts
    (``data_loader.scatter_pass_mask``), the group order
    (``data_loader.sort_group_values``) and the colours are the figure's:
    ungrouped, passing points are green and failing ones red; grouped by
    *group_col*, colour is the group value and a failing point is a cross.

    Returns ``(csv_path, tex_path)``. Raises ``ValueError`` when there is no
    point to plot.
    """
    from chipify import data_loader as _dl

    if (rows is None or len(rows) == 0
            or x_col not in rows.columns or y_col not in rows.columns):
        raise ValueError("No scatter data available for the current selection.")

    x, x_labels = _axis_coords(rows[x_col])
    y, y_labels = _axis_coords(rows[y_col])
    drawn = ~(np.isnan(x) | np.isnan(y))
    passed = _dl.scatter_pass_mask(rows).to_numpy(dtype=bool)
    group = _dl.group_column(rows, group_col)

    color_defs: list[tuple[str, str]] = []
    # (column key, points, colour name, drawn as a cross)
    series: list[tuple[str, np.ndarray, str, bool]] = []
    # (colour name, cross, label)
    legend: list[tuple[str, bool, str]] = []
    if group:
        col = rows[group]
        values = _dl.sort_group_values(col.dropna().unique())
        for i, value in enumerate(values):
            cname = f"cfGrp{i}"
            r, g, b = category_color(i, len(values))
            color_defs.append((cname, f"{r:.4f},{g:.4f},{b:.4f}"))
            members = (col == value).to_numpy(dtype=bool)
            series.append((f"g{i}_pass", members & passed, cname, False))
            series.append((f"g{i}_fail", members & ~passed, cname, True))
            legend.append((cname, False, f"{group}={value}"))
        # The figure draws a point without a group value grey; dropping it
        # here would lose a point without a word.
        unassigned = col.isna().to_numpy(dtype=bool)
        series.append(("nogroup_pass", unassigned & passed, "gray", False))
        series.append(("nogroup_fail", unassigned & ~passed, "gray", True))
        if not passed.all():
            legend.append(("gray", True, "Fail"))
    else:
        color_defs += [("cfPass", _hex_rgb(SCATTER_PASS_HEX)),
                       ("cfFail", _hex_rgb(SCATTER_FAIL_HEX))]
        series += [("pass", passed, "cfPass", False),
                   ("fail", ~passed, "cfFail", False)]
        legend += [("cfPass", False, "Pass"), ("cfFail", False, "Fail")]

    columns: dict[str, np.ndarray] = {}
    plotted: list[tuple[str, str, bool]] = []
    for key, mask, cname, cross in series:
        keep = mask & drawn
        if not keep.any():
            continue
        columns[f"x_{key}"] = x[keep]
        columns[f"y_{key}"] = y[keep]
        plotted.append((key, cname, cross))
    if not plotted:
        raise ValueError("No scatter data available for the current selection.")

    # NaN-pad to the longest series so every series fits in one CSV; pgfplots
    # discards the "nan" cells.
    max_len = max(len(v) for v in columns.values())
    table = pd.DataFrame({
        k: np.concatenate([v, np.full(max_len - len(v), np.nan)])
        for k, v in columns.items()
    })
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_filename = f"{name}_plot.csv"
    csv_path = Path(out_dir) / csv_filename
    table.to_csv(csv_path, index=False, na_rep="nan")

    def _mark(cname: str, cross: bool) -> str:
        if cross:
            return (f"mark=x, mark size=2.6pt, color={cname}, "
                    "mark options={line width=1.1pt}")
        return (f"mark=*, mark size=2pt, color={cname}, mark options="
                "{fill opacity=0.7, draw=black!70, line width=0.3pt}")

    # Explicit legend images and `forget plot` data: the legend lists each
    # group once, also a group whose points all failed, and never the
    # per-verdict series it is drawn as.
    lines = [f"    \\addlegendimage{{only marks, {_mark(cname, cross)}}}\n"
             f"    \\addlegendentry{{{_tex_escape(label)}}}"
             for cname, cross, label in legend]
    lines += [f"    \\addplot [only marks, forget plot, {_mark(cname, cross)}]"
              f" table [x=x_{key}, y=y_{key}] {{\\datatable}};"
              for key, cname, cross in plotted]

    title = f"Shmoo Plot: {_tex_escape(y_col)} vs {_tex_escape(x_col)}"
    if group:
        title += f", grouped by {_tex_escape(group)}"
    tex_content = (
        f"% Scatter plot: {y_col} vs {x_col}"
        + (f", grouped by {group}" if group else "") + "\n"
        f"\\pgfplotstableread[col sep=comma]{{{csv_filename}}}\\datatable\n\n"
        f"{_emit_color_defs(color_defs)}"
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"    title={{{title}}},\n"
        f"    xlabel={{{_tex_escape(x_col)}}},\n"
        f"    ylabel={{{_tex_escape(y_col)}}},\n"
        f"{_tick_options('x', x_labels)}"
        f"{_tick_options('y', y_labels)}"
        "    scale only axis, width=10cm, height=6cm,\n"
        "    grid=major, grid style={dashed, gray!30},\n"
        "    legend cell align={left},\n"
        "    legend style={\n"
        "        align=left, fill=white, draw=black!30, rounded corners=1pt,\n"
        "        at={(0.97,0.97)}, anchor=north east, font=\\small\n"
        "    },\n"
        "    axis line style={thick}, tick style={thick, black}\n"
        "]\n"
        + "\n".join(lines) + "\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}\n"
    )
    tex_path = Path(out_dir) / f"{name}_plot.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    return csv_path, tex_path


def _histogram_xy(data, bins):
    """``(x_hist, y_hist, counts)`` for a pgfplots ``ybar interval`` plot."""
    # 'ybar interval' in PGFPlots needs the bin edges, not the centers!
    counts, edges = np.histogram(data, bins=bins, density=True)

    # For 'ybar interval', X and Y must be the same length; the last Y is ignored.
    x_hist = edges
    y_hist = np.append(counts, counts[-1] if len(counts)>0 else 0)
    return x_hist, y_hist, counts


def _histogram_fit(data, dist_type, counts):
    """``(fit_x, fit_y, fit_params, fit_legend_latex)`` of one distribution."""
    import scipy.stats as stats  # only needed for the histogram fit path
    fit_x = None
    fit_y = None
    fit_params = {}
    fit_legend_latex = ""

    if dist_type != "None" and len(data) > 1:
        fit_x = np.linspace(min(data), max(data), 100)
        try:
            if dist_type == "Gauss (Normal)":
                mu, std = stats.norm.fit(data)
                fit_y = stats.norm.pdf(fit_x, mu, std)
                fit_params = {'mu': mu, 'sigma': std}
                fit_legend_latex = r"Gaussian fit \\ $\mu=\formatnum{3}{\muVal}$ \\ $\sigma=\formatnum{3}{\sigmaVal}$"
                
            elif dist_type == "KDE (Smoothed)":
                kde = stats.gaussian_kde(data)
                fit_y = kde(fit_x)
                fit_legend_latex = "KDE Smooth"
                
            elif dist_type == "Uniform":
                loc, scale = stats.uniform.fit(data)
                fit_y = stats.uniform.pdf(fit_x, loc, scale)
                fit_params = {'loc': loc, 'scale': scale}
                fit_legend_latex = r"Uniform fit \\ $loc=\formatnum{3}{\locVal}$ \\ $scale=\formatnum{3}{\scaleVal}$"
                
            elif dist_type == "Log-Normal":
                shape, loc, scale = stats.lognorm.fit(data)
                fit_y = stats.lognorm.pdf(fit_x, shape, loc, scale)
                fit_params = {'shape': shape, 'loc': loc, 'scale': scale}
                fit_legend_latex = r"Log-Normal fit \\ $s=\formatnum{3}{\shapeVal}$ \\ $loc=\formatnum{3}{\locVal}$ \\ $scale=\formatnum{3}{\scaleVal}$"
                
            elif dist_type == "Exponential":
                loc, scale = stats.expon.fit(data)
                fit_y = stats.expon.pdf(fit_x, loc, scale)
                fit_params = {'loc': loc, 'scale': scale}
                fit_legend_latex = r"Exponential fit \\ $loc=\formatnum{3}{\locVal}$ \\ $scale=\formatnum{3}{\scaleVal}$"
                
            elif dist_type == "Chi-Squared":
                df_stat, loc, scale = stats.chi2.fit(data)
                fit_y = stats.chi2.pdf(fit_x, df_stat, loc=loc, scale=scale)
                fit_params = {'df': df_stat, 'loc': loc, 'scale': scale}
                fit_legend_latex = r"$\chi^2$ fit \\ $df=\formatnum{3}{\dfVal}$ \\ $loc=\formatnum{3}{\locVal}$ \\ $scale=\formatnum{3}{\scaleVal}$"

            if fit_y is not None:
                max_h = max(counts) if len(counts) > 0 else 1.0
                fit_y = np.nan_to_num(fit_y, nan=0.0, posinf=0.0, neginf=0.0)
                fit_y = np.clip(fit_y, 0.0, max_h * 1.5)
        except Exception:
            fit_x, fit_y = None, None
            fit_legend_latex = ""
    return fit_x, fit_y, fit_params, fit_legend_latex


#: Fit parameter -> its symbol in a grouped histogram's legend entry, as the
#: ungrouped entry writes it.
_FIT_SYMBOLS = {"mu": r"\mu", "sigma": r"\sigma", "loc": "loc",
                "scale": "scale", "shape": "s", "df": "df"}


def _letters(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA: a TeX macro name cannot hold digits."""
    out = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def generate_latex_export(param_name, data_series, dist_type, bins, output_dir,
                          group_series=None):
    """Write ``<param>_plot.csv`` + ``.tex`` for one measurement's distribution.

    *group_series* is the swept input the histogram is grouped by: a Series
    named after its column and aligned with *data_series* by index. It draws
    one histogram and fit per group value, in the figure's order and colours
    (``data_loader.histogram_groups``, ``palette.category_color``).

    Returns ``(csv_path, tex_path)``, or ``None`` when the series holds no
    numeric data (callers must not report success then).
    """
    data = data_series.dropna().values
    if len(data) == 0:
        return None
    if group_series is not None and group_series.name:
        return _generate_grouped_histogram(param_name, data_series, group_series,
                                           dist_type, bins, output_dir)

    # 1. Compute the histogram.
    x_hist, y_hist, counts = _histogram_xy(data, bins)

    # 2. Compute the fit curve and its parameters.
    fit_x, fit_y, fit_params, fit_legend_latex = _histogram_fit(
        data, dist_type, counts)

    # 3. Generate the CSV with NaN padding so all columns are the same length.
    max_len = max(len(x_hist), len(fit_x) if fit_x is not None else 0)
    
    csv_dict = {
        'x_hist': np.pad(x_hist, (0, max_len - len(x_hist)), constant_values=np.nan),
        'y_hist': np.pad(y_hist, (0, max_len - len(y_hist)), constant_values=np.nan)
    }
    
    if fit_x is not None and fit_y is not None:
        csv_dict['x_curve'] = np.pad(fit_x, (0, max_len - len(fit_x)), constant_values=np.nan)
        csv_dict['y_curve'] = np.pad(fit_y, (0, max_len - len(fit_y)), constant_values=np.nan)
    else:
        csv_dict['x_curve'] = np.full(max_len, np.nan)
        csv_dict['y_curve'] = np.full(max_len, np.nan)
        
    for key, val in fit_params.items():
        arr = np.full(max_len, np.nan)
        arr[0] = val  # Write the value into row 0 exactly.
        csv_dict[key] = arr

    df = pd.DataFrame(csv_dict)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Measurement names may carry SPICE syntax (v(out), slashes …) — sanitise
    # for the filesystem; titles/labels keep the original name.
    stem = _safe_col(param_name)
    csv_filename = f"{stem}_plot.csv"
    csv_path = Path(output_dir) / csv_filename
    df.to_csv(csv_path, index=False)

    # 4. Generate the LaTeX code.
    tex_filename = f"{stem}_plot.tex"
    
    read_macros = ""
    for key in fit_params.keys():
        read_macros += f"\\pgfplotstablegetelem{{0}}{{{key}}}\\of\\datatable \\let\\{key}Val\\pgfplotsretval\n"

    # Build the TeX string.
    safe_param_name = param_name.replace('_', r'\_')
    tex_content = f"""% Plot for {param_name}
\\pgfplotstableread[col sep=comma]{{{csv_filename}}}\\datatable

% Read parameters from row 0
{read_macros}
% Define helper macro (if not already defined globally)
\\providecommand{{\\formatnum}}[2]{{\\pgfmathprintnumber[fixed, precision=#1, zerofill]{{#2}}}}

\\begin{{center}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={{Simulation: {safe_param_name}}},
    xlabel={{Simulated Value}},
    ylabel={{Probability density}},
    ymin=0, enlarge x limits=0.05,
    grid=major, grid style={{solid, gray!20}},
    legend cell align={{left}},
    legend style={{
        align=left, fill=white, draw=black!30, rounded corners=1pt,
        at={{(0.95,0.95)}}, anchor=north east
    }},
    width=10cm, height=8cm,
    axis line style={{thick}}, tick style={{thick, black}}
]
    % --- 1. Histogram ---
    \\addplot [ybar interval, fill=blue!30, draw=blue!70] 
        table [x=x_hist, y=y_hist] {{\\datatable}};
    \\addlegendentry{{Histogram}}
"""

    if fit_x is not None:
        tex_content += f"""
    % --- 2. Fit curve ---
    \\addplot [color=red, very thick, no marks] 
        table [x=x_curve, y=y_curve] {{\\datatable}};
    \\addlegendentry{{{fit_legend_latex}}}
"""

    tex_content += """\\end{axis}
\\end{tikzpicture}
\\end{center}
"""

    tex_path = Path(output_dir) / tex_filename
    tex_path.write_text(tex_content)
    return csv_path, tex_path


def _generate_grouped_histogram(param_name, data_series, group_series,
                                dist_type, bins, output_dir):
    """:func:`generate_latex_export` grouped: one histogram and fit per group.

    As the figure draws it: the groups of ``data_loader.histogram_groups``, the
    colour of group *i* of *n* is ``category_color(i, n)`` (also when an empty
    group is left out), translucent bars, the fit in the group's colour, and one
    legend entry per group carrying its fit parameters.
    """
    from chipify import data_loader as _dl

    group = str(group_series.name)
    value_col = f"{group}__value"         # any name but the group's
    # Every row of the group column, also those without a value: the figure
    # counts a group whose runs all lack the measurement, and the colours of
    # the groups after it depend on that count.
    rows = pd.DataFrame({group: group_series})
    rows[value_col] = data_series        # aligned by index
    groups = _dl.histogram_groups(rows, value_col, group)

    columns: dict[str, np.ndarray] = {}
    color_defs: list[tuple[str, str]] = []
    read_macros: list[str] = []
    plots: list[str] = []
    for i, (label, grp) in enumerate(groups):
        grp_data = grp.to_numpy(dtype=float)
        if len(grp_data) == 0:
            continue
        cname = f"cfGrp{i}"
        color_defs.append((cname, _rgb_tex(category_color(i, len(groups)))))
        x_hist, y_hist, counts = _histogram_xy(grp_data, bins)
        fit_x, fit_y, fit_params, _ = _histogram_fit(grp_data, dist_type, counts)
        key = f"g{i}"
        columns[f"x_hist_{key}"] = x_hist
        columns[f"y_hist_{key}"] = y_hist

        entry = _tex_escape(label)
        have_fit = fit_x is not None and fit_y is not None
        if have_fit and fit_params:
            parts = []
            for pname, pval in fit_params.items():
                col = f"{pname}_{key}"
                macro = f"\\{pname}Val{_letters(i)}"
                columns[col] = np.array([pval], dtype=float)
                read_macros.append(f"\\pgfplotstablegetelem{{0}}{{{col}}}"
                                   f"\\of\\datatable \\let{macro}\\pgfplotsretval")
                parts.append(f"${_FIT_SYMBOLS.get(pname, pname)}"
                             f"=\\formatnum{{3}}{{{macro}}}$")
            entry += " (" + ", ".join(parts) + ")"
        # An explicit swatch: the legend image of a `ybar interval` plot is a
        # thin outline that does not show the group's fill colour.
        bars = f"fill={cname}, fill opacity=0.5, draw=black, line width=0.3pt"
        plots.append(
            f"    % --- group {i} ---\n"
            f"    \\addlegendimage{{area legend, {bars}}}\n"
            f"    \\addlegendentry{{{entry}}}\n"
            f"    \\addplot [ybar interval, {bars}, forget plot]\n"
            f"        table [x=x_hist_{key}, y=y_hist_{key}] {{\\datatable}};\n"
        )
        if have_fit:
            columns[f"x_curve_{key}"] = fit_x
            columns[f"y_curve_{key}"] = fit_y
            plots.append(
                f"    \\addplot [color={cname}, very thick, no marks, forget plot]\n"
                f"        table [x=x_curve_{key}, y=y_curve_{key}] {{\\datatable}};\n"
            )
    if not columns:
        return None

    # NaN-pad to the longest column so all series fit in one CSV; pgfplots
    # discards the "nan" cells.
    max_len = max(len(v) for v in columns.values())
    table = pd.DataFrame({
        k: np.concatenate([v, np.full(max_len - len(v), np.nan)])
        for k, v in columns.items()
    })
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = _safe_col(param_name)
    csv_filename = f"{stem}_plot.csv"
    csv_path = Path(output_dir) / csv_filename
    table.to_csv(csv_path, index=False, na_rep="nan")

    tex_content = (
        f"% Plot for {param_name}, grouped by {group}\n"
        f"\\pgfplotstableread[col sep=comma]{{{csv_filename}}}\\datatable\n\n"
        "% Read parameters from row 0\n"
        + "".join(m + "\n" for m in read_macros) + "\n"
        "% Define helper macro (if not already defined globally)\n"
        "\\providecommand{\\formatnum}[2]{\\pgfmathprintnumber"
        "[fixed, precision=#1, zerofill]{#2}}\n\n"
        f"{_emit_color_defs(color_defs)}"
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"    title={{Simulation: {_tex_escape(param_name)},"
        f" grouped by {_tex_escape(group)}}},\n"
        "    xlabel={Simulated Value},\n"
        "    ylabel={Probability density},\n"
        "    ymin=0, enlarge x limits=0.05,\n"
        "    grid=major, grid style={solid, gray!20},\n"
        "    legend cell align={left},\n"
        "    legend style={\n"
        "        align=left, fill=white, draw=black!30, rounded corners=1pt,\n"
        "        at={(0.95,0.95)}, anchor=north east\n"
        "    },\n"
        "    width=10cm, height=8cm,\n"
        "    axis line style={thick}, tick style={thick, black}\n"
        "]\n"
        + "".join(plots)
        + "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}\n"
    )
    tex_path = Path(output_dir) / f"{stem}_plot.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    return csv_path, tex_path