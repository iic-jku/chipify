# Copyright (c) 2026 Santiago Hofwimmer
"""
run_store.py – Saving a finished run, the same way from the CLI and the GUI.

A run's files go into its output root (``settings.OUT_DIR``, or a batch
datasheet's own folder):

- ``simulation_results.csv`` and its sidecar: the latest run;
- ``history/run_<ts>.csv`` and its sidecar: the permanent copy;
- ``history/run_<ts>.datasheet.yaml``: the datasheet exactly as the run parsed
  it (its SHA-256 is in the sidecar, with the run's settings and engines);
- ``history/run_<ts>_templates/``: the netlist templates the run simulated,
  reusable with ``--templates-dir`` and by the per-sample netlist export;
- ``analysis_data/<kind>/.latest``: pointers to the run's waveform folders,
  for results that do not record their own.

The CLI and the GUI used to do this each in their own way, and the two drifted
apart: only the GUI archived the templates, and the CLI wrote no sidecar for
the latest CSV, so a sidecar left by an earlier GUI run described its results.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from chipify import data_loader as _dl
from chipify import run_meta, settings

log = logging.getLogger("chipify.run_store")

LATEST_CSV = "simulation_results.csv"


@dataclass
class SavedRun:
    """Where :func:`save_run` put a run."""
    latest_csv: Path
    #: The history copy, or None when it could not be written (see
    #: ``history_error``); the latest CSV is written regardless.
    history_csv: Path | None
    templates_dir: str = ""
    history_error: str = ""


def write_analysis_pointers(analysis_dirs: dict, out_dir=None) -> None:
    """Record each kind's latest data directory in ``analysis_data/<kind>/.latest``.

    The GUI's analysis-dir resolution reads these pointer files after a
    restart; the legacy ``tran_data/.latest`` pointer is also kept for
    transient so older consumers keep working.

    *out_dir* is the run's output root (default ``settings.OUT_DIR``). Batch
    mode passes each datasheet's own folder, so a pointer only ever names a
    directory of the datasheet it sits next to.
    """
    out_dir = Path(out_dir or settings.OUT_DIR)
    for kind, d in (analysis_dirs or {}).items():
        if not d:
            continue
        targets = [out_dir / "analysis_data" / kind / ".latest"]
        if kind == "transient":
            targets.append(out_dir / "tran_data" / ".latest")
        for ptr in targets:
            try:
                ptr.parent.mkdir(parents=True, exist_ok=True)
                ptr.write_text(str(d), encoding="utf-8")
            except Exception as exc:
                log.warning("Could not write %s pointer %s: %s", kind, ptr, exc)


def save_run(df, stim, *, out_dir: str | os.PathLike[str], duration_s: float | None,
             yaml_name: str = "") -> SavedRun:
    """Write the results of a finished run into *out_dir* (see module docstring).

    The latest CSV and its sidecar are always written; a failing history copy
    is reported in ``SavedRun.history_error`` instead of raising, because the
    results themselves are safe by then.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / LATEST_CSV
    df.to_csv(latest, index=False)

    analysis_dirs = dict(df.attrs.get("analysis_dirs", {}) or {})
    write_analysis_pointers(analysis_dirs, out_dir)

    s = _dl.result_summary(_dl.prepare_results(df))
    source = getattr(stim, "source_bytes", b"") or b""
    meta = dict(yaml_name=yaml_name, duration_s=duration_s, total_runs=s.total,
                valid_runs=s.valid, global_yield=round(s.yield_pct, 2),
                tran_dir=analysis_dirs.get("transient", ""),
                analysis_dirs=analysis_dirs,
                datasheet_sha256=hashlib.sha256(source).hexdigest() if source else "",
                settings=df.attrs.get("run_settings") or {},
                engines=df.attrs.get("engines") or {})

    history_csv: Path | None = None
    history_error = ""
    try:
        history_dir = out_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        history_csv = history_dir / f"run_{ts}.csv"
        df.to_csv(history_csv, index=False)
        if source:
            snapshot = history_dir / f"run_{ts}.datasheet.yaml"
            snapshot.write_bytes(source)
            meta["datasheet_snapshot"] = str(snapshot)
        meta["templates_dir"] = _archive_templates(
            stim, history_dir / f"run_{ts}_templates")
    except Exception as exc:  # noqa: BLE001 — the latest CSV is already safe
        log.warning("Could not save the history copy: %s", exc)
        history_error = str(exc)
        history_csv = None

    # The latest CSV needs its own sidecar: without it, a sidecar left by an
    # earlier run described these results, with another datasheet's name and
    # another run's waveform directories.
    run_meta.write_meta(latest, **meta)
    if history_csv is not None:
        run_meta.write_meta(history_csv, **meta)
    return SavedRun(latest, history_csv, meta.get("templates_dir", ""), history_error)


def _archive_templates(stim, dest_dir: Path) -> str:
    """Archive the run's netlist templates; "" when there are none or it fails."""
    from chipify.uikit.services import netlist_export
    try:
        return netlist_export.persist_templates(stim, dest_dir)
    except Exception as exc:  # noqa: BLE001 — a missing archive costs only exports
        log.warning("Could not persist netlist templates: %s", exc)
        return ""
