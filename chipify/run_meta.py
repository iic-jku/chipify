# Copyright (c) 2026 Santiago Hofwimmer
"""
run_meta.py – Sidecar metadata for simulation history runs.

For every `run_<ts>.csv` saved in `out/history/`, a companion
`run_<ts>.meta.json` file can be written and read.

Schema (v1)
-----------
{
  "schema_version": 1,
  "timestamp":      "2026-05-06T15:00:00",
  "yaml":           "datasheet.yaml",
  "host":           "mypc",
  "python":         "3.12.3",
  "ngspice":        "40",          // best-effort, "" if not found
  "git_commit":     "a1b2c3d",     // best-effort, "" if not in a repo
  "duration_s":     42.1,
  "total_runs":     500,
  "valid_runs":     497,
  "global_yield":   99.4,
  "templates_dir":  "",            // persisted netlist templates of this run
  "analysis_dirs":  {"transient": "analysis_data/transient/20260506_150000"},
  "notes":          "",
  "tags":           []
}

All fields except `schema_version` and `timestamp` are optional /
best-effort so older code loading new meta files does not crash.

``analysis_dirs`` / ``tran_dir`` are stored relative to the run's output root
(the folder holding ``history/``), so an ``out/`` folder that is moved or copied
keeps finding its waveforms. Sidecars from older versions hold absolute paths;
:func:`analysis_dirs_for` accepts both.
"""

from __future__ import annotations
import json
import os
import platform
import re
import subprocess
import sys
import datetime
import logging
from pathlib import Path

log = logging.getLogger("chipify.run_meta")

_SCHEMA_VERSION = 1


# ── helpers ───────────────────────────────────────────────────────────────────

_RE_NGSPICE_VERSION = re.compile(r"\bngspice-([\w.+]+)")


def _ngspice_version() -> str:
    """``ngspice-<n>`` from ``ngspice --version``.

    The banner opens with a row of asterisks, which is what the sidecar used
    to record as the version. Where only the Windows GUI build of ngspice is
    installed the version stays unknown: probing it opened a window for every
    sidecar written.
    """
    from chipify.engines.base import console_ngspice

    exe = console_ngspice()
    if exe is None:
        return ""
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return ""
    text = result.stdout or result.stderr or ""
    m = _RE_NGSPICE_VERSION.search(text)
    if m:
        return f"ngspice-{m.group(1)}"
    lines = [ln.strip("* \t") for ln in text.splitlines() if ln.strip("* \t")]
    return lines[0] if lines else ""


def _vacask_version() -> str:
    try:
        result = subprocess.run(
            ["vacask", "--version"], capture_output=True, text=True, timeout=5
        )
        first = (result.stdout or result.stderr or "").splitlines()[0]
        return first.strip()
    except Exception:
        return ""


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _meta_path(csv_path: str | os.PathLike[str]) -> Path:
    """Given a run CSV path, return the companion .meta.json path."""
    return Path(csv_path).with_suffix(".meta.json")


def output_root(csv_path: str | os.PathLike[str]) -> Path:
    """The output root a run CSV belongs to.

    ``<root>/history/run_<ts>.csv`` and ``<root>/simulation_results.csv`` both
    belong to ``<root>`` — the folder that also holds ``analysis_data/``.
    """
    parent = Path(csv_path).parent
    return parent.parent if parent.name == "history" else parent


def _relative_to_root(path: str, root: Path) -> str:
    """*path* relative to *root* when it lies inside it, else unchanged."""
    if not path:
        return path
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path


# ── public API ────────────────────────────────────────────────────────────────

def write_meta(
    csv_path: str | os.PathLike[str],
    *,
    yaml_name: str = "",
    duration_s: float | None = None,
    total_runs: int | None = None,
    valid_runs: int | None = None,
    global_yield: float | None = None,
    tran_dir: str = "",
    analysis_dirs: dict | None = None,
    templates_dir: str = "",
) -> str:
    """
    Write a sidecar .meta.json next to *csv_path*.

    ``analysis_dirs`` maps analysis kind (transient/dc/ac) to the per-run CSV
    directory of this run, so the GUI can resolve waveform data for any kind
    when a history run is re-loaded. ``tran_dir`` is the legacy
    transient-only alias. Both are written relative to the run's output root
    (see :func:`output_root`) when they lie inside it. ``templates_dir`` holds
    this run's persisted Jinja2 netlist templates (one ``<safe_tb>.spice|.sim``
    per testbench).

    Returns the path written, or "" on failure.
    """
    root = output_root(csv_path)
    tran_dir = _relative_to_root(str(tran_dir or ""), root)
    analysis_dirs = {
        kind: _relative_to_root(str(d or ""), root)
        for kind, d in (analysis_dirs or {}).items()
    }
    meta: dict = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "yaml": yaml_name,
        "host": platform.node(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ngspice": _ngspice_version(),
        "vacask": _vacask_version(),
        "git_commit": _git_commit(),
        "duration_s": duration_s,
        "total_runs": total_runs,
        "valid_runs": valid_runs,
        "global_yield": global_yield,
        "tran_dir": tran_dir,
        "analysis_dirs": dict(analysis_dirs or {}),
        "templates_dir": templates_dir,
        "notes": "",
        "tags": [],
    }
    path = _meta_path(csv_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log.info("Wrote run metadata: %s", path)
        return path
    except Exception as exc:
        log.warning("Could not write run metadata %s: %s", path, exc)
        return ""


def update_meta(csv_path: str | os.PathLike[str], **fields) -> dict:
    """
    Merge *fields* (e.g. ``notes=...``, ``tags=[...]``) into the sidecar of
    *csv_path*, creating a minimal sidecar if none exists yet.

    Returns the resulting meta dict ({} on write failure).
    """
    meta = read_meta(csv_path)
    if not meta:
        meta = {
            "schema_version": _SCHEMA_VERSION,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    meta.update(fields)
    path = _meta_path(csv_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta
    except Exception as exc:
        log.warning("Could not update run metadata %s: %s", path, exc)
        return {}


def read_meta(csv_path: str | os.PathLike[str]) -> dict:
    """
    Load the sidecar .meta.json for *csv_path*.

    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    path = _meta_path(csv_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read run metadata %s: %s", path, exc)
        return {}


def analysis_dirs_for(csv_path: str | os.PathLike[str],
                      meta: dict | None = None) -> dict[str, str]:
    """Absolute waveform directories recorded for the run *csv_path*.

    Reads the sidecar unless *meta* is given. Relative entries are resolved
    against the run's output root; absolute ones (older sidecars) are kept as
    they are. The legacy ``tran_dir`` fills in ``transient`` when
    ``analysis_dirs`` has no entry for it. Returns ``{}`` when nothing is
    recorded.
    """
    if meta is None:
        meta = read_meta(csv_path)
    root = output_root(csv_path)
    raw = meta.get("analysis_dirs")
    dirs = dict(raw) if isinstance(raw, dict) else {}
    if not dirs.get("transient") and meta.get("tran_dir"):
        dirs["transient"] = meta["tran_dir"]
    resolved: dict[str, str] = {}
    for kind, d in dirs.items():
        if not d:
            continue
        p = Path(str(d))
        resolved[str(kind)] = str(p if p.is_absolute() else root / p)
    return resolved
