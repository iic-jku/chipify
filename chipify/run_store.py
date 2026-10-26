# Copyright (c) 2026 Santiago Hofwimmer
"""
run_store.py – A run's files on disk, from its first case to its last.

A run's files go into its output root (``settings.OUT_DIR``, or a batch
datasheet's own folder). When the sweep starts (:func:`begin_run`, called by
``simulator.run_sim``):

- ``history/run_<ts>.meta.json``: the sidecar, ``"complete": false`` for now;
- ``history/run_<ts>.datasheet.yaml``: the datasheet exactly as the run parsed
  it (its SHA-256 is in the sidecar, with the run's settings and engines);
- ``history/run_<ts>_templates/``: the netlist templates the run simulates,
  reusable with ``--templates-dir`` and by the per-sample netlist export;
- ``history/run_<ts>.rows.jsonl``: the journal. One line per finished worker
  batch, ``{"t": <unix time>, "rows": [...]}``, synced to disk as it arrives.

When it finishes (:func:`save_run`, from the CLI and the GUI alike):

- ``history/run_<ts>.csv``, and the sidecar completed;
- ``simulation_results.csv`` and its sidecar: the latest run;
- ``analysis_data/<kind>/.latest``: pointers to the run's waveform folders,
  for results that do not record their own;
- the journal is deleted.

A run that ends any other way (abort, crash, killed process, host standby)
keeps its sidecar, datasheet copy, templates and journal, and ``--resume``
continues it (:func:`find_unfinished`). History listings only show
``run_*.csv``, so an unfinished run never passes for a finished one.

A ``--batch`` keeps a record of its own (:class:`BatchRecord`), so a batch
that dies midway resumes without simulating its finished datasheets again.

Results used to be written only after the last case: one interrupted case
lost all the others. The CLI and the GUI also saved them each in their own
way, and the two drifted apart: only the GUI archived the templates.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from chipify import data_loader as _dl
from chipify import run_meta, settings

log = logging.getLogger("chipify.run_store")

LATEST_CSV = "simulation_results.csv"
JOURNAL_SUFFIX = ".rows.jsonl"
SNAPSHOT_SUFFIX = ".datasheet.yaml"

#: Settings that only shape how fast a run goes. A resumed run takes every
#: other setting over from its first part, because those shape the results.
_SPEED_KEYS = frozenset({"num_cores", "chunk_size", "process_start_method"})


@dataclass
class SavedRun:
    """Where :func:`save_run` put a run."""
    latest_csv: Path
    #: The history copy, or None when it could not be written (see
    #: ``history_error``); the latest CSV is written regardless.
    history_csv: Path | None
    templates_dir: str = ""
    history_error: str = ""


def _history(out_dir) -> Path:
    return Path(out_dir) / "history"


def _run_csv(out_dir, ts: str) -> Path:
    return _history(out_dir) / f"run_{ts}.csv"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest() if data else ""


def _json_default(value):
    """numpy scalars as their Python value (a numpy bool must stay a bool)."""
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def new_run_ts(out_dir) -> str:
    """A timestamp naming a new run, unique in *out_dir*.

    It names the run's history files and its waveform folders alike, so two
    runs started within the same second must not share it.
    """
    out_dir = Path(out_dir)
    base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts, n = base, 1
    while (any(_history(out_dir).glob(f"run_{ts}.*"))
           or any((out_dir / "analysis_data").glob(f"*/{ts}"))):
        n += 1
        ts = f"{base}_{n}"
    return ts


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


# ── A run in progress ─────────────────────────────────────────────────────────

class RunJournal:
    """Appends a running sweep's finished cases to its journal file.

    A journal that cannot be written (full disk, a file lock) costs only the
    ability to resume: the sweep goes on, with a warning in chipify.log.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._fh = None
        if path is None:
            return
        try:
            self._fh = open(path, "a", encoding="utf-8")
        except OSError as exc:
            log.warning("Cannot write the run journal %s: %s", path, exc)

    def append(self, rows: list) -> None:
        """Write one worker batch's rows and push them to disk.

        Synced each time: a host that loses power or goes to standby right
        after must not lose what the journal already acknowledged.
        """
        if not rows or self._fh is None:
            return
        try:
            self._fh.write(json.dumps({"t": time.time(), "rows": rows},
                                      default=_json_default) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except (OSError, TypeError, ValueError) as exc:
            log.warning("Run journal %s stopped: %s", self.path, exc)
            self.close()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def begin_run(stim, out_dir, ts: str, *, planned_runs: int, analysis_dirs: dict,
              run_settings: dict, engines: dict) -> RunJournal:
    """Write the record of a run that is about to start; return its journal.

    A record that cannot be written costs only the ability to resume: the
    sweep still runs (a disabled journal is returned, with a warning).
    """
    history = _history(out_dir)
    source = getattr(stim, "source_bytes", b"") or b""
    snapshot = ""
    try:
        history.mkdir(parents=True, exist_ok=True)
        if source:
            path = history / f"run_{ts}{SNAPSHOT_SUFFIX}"
            path.write_bytes(source)
            snapshot = str(path)
    except OSError as exc:
        log.warning("Could not start the record of run_%s, it cannot be resumed: %s",
                    ts, exc)
        return RunJournal(None)
    yaml_path = getattr(stim, "yaml_path", "") or ""
    run_meta.write_meta(
        _run_csv(out_dir, ts),
        yaml_name=Path(yaml_path).name if yaml_path else "",
        tran_dir=analysis_dirs.get("transient", ""),
        analysis_dirs=analysis_dirs,
        templates_dir=_archive_templates(stim, history / f"run_{ts}_templates"),
        datasheet_sha256=_sha256(source),
        datasheet_snapshot=snapshot,
        settings=run_settings,
        engines=engines,
        complete=False,
        planned_runs=planned_runs,
    )
    return RunJournal(history / f"run_{ts}{JOURNAL_SUFFIX}")


@dataclass
class UnfinishedRun:
    """A run whose record says it started but never finished."""
    out_dir: Path
    ts: str
    meta: dict

    @property
    def csv(self) -> Path:
        return _run_csv(self.out_dir, self.ts)

    @property
    def journal(self) -> Path:
        return _history(self.out_dir) / f"run_{self.ts}{JOURNAL_SUFFIX}"

    @property
    def planned_runs(self) -> int | None:
        return self.meta.get("planned_runs")

    @property
    def templates_dir(self) -> str:
        return str(self.meta.get("templates_dir", "") or "")

    def _batches(self) -> list[dict]:
        try:
            lines = self.journal.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        batches = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                batch = json.loads(line)
            except ValueError:
                # The last line of a journal whose process died mid-write.
                log.warning("Skipping unreadable line %d of %s", number, self.journal)
                continue
            if isinstance(batch, dict) and isinstance(batch.get("rows"), list):
                batches.append(batch)
        return batches

    def rows(self) -> dict[str, dict]:
        """The journal's rows by run id; a later row of a run id wins."""
        rows: dict[str, dict] = {}
        for batch in self._batches():
            for row in batch["rows"]:
                run_id = str(row.get("run_id", "")) if isinstance(row, dict) else ""
                if run_id:
                    rows[run_id] = row
        return rows

    def done_rows(self) -> dict[str, dict]:
        """Rows of the cases that ran: all but the ones lost with a worker."""
        return {run_id: row for run_id, row in self.rows().items()
                if "WORKER_LOST" not in str(row.get("sim_error", ""))}

    def analysis_dirs(self) -> dict[str, str]:
        return run_meta.analysis_dirs_for(self.csv, self.meta)

    def result_settings(self) -> dict:
        """The settings the run started with that shape its results."""
        return {key: value for key, value in (self.meta.get("settings") or {}).items()
                if key not in _SPEED_KEYS}

    def problem_with(self, stim) -> str:
        """Why *stim* cannot continue this run ("" when it can)."""
        recorded = self.meta.get("datasheet_sha256") or ""
        if recorded and recorded == _sha256(getattr(stim, "source_bytes", b"") or b""):
            return ""
        snapshot = self.meta.get("datasheet_snapshot") or ""
        where = f"; as it was: {self.out_dir / snapshot}" if snapshot else ""
        return f"the datasheet changed since run_{self.ts} started{where}"

    def elapsed_s(self) -> float:
        """How long the run's parts so far ran: each from its start to its last
        journal write, so the pause before a resume does not count."""
        starts = sorted(t for t in (_unix_time(s) for s in
                                    [self.meta.get("timestamp"), *self.meta.get("resumed", [])])
                        if t is not None)
        writes = [b["t"] for b in self._batches() if isinstance(b.get("t"), (int, float))]
        total = 0.0
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else math.inf
            in_part = [t for t in writes if start <= t < end]
            if in_part:
                total += max(in_part) - start
        return round(total, 2)


def _unix_time(iso: object) -> float | None:
    try:
        return datetime.datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return None


def find_unfinished(out_dir, yaml_name: str) -> UnfinishedRun | None:
    """The run of *yaml_name* to resume: its newest run, if that is unfinished.

    An older unfinished run is not offered once a newer run of the datasheet
    exists: whoever started that one gave the old one up.
    """
    history = _history(out_dir)
    if not history.is_dir():
        return None
    for meta_path in sorted(history.glob("run_*.meta.json"), reverse=True):
        ts = meta_path.name[len("run_"):-len(".meta.json")]
        meta = run_meta.read_meta(_run_csv(out_dir, ts))
        if meta.get("yaml") != yaml_name:
            continue
        if run_meta.is_complete(meta) or _run_csv(out_dir, ts).exists():
            return None
        return UnfinishedRun(Path(out_dir), ts, meta)
    return None


def continue_run(run: UnfinishedRun, *, run_settings: dict) -> RunJournal:
    """Mark *run* as resumed now and reopen its journal for appending."""
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    run.meta = run_meta.update_meta(
        run.csv, resumed=[*run.meta.get("resumed", []), stamp],
        settings=run_settings) or run.meta
    return RunJournal(run.journal)


# ── A batch in progress ───────────────────────────────────────────────────────

#: Folder in the output root that holds the ``--batch`` records.
BATCH_DIR = "batches"


class BatchRecord:
    """Where a ``--batch`` run got to: ``<out>/batches/batch_<ts>.json``.

    Lists the batch's datasheets in order, each ``pending``, ``running``,
    ``done`` (with its history run) or ``failed``, and is rewritten whole
    after every change. A batch that dies midway can then be continued with
    ``--batch DIR --resume``: finished datasheets are not simulated again, the
    interrupted one resumes its run, the rest start.
    """

    def __init__(self, path: Path | None, data: dict) -> None:
        self.path = path
        self.data = data

    @property
    def ts(self) -> str:
        return self.path.stem[len("batch_"):] if self.path is not None else ""

    def names(self) -> list[str]:
        return [str(item["yaml"]) for item in self.data["datasheets"]]

    def _item(self, yaml_name: str) -> dict:
        return next(item for item in self.data["datasheets"] if item["yaml"] == yaml_name)

    def state(self, yaml_name: str) -> str:
        return str(self._item(yaml_name).get("state", "pending"))

    def run(self, yaml_name: str) -> str:
        return str(self._item(yaml_name).get("run", "") or "")

    def set(self, yaml_name: str, state: str, run: str = "") -> None:
        item = self._item(yaml_name)
        item["state"] = state
        if run:
            item["run"] = run
        self._write()

    def finish(self) -> None:
        self.data["complete"] = True
        self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        try:
            # Replaced whole, never rewritten in place: a batch killed during
            # the write still leaves the previous, readable record.
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not update the batch record %s: %s", self.path, exc)


def begin_batch(out_dir, batch_dir, yaml_names: list[str],
                only: list[str] | None = None) -> BatchRecord:
    """Write the record of a ``--batch`` run about to start."""
    folder = Path(out_dir) / BATCH_DIR
    data = {
        "schema_version": 1,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(Path(batch_dir).resolve()),
        "only": list(only or []),
        "complete": False,
        "datasheets": [{"yaml": name, "state": "pending", "run": ""}
                       for name in yaml_names],
    }
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Could not start the batch record, it cannot be resumed: %s", exc)
        return BatchRecord(None, data)
    base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ts, n = base, 1
    while (folder / f"batch_{ts}.json").exists():
        n += 1
        ts = f"{base}_{n}"
    record = BatchRecord(folder / f"batch_{ts}.json", data)
    record._write()
    return record


def find_unfinished_batch(out_dir, batch_dir) -> BatchRecord | None:
    """The batch of *batch_dir* to resume: its newest batch, if unfinished."""
    folder = Path(out_dir) / BATCH_DIR
    if not folder.is_dir():
        return None
    target = str(Path(batch_dir).resolve())
    for path in sorted(folder.glob("batch_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("batch_dir") != target:
            continue
        return None if data.get("complete") else BatchRecord(path, data)
    return None


# ── A finished run ────────────────────────────────────────────────────────────

def save_run(df, stim, *, out_dir: str | os.PathLike[str], duration_s: float | None,
             yaml_name: str = "") -> SavedRun:
    """Write the results of a finished run into *out_dir* (see module docstring).

    Results from ``run_sim`` carry their run's timestamp (``df.attrs["run_ts"]``)
    and complete the record it started; others get a new one. The latest CSV
    and its sidecar are always written; a failing history copy is reported in
    ``SavedRun.history_error`` instead of raising, because the results
    themselves are safe by then.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / LATEST_CSV
    df.to_csv(latest, index=False)

    analysis_dirs = dict(df.attrs.get("analysis_dirs", {}) or {})
    write_analysis_pointers(analysis_dirs, out_dir)

    s = _dl.result_summary(_dl.prepare_results(df))
    ts = str(df.attrs.get("run_ts", "") or "")
    record = run_meta.read_meta(_run_csv(out_dir, ts)) if ts else {}
    source = getattr(stim, "source_bytes", b"") or b""
    meta = dict(yaml_name=yaml_name, duration_s=duration_s, total_runs=s.total,
                valid_runs=s.valid, global_yield=round(s.yield_pct, 2),
                tran_dir=analysis_dirs.get("transient", ""),
                analysis_dirs=analysis_dirs,
                datasheet_sha256=record.get("datasheet_sha256") or _sha256(source),
                settings=df.attrs.get("run_settings") or {},
                engines=df.attrs.get("engines") or {},
                planned_runs=record.get("planned_runs") or s.total,
                resumed=record.get("resumed") or [])

    history_csv: Path | None = None
    history_error = ""
    try:
        history = _history(out_dir)
        history.mkdir(parents=True, exist_ok=True)
        if record:
            snapshot = record.get("datasheet_snapshot") or ""
            meta["datasheet_snapshot"] = str(out_dir / snapshot) if snapshot else ""
            meta["templates_dir"] = record.get("templates_dir", "") or ""
        else:
            ts = new_run_ts(out_dir)
            if source:
                path = history / f"run_{ts}{SNAPSHOT_SUFFIX}"
                path.write_bytes(source)
                meta["datasheet_snapshot"] = str(path)
            meta["templates_dir"] = _archive_templates(
                stim, history / f"run_{ts}_templates")
        history_csv = _run_csv(out_dir, ts)
        df.to_csv(history_csv, index=False)
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
        try:
            (_history(out_dir) / f"run_{ts}{JOURNAL_SUFFIX}").unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not remove the finished run's journal: %s", exc)
    return SavedRun(latest, history_csv, meta.get("templates_dir", ""), history_error)


def _archive_templates(stim, dest_dir: Path) -> str:
    """Archive the run's netlist templates; "" when there are none or it fails."""
    from chipify.uikit.services import netlist_export
    try:
        return netlist_export.persist_templates(stim, dest_dir)
    except Exception as exc:  # noqa: BLE001 — a missing archive costs only exports
        log.warning("Could not persist netlist templates: %s", exc)
        return ""
