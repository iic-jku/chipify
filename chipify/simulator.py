# Copyright (c) 2026 Santiago Hofwimmer
# simulator.py
"""
simulator.py – Sweep orchestration for Chipify.

This module owns the *engine-agnostic* half of a simulation run: parameter
case generation, per-testbench template generation, worker-pool execution,
per-analysis persistence, and result-DataFrame assembly.

Everything simulator-specific lives in :mod:`chipify.engines` — one
:class:`~chipify.engines.base.BaseSimulator` subclass per supported
simulator, resolved by name through the engine registry. Adding support for
another simulator therefore never requires touching this module (see
PLUGINS.md, "Simulator engine plugin").

For backward compatibility the engine API (``BaseSimulator``,
``NgspiceSimulator``, ``VacaskSimulator``, ``run_xschem``, abort helpers, …)
is re-exported here under its historical names.
"""
import os
import sys
import itertools
import time
import logging
import datetime
import shutil
from pathlib import Path
from multiprocessing import active_children, current_process, get_context

import pandas as pd
from tqdm import tqdm
from chipify import settings
from chipify import util
from chipify import app_config
from chipify import data_loader as _dl
from chipify.templating import render_netlist
# Saving a finished run lives in run_store now; re-exported for old callers.
from chipify.run_store import write_analysis_pointers  # noqa: F401

# ── Engine API (re-exported for backward compatibility) ──────────────────────
from chipify.engines import (  # noqa: F401
    DEFAULT_ENGINE,
    UnknownEngineError,
    engine_names,
    engine_selector,
    get_engine,
    netlist_extension,
    register_engine,
    resolve_engine_name,
)
from chipify.engines.abort import (  # noqa: F401
    ABORT_FLAG_PATH,
    abort_simulation,
    clear_abort_flag,
    is_aborted as _is_aborted,
)
from chipify.engines.base import (  # noqa: F401
    BaseSimulator,
    describe_exit_code,
    extract_error_line,
    extract_error_line as _extract_ngspice_error,
)
from chipify.engines.ngspice import NgspiceSimulator, _inject_capture  # noqa: F401
from chipify.engines.rawfile import (  # noqa: F401
    parse_ascii_raw as _parse_ascii_raw,
    read_raw_file as _read_raw_file,
    sanitise_key as _sanitise_key,
)
from chipify.engines.staging import (  # noqa: F401
    stage_files_to_ram,
    staged_copy_is_stale as _staged_copy_is_stale,
)
from chipify.engines.vacask import (  # noqa: F401
    VacaskSimulator,
    _ng2vc_argv0,
    _ng2vc_search_paths,
    _resolve_ng2vc,
    _resolve_vacask_bin,
    _run_ng2vc,
    _vacask_extract_results,
    _vacask_write_analysis_tabs,
)
from chipify.engines.xschem import (  # noqa: F401
    XSCHEM_DEFAULT_TIMEOUT_SEC,
    run_xschem,
    safe_tb_path as _safe_tb_path,
)

log = logging.getLogger("chipify.simulator")

#: Built-in engine names (plugin engines come on top; see
#: ``chipify.engines.engine_names()`` for the full selectable list).
SUPPORTED_ENGINES = engine_names(include_plugins=False)


def get_simulator_engine(simulator_name: str) -> BaseSimulator:
    """Instantiate the engine registered under *simulator_name*.

    Unknown names fall back to ngspice **with a warning** — an unvalidated
    source (a typo'd ``simulator_engine`` in settings.json) must not crash a
    worker, but it must not silently run the wrong simulator either.
    """
    try:
        return get_engine(simulator_name)
    except UnknownEngineError:
        log.warning("Unknown simulator engine %r – falling back to %r. "
                    "Available: %s", simulator_name, DEFAULT_ENGINE,
                    ", ".join(engine_names()))
        return get_engine(DEFAULT_ENGINE)


# ── Per-run timeout ───────────────────────────────────────────────────────────

_DEFAULT_SIM_TIMEOUT_SEC = 10.0


def _resolve_sim_timeout(cfg: dict) -> float:
    """Per-simulation wall-clock limit from the ``sim_timeout_sec`` setting."""
    raw = cfg.get("sim_timeout_sec", _DEFAULT_SIM_TIMEOUT_SEC)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        log.warning("Invalid sim_timeout_sec=%r, using %ss.",
                    raw, _DEFAULT_SIM_TIMEOUT_SEC)
        return _DEFAULT_SIM_TIMEOUT_SEC
    return val if val > 0 else _DEFAULT_SIM_TIMEOUT_SEC


# ── Analysis persistence (worker side) ────────────────────────────────────────

def _persist_analyses(analyses, analysis_tab_paths, analysis_dirs,
                      run_id, tb_safe, tb_path, sample, engine=None) -> list[str]:
    """Write each declared analysis's tab file into its per-run CSV.

    If an analysis was set up to capture (tab path + output dir both present) but
    produced no CSV — the tab is missing, or present-but-empty/unreadable so
    ``persist_to_csv`` writes nothing — record a one-line reason (the engine's
    own ``capture_error``, else its log's error line) on *sample* under
    ``<tb>__<kind>_capture``. Worker logging does not reach chipify.log
    (forkserver workers lack the file handler), so the result row is the
    reliable channel; ``run_sim`` surfaces these from the main process. This is
    what makes the silent Bode/AC "no data" case diagnosable.

    Returns the failure notes, so the caller can fail the testbench's row.
    """
    eng_name = getattr(engine, "name", "") or "simulator"
    failures: list[str] = []
    for an in analyses:
        tab = analysis_tab_paths.get(an.kind, "")
        dest_dir = analysis_dirs.get(an.kind, "")
        if not (tab and dest_dir):
            continue

        reason = ""
        engine_reason = ""
        if not Path(tab).exists():
            if engine is not None:
                engine_reason = engine.capture_error(an.kind)
            reason = engine_reason or (
                f"{eng_name} wrote no {an.kind} tab to capture "
                f"(does the testbench run its .{an.kind} analysis?)"
            )
        else:
            dest_csv = Path(dest_dir) / f"run_{run_id}__{tb_safe}.csv"
            an.persist_to_csv(tab, dest_csv)
            if not dest_csv.exists():
                reason = (
                    f"{eng_name} wrote an empty/unreadable {an.kind} tab "
                    f"(capture vectors were empty)"
                )

        if reason:
            err = ""
            if engine is not None and not engine_reason:
                err = engine.extract_error(engine.run_log_tail())
            note = f"{an.kind} produced no data - {reason}"
            if err:
                note += f" | {eng_name}: {err}"
            sample[f"{tb_path}__{an.kind}_capture"] = note
            log.warning("%s: %s", tb_path, note)  # also surfaced from run_sim (main)
            failures.append(note)
    return failures


def _log_capture_failures(results: list) -> None:
    """Surface analysis-capture failures recorded on result rows (main process).

    Workers can't write to chipify.log under forkserver, so ``_persist_analyses``
    stashes the reason on each row under ``<tb>__<kind>_capture``. Here, in the
    main process, we log each distinct failure once so it actually reaches the
    log file (and it also persists as a column in simulation_results.csv).
    """
    counts: dict[tuple[str, str], int] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        for key, val in row.items():
            if (isinstance(key, str) and key.endswith("_capture")
                    and isinstance(val, str) and val):
                counts[(key, val)] = counts.get((key, val), 0) + 1
    for (key, val), count in counts.items():
        tb = key[: -len("_capture")].rstrip("_")
        log.warning("Analysis capture failed on %s (%d run(s)): %s", tb, count, val)


# ── Case evaluation (worker side) ─────────────────────────────────────────────

def _eval_scalar_measures(test, sample: dict, params: dict) -> None:
    """Evaluate datasheet ``measure:`` expressions against this run's scalars.

    Runs after MY_DATA parsing so a measure can reference the test's measured
    values (e.g. ``gbw: gain * bandwidth``) as well as numeric sweep
    parameters. This makes ``measure:`` work on the ngspice engine; on the
    VACASK raw path, names already computed from waveforms are left untouched
    (``name in sample`` check). A failed expression records NaN plus a
    one-line note under ``<tb>__measure_error`` — worker logging doesn't
    reach chipify.log (see _persist_analyses), so the row is the channel.
    """
    measures = getattr(test, "measure", {}) or {}
    if not measures:
        return
    from chipify.expression import default_evaluator

    namespace = {
        k: v for k, v in {**params, **sample}.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    for name, expr in measures.items():
        if name in sample:
            continue
        try:
            val = default_evaluator.evaluate_spice_measure(expr, namespace)
            sample[name] = float(val)
        except Exception as exc:
            sample[name] = float('nan')
            note = f"measure '{name}' = {expr!r} failed: {exc}"
            sample[f"{test.tb_path}__measure_error"] = " ".join(str(note).split())[:200]


#: How much of a run's log to search for the line explaining missing values: a
#: fatal model or library error sits near the top, before any analysis output.
_HINT_LOG_LINES = 100_000


def _log_hint(engine) -> str:
    """``" | <engine>: <first error line of its last run>"``, or '' if none.

    Values missing because the simulation itself failed (a shell without the
    PDK setup, a vector the testbench never defined) used to read as a bare
    ``INVALID_OUTPUT(... got 0)`` or ``NO_MY_DATA_FOUND``, with the reason
    left in a log that worker processes never forward.
    """
    try:
        line = extract_error_line(engine.run_log_tail(_HINT_LOG_LINES),
                                  fallback=False)
    except Exception:  # noqa: BLE001 — a diagnostic must never fail the row
        return ""
    return f" | {engine.name}: {line}" if line else ""


def _record_error(sample: dict, test, message: str) -> None:
    """Record *message* against *test* in *sample*, scoped to that testbench.

    Writes the per-testbench ``<tb_path>__error`` column so a failure stays
    attributable to the testbench that caused it — a single ``sim_error`` slot
    would hide the other testbenches' perfectly good measurements behind the
    row-level filter. ``sim_error`` is still maintained as the row-level
    roll-up, but *accumulated* rather than overwritten: with several failing
    testbenches per case, the old assignment kept only the last message.
    """
    col = _dl.tb_error_col(test.tb_path)
    previous = sample.get(col, _dl.NO_ERROR)
    sample[col] = message if previous == _dl.NO_ERROR else f"{previous} | {message}"

    row_previous = sample.get('sim_error', _dl.NO_ERROR)
    sample['sim_error'] = (
        message if row_previous == _dl.NO_ERROR else f"{row_previous} | {message}"
    )


def _fail_test(sample: dict, test, message: str) -> None:
    """Mark every measurement of *test* failed with *message* in *sample*."""
    _record_error(sample, test, message)
    sample[f"{test.tb_path}_overall_pass"] = False
    for val_obj in test.value_lst:
        sample[val_obj.name] = float('nan')
        sample[f"{val_obj.name}_pass"] = False


def _simulate_single_case(params, tests, engine_for,
                          run_id: str = "",
                          analysis_dirs: dict | None = None):
    """Simulate one parameter case across *tests*.

    *engine_for* is a ``test -> BaseSimulator`` selector, so each testbench can
    run on its own engine within the same case. Every per-testbench failure
    (template rendering, engine crash, bad output) fails only that testbench's
    measurements — one bad testbench must never lose the whole case (or, worse,
    the whole worker batch).
    """
    analysis_dirs = analysis_dirs or {}
    sample = params.copy()
    sample['sim_error'] = _dl.NO_ERROR
    sample['run_id'] = run_id
    # Seed every testbench's error column so it exists on *every* row, not only
    # the ones that failed — a column that appears halfway through the sweep
    # would read as NaN for the successful runs.
    for test in tests:
        sample[_dl.tb_error_col(test.tb_path)] = _dl.NO_ERROR

    timeout_sec = _resolve_sim_timeout(app_config.load_config())

    for test in tests:
        # A testbench whose netlist couldn't be generated (e.g. its engine is
        # unavailable) fails only its own measurements; the rest of the case
        # still runs on their own engines.
        tmpl_err = getattr(test, "template_error", None)
        if tmpl_err or not getattr(test, "template_str", ""):
            _fail_test(sample, test, tmpl_err or f"{test.tb_path}: no netlist template")
            continue
        try:
            engine = engine_for(test)
        except Exception as exc:  # noqa: BLE001
            _fail_test(sample, test, f"{test.tb_path}: engine unavailable: {exc}")
            continue

        analyses = getattr(test, "analyses", []) or []
        pid = os.getpid()
        tb_safe = test.tb_path.replace("/", "__").replace("\\", "__")

        # One worker-private tab file per declared analysis. The Jinja
        # variable name (tran_out_path / dc_out_path / ac_out_path) tells the
        # rendered ngspice template where to wrdata; VacaskSimulator also
        # uses this mapping to dump signals from the .raw file.
        render_kwargs = dict(params)
        analysis_tab_paths: dict[str, str] = {}
        for an in analyses:
            if not analysis_dirs.get(an.kind):
                continue
            # Kept as str: this path is baked into the rendered netlist (ngspice
            # wrdata target) and travels through the engine as a plain path.
            tab = str(
                Path(settings.FAST_TMP)
                / f"sim_{pid}_{run_id}_{tb_safe}_{an.kind}.tab"
            )
            render_kwargs[an.jinja_var()] = tab
            analysis_tab_paths[an.kind] = tab

        # render_netlist raises on any placeholder the case's parameters
        # don't cover (e.g. a param-name typo in the testbench). That must
        # fail this testbench's row — an uncaught exception here would lose
        # the entire worker batch with only a main-process log line.
        try:
            rendering = render_netlist(test.template_str, render_kwargs)
        except Exception as exc:  # noqa: BLE001
            _fail_test(sample, test,
                       f"{test.tb_path}: TEMPLATE_RENDER_ERROR: {exc}")
            continue

        # Built-in engines report failures via error_msg; a plugin engine may
        # raise instead — contain it the same way.
        try:
            sim_output, error_msg = engine.run(
                rendering, timeout_sec=timeout_sec,
                test=test, analysis_tab_paths=analysis_tab_paths,
            )
        except Exception as exc:  # noqa: BLE001
            _fail_test(sample, test,
                       f"{test.tb_path}: [{engine.name}] ENGINE_ERROR: {exc}")
            continue

        if error_msg:
            _fail_test(sample, test, f"{test.tb_path}: {error_msg}")
            continue

        # Persist every analysis's tab file into its destination CSV (and surface
        # any analysis that ran but produced no output — see _persist_analyses).
        capture_failures = _persist_analyses(
            analyses, analysis_tab_paths, analysis_dirs,
            run_id, tb_safe, test.tb_path, sample, engine=engine,
        )
        # A declared waveform that was not captured is a failed output of this
        # testbench, like a missing scalar. As a diagnostic note only, it left
        # sim_error 'None' and global_pass True: a sweep lost its transfer
        # curves in 44 of 45 runs without a single error in the results.
        for note in capture_failures:
            _record_error(sample, test, f"{test.tb_path}: CAPTURE_ERROR: {note}")
        capture_ok = not capture_failures

        # Transient-only testbench: no scalar measurements defined → skip MY_DATA.
        if not test.value_lst:
            sample[f"{test.tb_path}_overall_pass"] = capture_ok
            _eval_scalar_measures(test, sample, params)
            continue

        if sim_output and sim_output.startswith("MY_DATA:"):
            clean_line = sim_output[len("MY_DATA:"):].strip()
            values = [v for v in clean_line.split(' ') if v]

            all_passed = capture_ok
            for i, val_str in enumerate(values):
                if i >= len(test.value_lst):
                    break
                val_obj = test.value_lst[i]
                try:
                    val_float = float(val_str)
                    sample[val_obj.name] = val_float
                    sample[f"{val_obj.name}_pass"] = val_obj.isPass(val_float)
                    if not val_obj.isPass(val_float):
                        all_passed = False
                except ValueError:
                    # Unparsable token → that value is failed data, not a
                    # silently absent column.
                    _record_error(
                        sample, test,
                        f"{test.tb_path}: INVALID_OUTPUT({val_str})",
                    )
                    sample[val_obj.name] = float('nan')
                    sample[f"{val_obj.name}_pass"] = False
                    all_passed = False

            # Fewer values than declared measurements is an error, not a
            # silent gap — mark the missing ones failed so the row can't
            # masquerade as a clean run with absent columns.
            n_expected = len(test.value_lst)
            if len(values) < n_expected:
                _record_error(
                    sample, test,
                    f"{test.tb_path}: INVALID_OUTPUT("
                    f"expected {n_expected} values, got {len(values)})"
                    f"{_log_hint(engine)}",
                )
                all_passed = False
                for val_obj in test.value_lst[len(values):]:
                    sample[val_obj.name] = float('nan')
                    sample[f"{val_obj.name}_pass"] = False

            sample[f"{test.tb_path}_overall_pass"] = all_passed
            _eval_scalar_measures(test, sample, params)
        else:
            _record_error(sample, test,
                          f"{test.tb_path}: NO_MY_DATA_FOUND{_log_hint(engine)}")
            sample[f"{test.tb_path}_overall_pass"] = False

    return sample


def _lost_batch_rows(batch, tests, exc) -> list:
    """Synthesize one error row per case of a worker batch that never returned.

    A crashed worker (or a lost pool process) used to take its whole batch with
    it: the rows simply never reached the results list, so the CSV came back
    shorter while progress still ran to 100 % — a silently truncated sweep that
    looked like a complete one. Recording the cases as explicit errors keeps the
    row count equal to the number of parameter cases and makes the loss visible.
    """
    message = f"WORKER_LOST: {exc}"
    rows: list = []
    for params, run_id in batch:
        sample = dict(params)
        sample['run_id'] = run_id
        sample['sim_error'] = _dl.NO_ERROR
        # _fail_test writes each testbench's error column, so every column the
        # successful rows carry is present here too.
        for test in tests:
            _fail_test(sample, test, f"{test.tb_path}: {message}")
        if not tests:  # nothing to attribute it to — keep the row-level slot
            sample['sim_error'] = message
        rows.append(sample)
    return rows


def _simulate_single_case_with_engine(params, tests, engine: BaseSimulator,
                                      run_id: str = "",
                                      analysis_dirs: dict | None = None):
    """Back-compat shim: run all *tests* with one explicit *engine*."""
    return _simulate_single_case(
        params, tests, lambda _t: engine, run_id, analysis_dirs,
    )


class _LostBatchDetector:
    """Find submitted batches whose worker process died before returning them.

    ``multiprocessing.Pool`` replaces a worker that dies abruptly (OOM kill,
    segfault, ``os._exit``) but never resolves the task it was running: that
    AsyncResult stays unready forever, ``get()`` never raises, and the sweep
    waits for a case nobody is simulating. So every worker writes its process
    name into a claim file when it takes a batch (``_claim_batch``), and when
    the set of pool processes changes, unready batches whose claimant is gone
    are reported lost. Names, not pids: Windows reuses pids quickly.

    A worker that dies between taking a batch and writing its claim (while
    unpickling the task) is not detected.
    """

    #: How long a batch whose worker is gone may still deliver its result. A
    #: worker that sent its result and then died is not a lost batch.
    GRACE_SEC = 2.0

    def __init__(self, claim_dir: Path) -> None:
        self.claim_dir = claim_dir
        claim_dir.mkdir(parents=True, exist_ok=True)
        self._live = self._live_workers()
        self._seen = dict(self._live)
        self._suspects: dict[int, tuple[float, str]] = {}

    @staticmethod
    def _live_workers() -> dict:
        return {p.name: p for p in active_children()}

    def claim_path(self, index: int) -> str:
        return str(self.claim_dir / f"batch_{index}")

    def _claimant(self, index: int) -> str:
        try:
            return Path(self.claim_path(index)).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _describe(self, name: str) -> str:
        proc = self._seen.get(name)
        if proc is None:
            return f"worker process {name} exited"
        return (f"worker process {name} (pid {proc.pid}) "
                f"{describe_exit_code(proc.exitcode)}")

    def poll(self, pending: list[int]) -> dict[int, str]:
        """Map each pending batch index whose worker is gone to the reason."""
        live = self._live_workers()
        now = time.monotonic()
        # Claims are only read when a pool process came or went: a replacement
        # appearing means a death even if the dead worker was never seen alive.
        if live.keys() != self._live.keys():
            self._seen.update(live)
            for index in pending:
                name = "" if index in self._suspects else self._claimant(index)
                if name and name not in live:
                    self._suspects[index] = (now, self._describe(name))
            self._live = live
        lost: dict[int, str] = {}
        for index in pending:
            since, reason = self._suspects.get(index, (None, ""))
            if since is not None and now - since >= self.GRACE_SEC:
                lost[index] = reason
        return lost

    def close(self) -> None:
        shutil.rmtree(self.claim_dir, ignore_errors=True)


def _claim_batch(claim_path: str) -> None:
    """Record which pool worker took a batch (see _LostBatchDetector)."""
    if not claim_path:
        return
    try:
        Path(claim_path).write_text(current_process().name, encoding="utf-8")
    except OSError:
        pass  # only loss detection is off for this batch; the batch still runs


def simulate_case_batch(batch_args):
    """Worker helper: process a batch of cases to reduce IPC overhead.

    Each ``Test`` carries its resolved ``engine`` name (set by run_sim); engines
    are instantiated once per worker and reused across the batch, so a datasheet
    can mix engines across testbenches.

    The run's settings travel with every batch and are pinned while it runs
    (``app_config.set_run_overrides``), so each config read in this worker —
    the timeout, the VACASK engine's binary and paths — returns what the run
    started with, CLI overrides included.
    """
    param_id_batch, tests, analysis_dirs, claim_path, run_cfg = batch_args
    _claim_batch(claim_path)
    previous = app_config.set_run_overrides(run_cfg)
    try:
        engine_for = engine_selector()
        results = []
        for params, run_id in param_id_batch:
            # Aborted, or the chipify process that queued this batch has died:
            # start no further simulator (see abort.parent_gone).
            if _is_aborted():
                break
            results.append(
                _simulate_single_case(params, tests, engine_for, run_id, analysis_dirs))
        return results
    finally:
        app_config.set_run_overrides(previous)


def _chunk_args(worker_args, chunk_size):
    for i in range(0, len(worker_args), chunk_size):
        yield worker_args[i:i + chunk_size]


def _resolve_chunk_size(cfg, total_tasks, num_cores):
    configured = str(cfg.get("chunk_size", "auto"))
    if configured == "auto":
        return max(1, min(16, total_tasks // (num_cores * 8) if num_cores > 0 else 1))
    try:
        parsed = int(configured)
        return max(1, parsed)
    except (TypeError, ValueError):
        log.warning("Unknown chunk_size=%r, falling back to auto.", configured)
        return max(1, min(16, total_tasks // (num_cores * 8) if num_cores > 0 else 1))


# ── Template generation (main process) ────────────────────────────────────────

def generate_templates(stim, templates_dir: str = "") -> None:
    """Populate ``test.template_str`` for every test in *stim*.

    Each testbench is rendered with **its own** engine (``test.engine``, resolved
    by run_sim), so a datasheet can mix engines. If a testbench's netlist can't
    be produced (e.g. its engine is unavailable), the failure is recorded on
    ``test.template_error`` and that testbench's runs fail individually — the rest
    of the sweep continues.

    If *templates_dir* is set, read pre-rendered xschem outputs from
    ``<templates_dir>/<safe_tb_path><engine.netlist_ext>`` instead of invoking
    xschem locally — useful for re-running a sweep against netlists that were
    already generated.
    """
    engine_for = engine_selector()

    for test in stim.tests:
        if _is_aborted():
            raise InterruptedError("Aborted before netlist generation.")
        test.template_error = None
        try:
            engine = engine_for(test)
        except Exception as exc:  # noqa: BLE001
            test.template_str = ""
            test.template_error = f"{test.tb_path}: engine unavailable: {exc}"
            log.warning("Engine resolution failed for %s: %s", test.tb_path, exc)
            continue
        ext = engine.netlist_ext
        try:
            if templates_dir:
                safe = test.tb_path.replace("/", "__").replace("\\", "__")
                fp = Path(templates_dir) / (safe + ext)
                test.template_str = fp.read_text(encoding="utf-8")
            else:
                test.template_str = engine.generate_test_template(test)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001
            test.template_str = ""
            test.template_error = (
                f"{test.tb_path}: [{engine.name}] netlist generation failed: {exc}"
            )
            log.warning("Template generation failed for %s [%s]: %s",
                        test.tb_path, engine.name, exc)


def generate_cases(stim) -> list:
    param_names = stim.params.keys()
    param_values = stim.params.values()
    return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]


# ── Result assembly ───────────────────────────────────────────────────────────

def _assemble_result_df(rows: list, analysis_dirs: dict) -> pd.DataFrame:
    """Build a normalized results DataFrame (matches GUI / CSV load semantics).

    The per-analysis directories are stored under ``df.attrs["analysis_dirs"]``
    so the GUI can locate transient / DC / AC CSVs by kind. ``df.attrs["tran_dir"]``
    is also set for backward compatibility with consumers that only know about
    transient.
    """
    df = _dl.prepare_results(pd.DataFrame(rows))
    if analysis_dirs:
        df.attrs["analysis_dirs"] = dict(analysis_dirs)
        if analysis_dirs.get("transient"):
            df.attrs["tran_dir"] = analysis_dirs["transient"]
    return df


def _create_analysis_dirs(stim, out_dir) -> dict[str, str]:
    """Create ``<out_dir>/analysis_data/<kind>/<timestamp>/`` per analysis kind.

    Only kinds actually used by at least one test get a directory. Run ids
    restart at ``000000`` for every run, so the directory is what keeps two
    runs' ``run_<id>__<tb>.csv`` files apart — which is why it lives under the
    run's own output root rather than one folder shared by every datasheet.
    """
    sim_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dirs: dict[str, str] = {}
    for kind in ("transient", "dc", "ac"):
        if any(
            any(a.kind == kind for a in getattr(t, "analyses", []) or [])
            for t in stim.tests
        ):
            d = Path(out_dir) / "analysis_data" / kind / sim_timestamp
            d.mkdir(parents=True, exist_ok=True)
            # Stored as str: rides df.attrs and is JSON-serialised into the
            # run's .meta.json sidecar.
            analysis_dirs[kind] = str(d)
            log.info("%s store: %s", kind, d)
    return analysis_dirs


def _check_executable_plan(stim) -> None:
    """Refuse a plan that would simulate without checking anything.

    With no testbench, every parameter case came back with no error and no
    verdict column, which ``global_pass`` reads as passed: 100 % yield for a
    run that measured nothing. A testbench that declares no measurement, no
    ``measure:`` and no signals passes vacuously the same way. The datasheet
    schema allows both, since a datasheet is built up step by step in the GUI.
    """
    if not stim.tests:
        raise ValueError(
            "The datasheet declares no testbenches under 'tests:', "
            "so there is nothing to simulate."
        )
    idle = [t.tb_path for t in stim.tests
            if not (t.value_lst or getattr(t, "measure", None)
                    or getattr(t, "analyses", None))]
    if idle:
        raise ValueError(
            f"Testbench(es) {', '.join(idle)} declare no measurements or "
            f"signals, so their runs could only pass without checking anything."
        )


def _check_any_netlist(stim) -> None:
    """Refuse to dispatch when not a single testbench could be netlisted.

    A testbench whose netlist failed fails only its own rows, so the others
    still run. With none left, every case is a foregone failure: a datasheet
    whose testbenches had all been moved ran 9000 cases into 9000 crashes.
    """
    errors = [t.template_error or f"{t.tb_path}: no netlist template"
              for t in stim.tests
              if getattr(t, "template_error", None) or not getattr(t, "template_str", "")]
    if stim.tests and len(errors) == len(stim.tests):
        raise RuntimeError(
            "No testbench could be netlisted, so every case would fail: "
            + "; ".join(errors)
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_sim(stim, progress_callback=None, simulator=None, chunk_callback=None,
            templates_dir: str = "", out_dir=None, overrides=None):
    """
    Main simulation entry point.

    Uses multiprocessing with a non-fork context to avoid deadlocks in GUI apps:
    - Linux:  'forkserver' (stable + faster startup than spawn)
    - others: 'spawn'

    Parameters
    ----------
    chunk_callback:
        If ``None`` (default CLI path), no per-batch DataFrame assembly runs.
        If set, called with incremental row batches according to
        ``live_plot_emit_stride`` in settings — omit passing this when live
        plotting is disabled so the sweep stays at baseline CPU cost.
    templates_dir:
        When set, skip xschem and load pre-rendered Jinja2 templates from
        this directory (see ``--templates-dir``) instead of regenerating them.
    out_dir:
        Output root that receives ``analysis_data/`` (default
        ``settings.OUT_DIR``). Batch mode passes each datasheet's own folder.
    overrides:
        Config values for this run only, over settings.json — the CLI's
        ``--cores`` and ``--timeout`` as ``num_cores`` / ``sim_timeout_sec``.
        ``None`` values are ignored.

    Returns
    -------
    pandas.DataFrame | None
        The results frame, or ``None`` **only** when the run was aborted (abort
        flag or a cancelling ``progress_callback``). Every other failure raises:
        callers must report it rather than mistake it for a cancellation.
    """
    pool = None
    detector = None
    log.info("run_sim() started. Testbenches: %d", len(stim.tests))

    try:
        clear_abort_flag()
        _check_executable_plan(stim)

        # ── Init phase ────────────────────────────────────────────────────────
        log.info("Phase 1/3: generating parameter cases...")
        param_sets = generate_cases(stim)
        log.info("Total cases: %d", len(param_sets))
        cfg = app_config.load_config()
        cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
        # Resolve each testbench's engine: its own ``engine:`` wins, else the CLI
        # --simulator override, else the simulator_engine setting, else ngspice.
        # Write the concrete name back onto the (picklable) Test so workers and
        # template generation pick the right engine per testbench.
        for test in stim.tests:
            test.engine = resolve_engine_name(test, simulator, cfg)
        engines_in_use = sorted({t.engine for t in stim.tests})
        engine_instances = [get_simulator_engine(n) for n in engines_in_use]
        log.info("Simulator engine(s) in use: %s", ", ".join(engines_in_use))

        if _is_aborted():
            raise InterruptedError("Aborted before template generation.")
        log.info("Phase 2/3: generating Xschem templates...")
        generate_templates(stim, templates_dir=templates_dir)
        _check_any_netlist(stim)

        if _is_aborted():
            raise InterruptedError("Aborted before RAM staging.")
        log.info("Phase 3/3: staging files to RAM disk...")
        stage_files_to_ram(engine_instances)

        if _is_aborted():
            raise InterruptedError("Aborted before pool start.")

        # Assign a stable zero-padded run_id to every parameter case.
        run_ids = [f"{i:06d}" for i in range(len(param_sets))]

        analysis_dirs = _create_analysis_dirs(stim, out_dir or settings.OUT_DIR)

        results = []
        num_cores = cfg.get("num_cores") or util.get_num_cores()
        log.info("Spawning pool: %d workers, %d tasks.", num_cores, len(param_sets))

        total_tasks = len(param_sets)
        completed = 0

        # ── Pool execution ────────────────────────────────────────────────────
        # Avoid plain 'fork' in GUI/threaded parents; use forkserver/spawn.
        configured_method = cfg.get("process_start_method", "auto")
        if configured_method == "auto":
            start_method = "forkserver" if sys.platform.startswith("linux") else "spawn"
        elif configured_method in {"forkserver", "spawn"}:
            start_method = configured_method
        else:
            log.warning("Unknown process_start_method=%r, falling back to auto.", configured_method)
            start_method = "forkserver" if sys.platform.startswith("linux") else "spawn"
        ctx = get_context(start_method)
        pool = ctx.Pool(processes=num_cores)
        log.debug("Pool created (%s, %d workers).", start_method, num_cores)
        detector = _LostBatchDetector(
            Path(settings.FAST_TMP) / f"claims_{os.getpid()}_{time.time_ns()}"
        )

        # Batch tasks to reduce scheduler/IPC overhead while keeping polling.
        # Each batch item is a (params, run_id) pair so workers can persist
        # per-analysis CSVs with the correct run_id in the filename.
        chunk_size = _resolve_chunk_size(cfg, total_tasks, num_cores)
        # The run's settings, fixed for its whole duration: every worker batch
        # pins them (simulate_case_batch).
        run_cfg = {key: cfg[key] for key in app_config.RUN_KEYS if key in cfg}
        run_cfg.update(num_cores=num_cores, process_start_method=start_method,
                       chunk_size=chunk_size,
                       sim_timeout_sec=_resolve_sim_timeout(cfg))
        param_id_pairs = list(zip(param_sets, run_ids))
        param_id_batches = list(_chunk_args(param_id_pairs, chunk_size))
        # Each entry keeps its own batch alongside the AsyncResult so a worker
        # that dies can still be accounted for case by case (_lost_batch_rows).
        pending = [
            (
                pool.apply_async(
                    simulate_case_batch,
                    ((batch, stim.tests, analysis_dirs,
                      detector.claim_path(index), run_cfg),),
                ),
                batch,
                index,
            )
            for index, batch in enumerate(param_id_batches)
        ]
        tasks_lost = False
        log.debug("%d batch tasks submitted (chunk_size=%d).", len(pending), chunk_size)

        chunk_emit_stride = max(1, app_config.get_live_plot_emit_stride()) if chunk_callback else 1
        chunk_batch_counter = 0
        chunk_row_buffer: list = []

        with tqdm(total=total_tasks) as pbar:
            while pending:
                if _is_aborted():
                    log.info("Abort flag detected – terminating pool.")
                    raise InterruptedError("Abort flag detected during run.")

                lost = detector.poll([index for _ar, _batch, index in pending])
                still_pending = []
                for ar, batch, index in pending:
                    if ar.ready():
                        try:
                            batch_results = ar.get(timeout=0)
                        except InterruptedError:
                            raise
                        except Exception as exc:  # incl. WorkerLostError
                            log.error(
                                "Worker error (%d case(s) recorded as errors): %s",
                                len(batch), exc,
                            )
                            batch_results = _lost_batch_rows(batch, stim.tests, exc)
                    elif index in lost:
                        # Its worker died with it, so this result never arrives
                        # (see _LostBatchDetector).
                        log.error(
                            "Worker lost (%d case(s) recorded as errors): %s",
                            len(batch), lost[index],
                        )
                        batch_results = _lost_batch_rows(batch, stim.tests,
                                                         lost[index])
                        tasks_lost = True
                    else:
                        still_pending.append((ar, batch, index))
                        continue

                    results.extend(batch_results)
                    completed += len(batch_results)
                    if chunk_callback and batch_results:
                        chunk_row_buffer.extend(batch_results)
                        chunk_batch_counter += 1
                        sweep_complete = len(results) >= total_tasks
                        emit_now = sweep_complete or (
                            chunk_batch_counter % chunk_emit_stride == 0
                        )
                        if emit_now and chunk_row_buffer:
                            try:
                                chunk_df = _assemble_result_df(
                                    chunk_row_buffer, analysis_dirs
                                )
                                chunk_callback(chunk_df)
                            except Exception:
                                log.debug("chunk_callback failed.", exc_info=True)
                            chunk_row_buffer = []

                    completed = min(total_tasks, completed)
                    pbar.update(min(total_tasks - pbar.n, max(0, completed - pbar.n)))
                    if progress_callback:
                        progress_callback(completed, total_tasks)

                pending = still_pending
                if pending:
                    time.sleep(0.05)

        # Flush any live rows left in stride buffer (should usually be empty).
        if chunk_callback and chunk_row_buffer:
            try:
                chunk_df = _assemble_result_df(chunk_row_buffer, analysis_dirs)
                chunk_callback(chunk_df)
            except Exception:
                log.debug("chunk_callback failed on final flush.", exc_info=True)
            chunk_row_buffer = []

        if tasks_lost:
            # A lost task stays in the pool's cache, and a closed pool waits
            # for that cache to drain: join() would never return. Every worker
            # left is idle by now.
            pool.terminate()
        else:
            pool.close()
        pool.join()
        log.info("Pool finished cleanly. %d results collected.", len(results))

        # Surface analysis-capture failures recorded by workers (worker logging
        # doesn't reach chipify.log under forkserver, so the reason rides the row).
        _log_capture_failures(results)

        result_df = _assemble_result_df(results, analysis_dirs)
        return result_df

    except InterruptedError:
        log.info("Simulation interrupted by user.")
        if pool is not None:
            pool.terminate()
            pool.join()
            log.info("Pool terminated.")
        return None

    except Exception as exc:
        # Re-raise rather than returning None. Returning None made an
        # unexpected failure indistinguishable from a user abort, and the GUI
        # worker treats None as "aborted" — so the run died in total silence,
        # with the reason reaching nothing but chipify.log. ``None`` now means
        # "aborted" and nothing else; callers report the exception.
        log.exception("Unexpected error during simulation: %s", exc)
        if pool is not None:
            pool.terminate()
            pool.join()
            log.info("Pool terminated after error.")
        raise

    finally:
        if detector is not None:
            detector.close()
        clear_abort_flag()
        log.info("run_sim() exited.")
