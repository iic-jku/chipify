# Copyright (c) 2026 Santiago Hofwimmer
import argparse
import fnmatch
import math
import os
import sys
import json
from pathlib import Path

from chipify import util
from chipify import settings
from chipify import simulator
from chipify.analyzer import print_summary


def _json_summary(df, yaml_name: str, duration_s: float) -> dict:
    """Return a machine-readable summary dict for the completed run."""
    from chipify import data_loader as _dl
    s = _dl.result_summary(_dl.prepare_results(df))
    return {
        "yaml":       yaml_name,
        "total":      s.total,
        "crashes":    s.crashes,
        "valid":      s.valid,
        "passed":     s.passed,
        "yield":      round(s.yield_pct, 2),
        "duration_s": round(duration_s, 2),
    }


def _with_scalar_equations(df, stim):
    """Add the datasheet's scalar-equation columns, as the GUI does on load.

    Without them a report plot of a derived quantity was skipped as a missing
    measurement here while the GUI drew it. Applied after the result CSVs are
    written, so saved results stay as simulated, like the GUI's.
    """
    from chipify.uikit.services import equation_service as _eq
    df, _derived, lines = _eq.apply_scalar_equations(df, _eq.scalar_equations(stim))
    for line in lines:
        if line.startswith("[✗]"):
            # ASCII only: a Windows console cannot print the service's arrows.
            print(f"[!] Equation skipped: {line[4:].replace('→', '->')}")
    return df


def _generate_reports(df, stim, yaml_path, out_dir, duration_s):
    """Write the datasheet's configured plots/reports; return the ReportResult."""
    from chipify import report_service

    result = report_service.generate_reports(
        df, stim, str(yaml_path), out_dir, duration_s=duration_s)
    for path in result.files:
        print(f"[+] {path}")
    for warning in result.warnings:
        print(f"[!] {warning}")
    if result.files:
        print(f"[+] {len(result.files)} report file(s) in {result.out_dir}")
    return result


def _write_markdown(entry, md_path) -> None:
    """Write *entry*'s Markdown report; a failure is reported, not raised."""
    try:
        from chipify import md_export
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_export.generate_md_report(entry.df, entry.stim, entry.yaml_path, md_path)
        print(f"[+] Markdown report saved to {md_path}")
    except Exception as exc:
        print(f"[!] Markdown report failed for {entry.label}: {exc}")


def _make_progress_stream_cb():
    """Return a progress_callback that emits 'PROGRESS: <done> <total>' to stdout.

    Enabled by --progress-stream so a parent process can tail stdout for these
    lines and drive its own progress bar.
    """
    def _cb(current: int, total: int) -> None:
        print(f"PROGRESS: {current} {total}", flush=True)
    return _cb


def _run_single(yaml_path: str | os.PathLike[str], *, json_out: bool = False,
                simulator_override: str | None = None,
                templates_dir: str | None = None,
                progress_stream: bool = False,
                out_dir: str | os.PathLike[str] | None = None,
                reports: bool = False,
                overrides: dict | None = None,
                resume: bool = False):
    """Run simulation for one yaml file.

    Returns a ``report_service.BatchEntry`` (results, datasheet, summary dict,
    reports), or None on failure.

    out_dir:
        Where to write simulation_results.csv, history/ and analysis_data/.
        Defaults to the global OUT_DIR; batch mode passes a per-datasheet
        subdirectory so consecutive runs don't overwrite or read each other's
        results and waveforms.
    overrides:
        Settings for this run over settings.json (``--cores``, ``--timeout``).
    resume:
        Continue the datasheet's interrupted run (``--resume``).
    """
    import time
    from chipify import report_service, run_store

    yaml_path = Path(yaml_path)
    print(f"[*] Loading configuration: {yaml_path.name}")
    out_dir = Path(out_dir) if out_dir else Path(settings.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stim = util.Stimuli(yaml_path)
    unfinished = None
    earlier_s = 0.0
    if resume:
        unfinished = run_store.find_unfinished(out_dir, yaml_path.name)
        if unfinished is None:
            print(f"[-] Nothing to resume for {yaml_path.name}: its newest run in "
                  f"{out_dir / 'history'} finished, or it has none.")
            return None
        problem = unfinished.problem_with(stim)
        if problem:
            print(f"[-] Cannot resume {yaml_path.name}: {problem}.")
            return None
        print(f"[*] Resuming run_{unfinished.ts}: {len(unfinished.done_rows())} of "
              f"{unfinished.planned_runs} case(s) done")
        earlier_s = unfinished.elapsed_s()
    t0 = time.perf_counter()
    progress_cb = _make_progress_stream_cb() if progress_stream else None
    try:
        df = simulator.run_sim(
            stim,
            simulator=simulator_override,
            templates_dir=templates_dir or "",
            progress_callback=progress_cb,
            out_dir=out_dir,
            overrides=overrides,
            resume=unfinished,
        )
    except Exception as exc:
        # run_sim raises on unexpected failures and returns None only on abort,
        # so print the actual reason instead of a generic "no data".
        from chipify import app_config
        print(f"[-] Simulation failed for {yaml_path}: {exc}")
        print(f"    See {app_config.LOG_PATH} for the traceback.")
        _report_unfinished(out_dir, yaml_path)
        return None
    duration_s = time.perf_counter() - t0 + earlier_s

    if df is None:
        print(f"[-] Simulation aborted for {yaml_path}")
        _report_unfinished(out_dir, yaml_path)
        return None

    from chipify import run_store
    saved = run_store.save_run(df, stim, out_dir=out_dir, duration_s=duration_s,
                               yaml_name=yaml_path.name)
    print(f"[+] Results saved to {saved.latest_csv}")
    if saved.history_error:
        print(f"[!] Could not save history: {saved.history_error}")

    df = _with_scalar_equations(df, stim)
    print_summary(df, stim)

    summary = _json_summary(df, yaml_path.name, duration_s)
    summary["run"] = saved.history_csv.name if saved.history_csv else ""
    entry = report_service.BatchEntry(yaml_path, out_dir, df=df, stim=stim,
                                      duration_s=duration_s, summary=summary)
    if reports:
        entry.reports = _generate_reports(df, stim, yaml_path, out_dir, duration_s)
        summary["reports_dir"] = str(entry.reports.out_dir)
    if json_out:
        print(json.dumps(summary))
    return entry


def _report_unfinished(out_dir, yaml_path) -> None:
    """After a run ended early: what it kept, and how to continue it."""
    from chipify import run_store

    run = run_store.find_unfinished(out_dir, Path(yaml_path).name)
    if run is None:
        return
    print(f"    {len(run.done_rows())} of {run.planned_runs} case(s) finished and are "
          f"kept in {run.journal}; add --resume to the same command to continue.")


def _load_finished_run(yaml_path: str | os.PathLike[str], *,
                       out_dir: str | os.PathLike[str], json_out: bool = False,
                       run: str = "", reports: bool = True):
    """Load one datasheet's newest finished run and write its reports.

    The ``--report-only`` counterpart of :func:`_run_single`: nothing is
    simulated. Results, duration and waveform directories come from the newest
    ``<out_dir>/history/run_*.csv`` recorded for this datasheet, or from *run*
    (a history CSV name) when given. Returns the ``report_service.BatchEntry``,
    or None when there is no such run.
    """
    from chipify import data_loader as _dl
    from chipify import report_service, run_meta

    yaml_path, out_dir = Path(yaml_path), Path(out_dir)
    print(f"[*] Loading finished run: {yaml_path.name}")
    runs = [label for label in _dl.list_history_runs(out_dir, yaml_name=yaml_path.name)
            if label.startswith("run_")]
    if run:
        runs = [label for label in runs if label == run]
    if not runs:
        print(f"[-] No finished run of {yaml_path.name} in {out_dir / 'history'}"
              + (f" named {run}" if run else ""))
        return None
    csv_path = out_dir / "history" / runs[0]
    stim = util.Stimuli(yaml_path)
    # load_csv also restores the run's waveform dirs. The report prints the
    # datasheet's current limits, so the verdicts are graded against them.
    df = _with_scalar_equations(_dl.regrade(_dl.load_csv(str(csv_path)), stim), stim)
    duration_s = run_meta.read_meta(csv_path).get("duration_s")
    print(f"[+] Using {csv_path}")

    summary = _json_summary(df, yaml_path.name, duration_s or 0.0)
    summary["run"] = csv_path.name
    entry = report_service.BatchEntry(yaml_path, out_dir, df=df, stim=stim,
                                      duration_s=duration_s, summary=summary)
    if reports:
        entry.reports = _generate_reports(df, stim, yaml_path, out_dir, duration_s)
        summary["reports_dir"] = str(entry.reports.out_dir)
    if json_out:
        print(json.dumps(summary))
    return entry


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a number greater than 0, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _batch_files(batch_dir: Path, only: list[str] | None) -> tuple[list[Path], int]:
    """The datasheets a ``--batch`` covers, in order, and how many DIR has.

    Exits with a message when DIR has none or ``--only`` matches none.
    """
    from chipify import project_config

    found = project_config.order_datasheets(list(batch_dir.glob("*.yaml")))
    if not found:
        print(f"[-] No *.yaml files found in: {batch_dir}")
        sys.exit(1)
    selected = _select_datasheets(found, only)
    if not selected:
        print(f"[-] --only {' '.join(only or [])} matched no datasheet in "
              f"{batch_dir} (there are: {', '.join(p.stem for p in found)})")
        sys.exit(1)
    return selected, len(found)


def _preflight_one(yaml_path: Path, *, simulator_override: str | None,
                   templates_dir: str | None, json_out: bool) -> bool:
    """``--preflight`` for one datasheet; True when it is ready to run."""
    print(f"[*] Preflight: {yaml_path.name}")
    summary: dict = {"yaml": yaml_path.name}
    try:
        stim = util.Stimuli(yaml_path)
        report = simulator.preflight(stim, simulator=simulator_override,
                                     templates_dir=templates_dir or "")
    except Exception as exc:  # noqa: BLE001 — a datasheet that does not load is a finding
        report = None
        summary["error"] = " ".join(str(exc).split())[:300] or type(exc).__name__
        print(f"    [-] {summary['error']}")
    if report is not None:
        summary.update(cases=report.cases, testbenches=report.testbenches)
        if report.plan_error:
            summary["error"] = report.plan_error
            print(f"    [-] {report.plan_error}")
        else:
            sizes = " x ".join(f"{n} {name}" for name, n in report.sizes.items())
            print(f"    {report.cases} case(s)" + (f" ({sizes})" if sizes else ""))
            for tb_path, info in report.testbenches.items():
                mark, what = ("[-]", info["error"]) if info["error"] else ("[+]", "ok")
                print(f"    {mark} {tb_path} [{info['engine']}]: {what}")
    summary["ok"] = report is not None and report.ok
    if json_out:
        print(json.dumps(summary))
    return summary["ok"]


def _run_case(yaml_path: Path, index: int, *, out_dir: Path,
              simulator_override: str | None, templates_dir: str | None,
              overrides: dict, json_out: bool) -> bool:
    """``--case`` for one datasheet; True when the case ran without an error."""
    from chipify import data_loader as _dl

    keep = out_dir / "debug" / yaml_path.stem / f"case_{index:06d}"
    try:
        stim = util.Stimuli(yaml_path)
        row = simulator.run_case(stim, index, keep, simulator=simulator_override,
                                 templates_dir=templates_dir or "", overrides=overrides)
    except Exception as exc:  # noqa: BLE001 — reported, like a failed run
        error = " ".join(str(exc).split())[:300] or type(exc).__name__
        print(f"[-] {yaml_path.name}, case {index}: {error}")
        if json_out:
            print(json.dumps({"yaml": yaml_path.name, "case": f"{index:06d}",
                              "ok": False, "error": error}))
        return False

    params = {name: row.get(name) for name in stim.params}
    print(f"[*] Case {index:06d} of {yaml_path.name}: "
          + ", ".join(f"{name}={value}" for name, value in params.items()))
    for test in stim.tests:
        for val in test.value_lst:
            value = row.get(val.name)
            verdict = "PASS" if row.get(f"{val.name}_pass") else "FAIL"
            shown = f"{value:.6g}" if isinstance(value, float) else value
            print(f"    {val.name} = {shown}  {verdict}")
        error = row.get(_dl.tb_error_col(test.tb_path), _dl.NO_ERROR)
        print(f"    {test.tb_path}: {'ok' if error == _dl.NO_ERROR else error}")
    print(f"[+] Decks, logs and waveforms in {keep}")
    ok = row.get("sim_error", _dl.NO_ERROR) == _dl.NO_ERROR
    if json_out:
        print(json.dumps({"yaml": yaml_path.name, "case": f"{index:06d}", "ok": ok,
                          "params": params, "debug_dir": str(keep),
                          "sim_error": row.get("sim_error")}, default=str))
    return ok


def _select_datasheets(paths: list[Path], patterns: list[str] | None) -> list[Path]:
    """The datasheets whose file name or stem matches one of *patterns*.

    Shell-style and case-sensitive on every platform, so a selection written
    on one machine picks the same datasheets on another. No patterns: all.
    """
    if not patterns:
        return paths
    return [p for p in paths
            if any(fnmatch.fnmatchcase(p.name, pat) or fnmatch.fnmatchcase(p.stem, pat)
                   for pat in patterns)]


def main():
    # Pin the non-interactive backend before anything draws. pdf_export and
    # plot_manager use pyplot without choosing one, so with DISPLAY set and
    # PySide6 installed matplotlib picks QtAgg, and the CLI hung in Qt after
    # writing the PNGs, before the PDF existed.
    import matplotlib
    matplotlib.use("Agg")

    import chipify

    parser = argparse.ArgumentParser(
        description="Chipify: High-Performance Mismatch Simulation Wrapper for Xschem and Ngspice.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version=chipify.version_info(),
        help="Print the chipify version and the install path it resolves to.",
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="datasheet.yaml",
        help=(
            "Name of .yaml config file.\n"
            "(Searched in the input folder — 'datasheets/' by default,\n"
            "configurable via the in_dir key in settings.json.)\n"
            "Default: datasheet.yaml"
        ),
    )

    parser.add_argument(
        "--batch",
        metavar="DIR",
        default=None,
        help=(
            "Batch mode: run every *.yaml found in DIR.\n"
            "Results are written to per-datasheet subdirectories inside out/.\n"
            "A JSON summary line is printed for each run. The order is by\n"
            "file name, or datasheet_order in project.yaml."
        ),
    )

    parser.add_argument(
        "--only",
        metavar="PATTERN",
        action="append",
        default=None,
        help=(
            "With --batch: only the datasheets whose file name or stem\n"
            "matches PATTERN (shell-style, case-sensitive, e.g. 'corners_*').\n"
            "Repeat to select more. The combined report covers the selection."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print a JSON summary line to stdout after each run (useful for CI parsing).",
    )

    parser.add_argument(
        "--markdown",
        metavar="OUTPUT",
        default=None,
        help=(
            "After simulation, write a Markdown report to OUTPUT path.\n"
            "In --batch mode, OUTPUT is treated as a directory and one\n"
            "<datasheet>.md report is written per datasheet."
        ),
    )

    from chipify.engines import engine_names
    engines = list(engine_names())
    parser.add_argument(
        "--simulator",
        choices=engines,
        default=None,
        help=(
            "Default engine for this run, overriding the simulator_engine setting\n"
            f"({'|'.join(engines)}). Per-testbench 'engine:' keys in the datasheet\n"
            "still take precedence."
        ),
    )

    parser.add_argument(
        "--templates-dir",
        metavar="DIR",
        default=None,
        help=(
            "Skip xschem netlist generation and load pre-rendered Jinja2\n"
            "templates from DIR instead — useful for re-running a sweep\n"
            "against netlists that were already generated."
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Check the datasheet(s) instead of simulating: case count, each\n"
            "testbench's engine, its netlist (missing schematics, symbols,\n"
            "include files) and the first case's rendering. Exit code 1 when\n"
            "anything would fail. Nothing is written to history."
        ),
    )

    parser.add_argument(
        "--case",
        metavar="K",
        type=_non_negative_int,
        default=None,
        help=(
            "Simulate only case K (0-based; the run id 00000K) and keep each\n"
            "testbench's deck, simulator log and waveforms in\n"
            "out/debug/<datasheet>/case_00000K/. Nothing is written to\n"
            "history. With --batch: case K of every datasheet (a smoke test)."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the datasheet's interrupted run (aborted, crashed or\n"
            "killed) instead of starting a new one: only the cases it has no\n"
            "result for are simulated, with the run's own netlists and\n"
            "settings. --cores and --timeout still apply."
        ),
    )

    parser.add_argument(
        "--cores",
        metavar="N",
        type=_positive_int,
        default=None,
        help=(
            "Worker processes for this run, instead of num_cores in\n"
            "settings.json (default: the setting, else every CPU core)."
        ),
    )

    parser.add_argument(
        "--timeout",
        metavar="SEC",
        type=_positive_float,
        default=None,
        help=(
            "Time limit per simulation in seconds for this run, instead of\n"
            "sim_timeout_sec in settings.json."
        ),
    )

    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help=(
            "Output folder for this run instead of out/ (or out_dir in\n"
            "settings.json): results, history, waveforms, reports and\n"
            "chipify.log. --report-only reads the finished runs from there."
        ),
    )

    parser.add_argument(
        "--reports",
        action="store_true",
        help=(
            "After simulation, write the plots and reports declared in the\n"
            "datasheet's 'reports:' block to out/reports/<timestamp>/.\n"
            "With --batch, also write one combined PDF (a summary page, then\n"
            "every datasheet's report) to out/reports/<timestamp>/."
        ),
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Write the reports from the newest finished run instead of\n"
            "simulating (out/history/, or out/<datasheet>/history/ with\n"
            "--batch). Implies --reports."
        ),
    )

    parser.add_argument(
        "--progress-stream",
        action="store_true",
        default=False,
        help=(
            "Emit 'PROGRESS: <done> <total>' lines to stdout for each batch\n"
            "completion, so a parent process can track progress; harmless\n"
            "otherwise."
        ),
    )

    args = parser.parse_args()
    if args.only and not args.batch:
        parser.error("--only selects datasheets of a --batch run")
    if args.resume and args.report_only:
        parser.error("--resume simulates; --report-only does not")
    if args.resume and args.templates_dir:
        parser.error("--resume uses the netlists the run started with, not --templates-dir")
    if args.resume and args.only:
        parser.error("--resume continues the batch with the datasheets it started with; "
                     "drop --only")
    if args.preflight and args.case is not None:
        parser.error("--preflight checks without simulating; --case simulates one case")
    if args.preflight or args.case is not None:
        clash = [flag for flag, on in (("--resume", args.resume),
                                       ("--report-only", args.report_only),
                                       ("--reports", args.reports),
                                       ("--markdown", args.markdown)) if on]
        if clash:
            parser.error(f"{'--preflight' if args.preflight else '--case'} writes no "
                         f"results, so {clash[0]} does not apply")

    # One output root for everything this invocation writes. The settings
    # overrides go to run_sim, which pins them for its workers.
    out_root = Path(args.out).expanduser().resolve() if args.out else Path(settings.OUT_DIR)
    overrides = {"num_cores": args.cores, "sim_timeout_sec": args.timeout}

    # Initialised after parse_args so --help / --version don't open a log
    # file. Full DEBUG detail goes to out/chipify.log; the terminal keeps
    # the CLI's own [*] progress output and summary table, with only
    # warnings breaking in.
    import logging
    from chipify import app_config
    app_config.setup_logging(console_level=logging.WARNING,
                             log_path=out_root / "chipify.log" if args.out else None)

    # ── Checking, or one case: nothing goes to history ────────────────────────
    if args.preflight or args.case is not None:
        if args.batch:
            datasheets, _found = _batch_files(Path(args.batch), args.only)
        else:
            datasheets = [Path(settings.IN_DIR) / args.config]
            if not datasheets[0].exists():
                print(f"[-] Fatal Error: configuration file '{datasheets[0]}' not found!")
                sys.exit(1)
        ok = True
        for yaml_path in datasheets:
            if args.preflight:
                ok &= _preflight_one(yaml_path, simulator_override=args.simulator,
                                     templates_dir=args.templates_dir,
                                     json_out=args.json)
            else:
                ok &= _run_case(yaml_path, args.case, out_dir=out_root,
                                simulator_override=args.simulator,
                                templates_dir=args.templates_dir,
                                overrides=overrides, json_out=args.json)
        sys.exit(0 if ok else 1)

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        from chipify import report_service, run_store

        batch_dir = Path(args.batch)
        record = None
        if args.resume:
            record = run_store.find_unfinished_batch(out_root, batch_dir)
            if record is None:
                print(f"[-] Nothing to resume for {batch_dir}: its newest batch in "
                      f"{out_root / run_store.BATCH_DIR} finished, or it has none.")
                sys.exit(1)
            yaml_files = [batch_dir / name for name in record.names()]
            finished = sum(record.state(name) == "done" for name in record.names())
            print(f"[*] Resuming batch_{record.ts}: {finished} of {len(yaml_files)} "
                  f"datasheet(s) done")
        else:
            yaml_files, found = _batch_files(batch_dir, args.only)
            if args.only:
                print(f"[*] Batch mode: {len(yaml_files)} of {found} datasheet(s) "
                      f"selected in {batch_dir}")
            else:
                print(f"[*] Batch mode: {len(yaml_files)} datasheet(s) found in {batch_dir}")
            if not args.report_only:
                record = run_store.begin_batch(out_root, batch_dir,
                                               [p.name for p in yaml_files], args.only)
        summaries = []
        entries = []
        all_ok = True
        for yaml_path in yaml_files:
            print(f"\n{'='*60}")
            # Per-datasheet output subdirectory so consecutive runs don't
            # overwrite each other's simulation_results.csv.
            stem = yaml_path.stem
            run_out_dir = out_root / stem
            state = record.state(yaml_path.name) if args.resume else "pending"
            error = "no finished run" if args.report_only else "no data"
            try:
                if args.report_only:
                    entry = _load_finished_run(yaml_path, out_dir=run_out_dir,
                                               json_out=args.json)
                elif state == "done":
                    # Finished before the batch was interrupted: its run is
                    # loaded, not simulated again.
                    entry = _load_finished_run(yaml_path, out_dir=run_out_dir,
                                               json_out=args.json, reports=args.reports,
                                               run=record.run(yaml_path.name))
                else:
                    if record is not None:
                        record.set(yaml_path.name, "running")
                    entry = _run_single(
                        yaml_path, json_out=args.json,
                        simulator_override=args.simulator,
                        templates_dir=args.templates_dir,
                        progress_stream=args.progress_stream,
                        out_dir=run_out_dir, reports=args.reports,
                        overrides=overrides,
                        resume=(args.resume and run_store.find_unfinished(
                            run_out_dir, yaml_path.name) is not None))
            except Exception as exc:  # noqa: BLE001 — one bad datasheet must not end the batch
                # The datasheet is parsed before _run_single's own guard: a
                # malformed YAML ended the whole batch, so every later datasheet
                # and the batch summary were skipped.
                logging.getLogger("chipify.cli").exception(
                    "Batch entry %s failed", yaml_path.name)
                entry = None
                error = " ".join(str(exc).split())[:200] or type(exc).__name__
                print(f"[-] {yaml_path.name}: {error}")
            if record is not None and state != "done":
                record.set(yaml_path.name, "failed" if entry is None else "done",
                           run="" if entry is None else entry.summary.get("run", ""))
            if entry is None:
                all_ok = False
                entry = report_service.BatchEntry(
                    yaml_path, run_out_dir, error=error,
                    summary={"yaml": yaml_path.name, "error": error})
            elif args.markdown:
                _write_markdown(entry, Path(args.markdown) / f"{stem}.md")
            entries.append(entry)
            summaries.append(entry.summary)

        if record is not None:
            record.finish()
        batch_line = {"batch_summary": summaries}
        if args.reports or args.report_only:
            print(f"\n{'='*60}")
            result = report_service.generate_batch_report(entries, out_root)
            for warning in result.warnings:
                print(f"[!] {warning}")
            if result.pdf_report:
                print(f"[+] Combined report: {result.pdf_report}")
            batch_line["combined_report"] = (
                str(result.pdf_report) if result.pdf_report else None)

        print(f"\n{'='*60}")
        print(f"[*] Batch complete: {len(summaries)} run(s).")
        failed = [s for s in summaries if s.get("yield", 0) < 100]
        if failed:
            print(f"[!] {len(failed)} datasheet(s) with yield < 100%:")
            for s in failed:
                print(f"    {s['yaml']}: {s.get('yield', '?')}%")

        if args.json:
            print(json.dumps(batch_line))

        sys.exit(0 if all_ok else 1)

    # ── Single run ────────────────────────────────────────────────────────────
    yaml_path = Path(settings.IN_DIR) / args.config
    if not yaml_path.exists():
        print(f"[-] Fatal Error: configuration file '{yaml_path}' not found!")
        sys.exit(1)

    if args.report_only:
        entry = _load_finished_run(yaml_path, out_dir=out_root,
                                   json_out=args.json)
    else:
        print("[*] Initialising Chipify...")
        entry = _run_single(yaml_path, json_out=args.json,
                            simulator_override=args.simulator,
                            templates_dir=args.templates_dir,
                            progress_stream=args.progress_stream,
                            out_dir=out_root, reports=args.reports,
                            overrides=overrides, resume=args.resume)
    if entry is None:
        sys.exit(1)

    # Optional Markdown report
    if args.markdown:
        _write_markdown(entry, args.markdown)

if __name__ == "__main__":
    main()
