# Project Briefing: Chipify

## 1. Project Overview
*   **Project Name:** `Chipify` (Core CLI/Engine & PySide6 (Qt) Desktop GUI).
*   **Purpose:** A high-performance EDA (Electronic Design Automation) tool for mismatch simulations, parameter sweeping, and yield analysis wrapping around Xschem and Ngspice.
*   **Core Tech Stack:** Python 3.11+, `PySide6` (Qt GUI), `pandas` (Data Crunching), `matplotlib` & `scipy` (Visualization & Stats), `multiprocessing` (Parallel execution), `jinja2` (Netlist templating), `asteval` (sandboxed expression evaluation).

---

## 2. File Architecture & Modules (v0.2 — post-Phase-1 refactor)

### Engine (no Tk deps)

| Module | Purpose |
|---|---|
| `settings.py` | Project paths (`IN_DIR`, `OUT_DIR`, `WORK_DIR`, `TB_DIR`, `FAST_TMP`). The four project folders are configurable via `settings.json` (keys `in_dir/out_dir/work_dir/tb_dir`); missing/blank ⇒ default structure. `FAST_TMP` is fixed. |
| `util.py` | Domain objects: `Stimuli`, `Test`, `Value`. Delegates YAML parsing to `schema.py`. |
| `schema.py` | `validate_datasheet()` — validates `datasheet.yaml` against typed schema; safe `_parse_range_dsl()` for `range/linspace/logspace` strings. |
| `expression.py` | `SafeEvaluator` — sandboxed `asteval`-backed evaluator with `numexpr` fast path. Replaces all `eval()` / `df.eval()` call-sites. |
| `templating.py` | `render_netlist()` — the one Jinja2 configuration for per-sample netlists (simulation workers and the netlist export): `StrictUndefined`, and `keep_trailing_newline` because VACASK needs the deck's final newline. |
| `run_store.py` | `save_run()` — the one way a finished run is written, for the CLI and the GUI alike: latest CSV, history copy, both sidecars, the archived netlist templates, the datasheet snapshot (`Stimuli.source_bytes`, the bytes it parsed, with their SHA-256 in the sidecar) and the `.latest` waveform pointers. The run's settings and engines arrive as `df.attrs["run_settings"]` / `["engines"]` from `run_sim`. |
| `app_config.py` | Persistent user preferences (`settings.json`), application-wide logging setup. |
| `simulator.py` | Multiprocessing simulation engine (`NgspiceSimulator`, `VacaskSimulator`). File-based abort via `/tmp/sim_work/abort.flag`. |
| `plot_manager.py` | All Matplotlib logic. Avoids GUI bloat. |
| `data_loader.py` | Results loading / pass-fail / plot-column classification / history — shared by the engine, exporters, and GUI; headless, no GUI deps. |
| `reports.py` | Vocabulary for the datasheet's `reports:` block: `PlotSpec` / `ReportsConfig` and the `PLOT_TYPES` registry. |
| `report_service.py` | Renders that block headlessly to `out/reports/<timestamp>/` — the single path behind `--reports`, the GUI button and the one-click PDF (`config=pdf_only_config()`). |
| `cli.py` | Entry point for headless execution or launching the GUI (`run_gui()` → Qt app). |

### Toolkit-agnostic GUI-support layer (`chipify/uikit/`)

```
uikit/                      – NO GUI-toolkit imports; unit-testable headlessly
├── state.py                – AppState (single source of truth) + Signal pub/sub
├── services/
│   ├── equation_service.py – apply_scalar_equations, apply_transient_equations (via SafeEvaluator)
│   ├── measurements.py     – measurement_rows / equation_rows / worst_cases / error_rows (PASS/FAIL/ERROR stats; shared by GUI + analyzer + md/pdf exports)
│   ├── transient_loader.py – resolve_analysis_dir, list_analysis_signals, load_analysis_df,
│   │                         list_kind_signals, run_pass_map / run_group_map (waveform grouping)
│   ├── hover_bubble.py     – the tooltip bubble + fmt_value, shared by both hover managers
│   ├── scatter_hover.py    – matplotlib scatter hover/click manager
│   ├── curve_hover.py      – waveform-curve hover: nearest-curve hit test, blitted tooltip + highlight
│   ├── netlist_export.py   – per-sample SPICE netlist rendering (pure)
│   ├── yaml_editor_service.py – get_params_dict, get_tests_dict, gui_repr_param, sync_form_to_yaml
│   └── plugin_context.py   – PluginContext facade handed to tab plugins (see PLUGINS.md)
└── widgets/
    └── yaml_dumper.py      – QuotedString + inline-list YAML representers
```

### Qt GUI Package (`chipify/gui_qt/`)

```
gui_qt/
├── app.py                  – QApplication bootstrap + main() (the `chipify` GUI entry point)
├── main_window.py          – QMainWindow shell: left control panel + QTabWidget + status bar
├── theme.py                – night/dark/light palettes → QSS + QPalette; plot_theme()
├── controllers/            – simulation_controller, history_controller (Qt signals, no after())
├── workers/sim_worker.py   – QThread worker emitting queued progress/chunk/finished signals
├── services/               – throttle, canvas_menu (QMenu), figure_export, latex_export
├── tabs/                   – editor / measurements / histogram / analytics / plots / equations
└── widgets/                – settings_dialog, run_annotation_dialog, mpl_canvas, helpers
```

Plugins: the Qt GUI loads `QtTabPlugin`s; legacy Tk `TabPlugin`s are detected and skipped with a warning (`plugin_loader.warn_unsupported_tab_plugins()`).

### Tests (`tests/`)

| File | Covers |
|---|---|
| `test_expression.py` | SafeEvaluator sandbox, helpers, SPICE sanitization, security |
| `test_util_range_dsl.py` | `_parse_range_dsl` whitelist, `validate_parameters` |
| `test_equation_service.py` | apply_scalar/transient equations, NaN propagation |
| `test_yaml_editor_service.py` | get_params_dict, get_tests_dict, gui_repr_param |
| `test_data_loader_history.py` | `data_loader.list_history_runs` |
| `test_netlist_export.py` | per-sample netlist rendering (pure logic) |
| `test_plugin_context.py` | `PluginContext` facade, JSON-serialization |
| `test_gui_qt_smoke.py` | Qt GUI smoke tests (offscreen): window, tabs, themes, worker, plugins |
| `test_measurements_errors.py` | Per-testbench error scoping, ERROR status, `error_rows`, lost-batch rows |
| `test_curve_hover.py` | Waveform-curve hover: nearest-curve hit test, tooltip text, redraw lifecycle |

---

## 3. Critical Technical Constraints (DO NOT REVERT)

### Process Management & Abort System (`simulator.py`)
*   Do **NOT** use `ProcessPoolExecutor`, `Manager().Event()`, or signals to stop simulations. They fail to kill underlying Ngspice C-processes and cause RAM leaks.
*   **The Fix:** File-based abort flag at `FAST_TMP/abort.flag` (`/tmp/sim_work/abort.flag`). The worker loop polls `os.path.exists(flag)` every 0.1 s; if found, it executes a hard `process.kill()`. GUI stop button → `simulation_controller.stop_simulation()` → `simulator.abort_simulation()`.
*   **Orphaned workers:** `abort.is_aborted()` is also true in a worker whose chipify process died (`abort.parent_gone()`, via `multiprocessing.parent_process()`, which tracks the creating process for spawn and forkserver alike). A killed CLI therefore stops its workers through the same poll-and-kill path as the abort flag, and `simulate_case_batch` starts no further case once aborted. No signals involved.
*   **Lost tasks:** `Pool` replaces a worker that dies abruptly but never resolves the task it was running, so that AsyncResult stays unready forever. Each worker writes its process name into a claim file per batch (`FAST_TMP/claims_*/batch_<i>`), and `simulator._LostBatchDetector` turns unready batches whose claimant has left `multiprocessing.active_children()` into `WORKER_LOST` rows. After a lost task the pool must be `terminate()`d, never `close()`d: a closed pool waits for its task cache to drain, so `join()` would never return.
*   **A run's record starts before its first case** (`run_store.begin_run`, called by `run_sim` once the netlists exist): the sidecar with `"complete": false`, `planned_runs`, the datasheet copy, the templates archive and the journal `history/run_<ts>.rows.jsonl`, one line per finished worker batch (`{"t": <unix time>, "rows": [...]}`, flushed and fsynced). `run_store.save_run` completes the same record (`df.attrs["run_ts"]`) and deletes the journal. History listings only glob `run_*.csv`, which is what keeps an unfinished run out of the GUI's history and `--report-only`; never give a journal or snapshot a `.csv` name. `--resume` (`run_store.find_unfinished`) only takes the datasheet's newest run, refuses another datasheet hash, case count or engine set (run ids are positions in `generate_cases`), reuses the archived templates and the result-shaping settings (everything but `num_cores`, `chunk_size`, `process_start_method`), and reruns only missing cases and `WORKER_LOST` rows. A record or journal that cannot be written disables resuming, never the sweep. A `--batch` (not `--report-only`) keeps `<out>/batches/batch_<ts>.json` (`run_store.BatchRecord`: each datasheet `pending`/`running`/`done` with its history run/`failed`, replaced atomically after every change, `complete` once the loop ends, also when datasheets failed); `--batch DIR --resume` takes the newest record of that folder only if it is not complete, loads `done` datasheets from their recorded run and resumes the others' unfinished runs.
*   **A run's settings are pinned in its workers.** `run_sim` takes the config once (plus the CLI's `--cores` / `--timeout` as `overrides`), keeps `app_config.RUN_KEYS` with the resolved worker count, chunk size, start method and timeout, and sends that dict with every batch; `simulate_case_batch` pins it with `app_config.set_run_overrides()` for the batch, so every `load_config()` in a worker (the per-case timeout, the VACASK engine's binary and paths) returns the run's values. Workers read the config per case, so without the pin an override never reached them and a `settings.json` edit changed a running sweep. Never pin in the parent process: in the GUI the settings dialog reads the config and would save pinned values back into `settings.json`. The one exception is `simulator.run_case` (the CLI's `--case`), which simulates in its own process and pins only for that case.
*   **`--preflight` and `--case` write no history** (`simulator.preflight`, `simulator.run_case`): they reuse case generation, engine resolution and netlisting, never start a pool, and `run_case` passes `keep_dir` to `_simulate_single_case`, which saves each deck exactly as handed to `engine.run()` (the text the simulator runs) and the engine's log via `run_log_tail`, so the engine API stays as it is.

### Scalar capture: chipify owns the `MY_DATA:` line (`simulator.py`)
*   ngspice scalar measurements come back on a single `MY_DATA:` **stdout** line. **Do not hand-write `echo MY_DATA:` in a testbench** — `NgspiceSimulator._finalize_netlist` (called by `generate_test_template`) auto-injects it from the datasheet's `value_lst` (`echo MY_DATA:$&<key0> $&<key1> …`, in order) via `_inject_capture`, and strips any stale hand-written one so only chipify's survives.
*   **Anchors are whole lines** (`ngspice._RE_CONTROL` / `_RE_ENDC` / `_RE_QUIT`): `.control`, `.endc` and `quit`/`exit` count only as their own lines, optionally behind the `* {{ NL }}` prefix testbenches use to hide a command from xschem's own simulation. Matching plain text made a comment mentioning `.control` the anchor.
*   **The contract:** each scalar key under `tests.<tb>` must name an ngspice vector the testbench defines with `let`/`meas` — `$&<key>` has to resolve. The run side parses the same line positionally back into `value_lst`, so chipify controls both ends and the order can't drift. Waveforms work the same way: declare `transient/dc/ac_signals` and chipify injects the `wrdata`/`setplot` capture (`analyses.py`).
*   Vacask is unaffected — it extracts scalars from the `.raw` file (named `meas` results / `measure:` exprs) or its own `printf "MY_DATA: …"`.
*   **Imported netlists (`source: netlist` key):** a testbench may set `source: netlist` (default `xschem`) to load an existing deck at `tb/<tb_path><engine.netlist_ext>` (`.spice`/`.sim`, via `safe_tb_file`) instead of netlisting a `.sch` via xschem. The ngspice path runs the imported deck through the same `_finalize_netlist`, so managed capture is identical; vacask reads the `.sim` verbatim and captures from the `.raw` as usual. `--templates-dir` still short-circuits before the engine, so it overrides `source: netlist`.

### Xschem netlists with missing symbols (`engines/xschem.py`)
*   `run_xschem` rejects a SPICE netlist containing xschem's `* <inst> - <sym> IS MISSING !!!!` comment (`missing_symbols`), which fails that testbench via `template_error`. The exit code cannot be used: xschem 3.4.8RC returns rc=10 for clean schematics too. `--spectre` netlists drop a missing instance with no marker, so the VACASK xschem path is not covered.

### Safe Expression Evaluation (`expression.py`)
*   **Never** call Python `eval()`, `exec()`, or `df.eval(engine='python')` directly in user-facing code.
*   **The Fix:** Route all expression evaluation through `SafeEvaluator` (module-level singleton: `from chipify.expression import default_evaluator`). It blocks `__import__`, `open`, `exec`, and dunder attribute access via `asteval`.
*   **Never cast measure data to float.** AC vectors are complex: `evaluate_spice_measure` and the `db`/`first`/`last` helpers keep them complex (`expression._numeric`), and a measure result that is still complex raises `ExpressionError` (`_real_result`). The float cast kept only the real part, with nothing but a numpy ComplexWarning.

### YAML Range Parsing (`schema.py`)
*   **Never** use `str.replace("range(","")` or raw `eval()` to expand YAML range strings.
*   **The Fix:** `_parse_range_dsl(value)` uses `ast.parse` + an allowlist (`range`, `linspace`, `logspace`) with constant-only arguments. Any other node raises `SchemaError`.

### Matplotlib Ghosting (`plot_manager.py`)
*   When drawing plots with colorbars, calling `ax.clear()` leaves ghost colorbars stacking up.
*   **The Fix:** Always use `fig.clf()` then `ax = fig.add_subplot(111)` when switching plot modes.

### Sweep vs Output Column Separation
*   The "Corner Yield Matrix" requires discrete inputs. Mixing continuous output columns causes a pivot table crash.
*   **The Fix:** `data_loader.compute_plot_cols()` returns a typed `PlotColumns` dataclass separating `sweep_params` from numeric output columns. `data_loader.valid_rows(df)` is the single filter for `sim_error == 'None'`.

### No vacuous plans
*   `validate_datasheet` rejects a block of the wrong type (`_mapping_block`, list-valued limits, string-valued `*_signals`, non-mapping `measure:`); it used to skip them, so a list under `tests:` became an empty plan. Empty (`null`) blocks still load, because a datasheet is built up in the GUI.
*   `run_sim` refuses a plan with no testbench, or a testbench with no measurement, `measure:` or signals (`_check_executable_plan`): every case of such a plan comes back without a verdict column, and `global_pass` reads that as passed.
*   It also refuses to dispatch when *no* testbench could be netlisted (`_check_any_netlist`). One failed testbench still fails only its own rows; the rule only stops a run whose every case is a foregone failure.

### One results column per name (`schema._check_result_columns`)
*   Results are stored by name alone (`sample[name]` per run), so `validate_datasheet` rejects a measurement (value or `measure:` entry) declared by two testbenches, a measurement named like an input parameter, and any datasheet name equal to a column chipify writes itself (`data_loader.RESERVED_COLUMNS`, `<tb>__error`, `<tb>_overall_pass`, `<measurement>_pass`). A shared column let one testbench overwrite another's value while the error scope still pointed at the first.

### Error Handling in DataFrames
Errors are recorded at **two scopes**, because a datasheet can hold several testbenches and one of them failing must not invalidate the others.

*   **Per testbench** — `<tb_path>__error` (`data_loader.TB_ERROR_SUFFIX`), written by `simulator._record_error`. `'None'` when that testbench succeeded. This is the authoritative scope for anything measurement-related.
*   **Per row** — `sim_error` is the roll-up of every testbench failure in that run (accumulated, `' | '`-joined), used for run-level yield and plotting.

Which filter to use:

*   `data_loader.valid_rows(df)` — row-level. Only for plotting and yield: it keeps a row only when *every* testbench in it succeeded.
*   `data_loader.measurement_ok_mask(df, tb_path, name)` — per measurement. Required for measurement statistics. Also excludes NaN values, so an engine that silently fails to resolve a signal is caught.
*   Both fall back to `sim_error` when the per-testbench column is absent, so pre-existing history CSVs keep loading.

**Never compute measurement statistics from a `valid_rows`-filtered frame.** A testbench that fails in every corner sets `sim_error` on every row, `valid_rows` then returns *nothing*, and `Series([]).all()` is vacuously `True` — which is exactly how a completely failed run used to report `PASS` with blank statistics. `uikit/services/measurements.py` takes the **full** frame and scopes validity itself; `measurement_rows`, `worst_cases` and `error_rows` are the single implementation shared by the Qt tab, `analyzer.py`, `md_export.py` and `pdf_export.py`.

A measurement's status is `PASS` / `FAIL` / `ERROR` (`measurements.STATUS_*`). `ERROR` means at least one run produced no trustworthy value at all — a different problem from an out-of-spec `FAIL`, and never reported as `PASS`.

**Plots follow the same rule.** `data_loader.plot_rows(df, stim, columns)` returns the rows usable for the measurements being plotted; an empty `columns` means no constraint. `plot_manager.draw_histogram` / `draw_adv_plot` take the **full** frame and scope it themselves (`_adv_plot_columns` maps each mode to the measurements it plots), so tabs must not pre-filter — passing a `valid_rows` frame blanked every chart as soon as any testbench failed.

Yield *visualisations* use `data_loader.effective_pass(df)`, not `global_pass`: `global_pass` ANDs every testbench, so one permanently broken testbench drives it false everywhere and flattens the Corner Yield Matrix to 0 %. `effective_pass` ignores the testbenches that errored in a row (NaN when none ran, so those rows drop out of a `mean`), and the matrix names the excluded testbenches in its title. `global_pass` remains correct for run-level yield counts.

`run_sim` returns `None` **only** on user abort; every other failure raises, so callers must report it rather than mistake it for a cancellation.

A run reloaded from history (`HistoryController.on_select`, `cli._load_finished_run`) goes through `data_loader.regrade(df, stim)`: every view prints the datasheet's *current* limits, so the saved `<name>_pass` / `<tb>_overall_pass` flags are recomputed with `Value.isPass` against them. No datasheet snapshot is stored with a run, so "the verdict at the time" cannot be reconstructed.

A declared analysis (`transient/dc/ac_signals`) that produced no waveform CSV is a **testbench error** (`CAPTURE_ERROR`, recorded through `_record_error`), not only the `<tb>__<kind>_capture` diagnostic note: as a note alone it left `sim_error` `'None'` and the run passing with its curves missing. Engines name the cause through `BaseSimulator.capture_error(kind)`; the VACASK writers raise `analyses.CaptureError` for a declared signal missing from the `.raw` and never substitute zeros.

### Auto-generated reports (`reports.py` + `report_service.py`)
*   The datasheet's optional `reports:` block declares which figures and reports a run should produce; `schema._validate_reports` turns it into a `ReportsConfig` on `Stimuli.reports`, exactly as `equations:` becomes `Stimuli.equations`.
*   `PLOT_TYPES` in `reports.py` is the **single registry** the validator, the renderer and the tests read. Adding a plot type is one entry — a type that is not in the render matrix fails `test_registry_and_test_matrix_stay_in_step`.
*   `report_service.generate_reports` is headless: figures come from the same `PlotManager` entry points the tabs use, drawn on a `FigureCanvasAgg`; images are written by the `ExporterPlugin` registry (so a user exporter is a usable `formats:` value), and PDF / Markdown / LaTeX delegate to the existing generators. **No plotting code lives in the service.**
*   Each plot is rendered under its own guard: a spec naming a missing measurement, or a waveform directory that was never produced, records a warning in `ReportResult.warnings` and the remaining plots still render. `latex` is only honoured for the types whose `PlotType.supports_latex` is true.
*   Output goes to `out/reports/<timestamp>/` with an `out/reports/.latest` pointer — the convention `simulator.write_analysis_pointers` uses for analysis data.
*   Opt-in: `chipify-cli --reports`, or the GUI's *Generate Reports* button (which works on whatever run is loaded, including one picked from history).

### Loaded results keep their datasheet
*   `Stimuli.yaml_path` records the file a plan was parsed from. `MainWindow._run_report_job` and `reapply_equations` use the loaded stim's path (`_results_yaml_path`), not the datasheet combo: the combo can change without a reload (the ↺ rescan resets it), and a report once paired the selected datasheet's path with another run's tests.
*   Selecting a datasheet (`_on_datasheet_changed`, `set_active_datasheet`) clears the results and loads that datasheet's newest run, so the tabs never show one datasheet's run under another's name.

### One report path, one run-selection, one formatter
*   `report_service.generate_reports(..., config=...)` overrides the datasheet's block. That override *is* the one-click PDF: `MainWindow` has a single `_run_report_job` helper and no separate `export_pdf`, so both write to `out/reports/<timestamp>/` and share the guard / status / dialog handling. A PDF must never land loose in `out/reports/` again.
*   `transient_loader.select_run_ids(df, mode, n=, custom=)` owns run selection for the Plots tab, the dashboard cell **and** a datasheet's `runs:` key, including the padding and the `RUN_CAP`. Canonical modes are `all_valid` / `failing` / `first` / `custom`; `RUN_MODE_LABELS` maps the GUI labels onto them and `schema._validate_reports` validates `runs:` against the same vocabulary, so the datasheet and the GUI cannot disagree about which runs a plot covers.
*   `measurements.fmt_eng` is the engineering-unit formatter shared by the Markdown and PDF reports (`373.5 m`, `2.686 G`). `measurements.fmt_value` stays the plain 4-digit form for the GUI table and for **dimensionless** quantities — Cpk must never be rendered as `380 m`.
*   `plot_manager.param_plugin_modes()` is the one home for the plugin-mode lookup the Analytics tab and the dashboard cell both need.
*   `tests/test_no_duplicate_functions.py` fails on any module-level function name defined in two files, with an `_ALLOWED` map carrying the reason for each deliberate exception (`main`, `fmt_value`) — the audit is re-run by the suite rather than by hand.

### Batch runs and the combined report (`cli.py`, `report_service.generate_batch_report`)
*   `--batch` gives every datasheet its own output root, `out/<stem>/`: results, `history/`, `analysis_data/` (via `run_sim(out_dir=…)` and `write_analysis_pointers(…, out_dir)`) and `reports/`. Run ids restart at `000000` for every run, so a shared waveform folder let two datasheets on one testbench read each other's curves without any error. Waveforms always live under the run's own output root.
*   Sidecars store `analysis_dirs` relative to that root (`run_meta.output_root`). `run_meta.analysis_dirs_for` resolves them (old absolute paths still work), and `data_loader.load_csv` restores them into `df.attrs`, so a reloaded history run finds its own waveforms rather than the `.latest` pointer's.
*   Recorded directories are authoritative: `transient_loader.resolve_analysis_dir` returns `""` when the run recorded its waveform directories but the one for that kind is missing or was never captured. The `.latest` / newest-subdir guesses are only for runs that recorded none.
*   The combined PDF is a summary page followed by the datasheets' own report PDFs, merged with pypdf (`pdf_export.generate_batch_pdf`): **nothing is rendered twice**. The summary is laid out before it is drawn, so its page count, and with it every section's start page, is known up front. A datasheet whose `reports:` block asks for no PDF gets one rendered in a scratch dir for the combined report only; its own outputs stay exactly as declared.
*   Summary numbers come from `result_summary` / `measurement_rows`, the same as each datasheet's cover. `--report-only` rebuilds everything from each datasheet's newest `history/run_*.csv`.
*   The CLI pins matplotlib's Agg backend at the top of `main()`: `pdf_export` draws through pyplot, which picks QtAgg when a display and PySide6 are present and then hangs a headless run.

### The PDF contains exactly the configured plots
*   `report_service.render_spec(spec, df, stim, out_root=…, theme=…, figsize=…)` is the public renderer for one plot spec. `pdf_export._add_report_plots` calls it once per entry in `reports: plots:` and gives each its own page — the renderers all `fig.clf()`, so they cannot share one.
*   `reports: plots_per_page: 2` stacks two plots per portrait page at reduced height. Because the renderers cannot share a figure, each plot is drawn on its own A4-wide figure of `_SLOT_H` inches, saved as a one-page PDF, and placed onto the page with pypdf after the report is closed (`_place_slots`): vector throughout, never rasterised. Without pypdf it falls back to one plot per page with a warning.
*   With a `plots:` list the report's figure pages are **exactly** those plots; `_add_histograms` / `_add_correlation` only run when nothing is declared. `generate_pdf_report(reports_config=…, out_root=…)` takes the effective config so a PDF-only run still gets the automatic sections.
*   Pages are written inside `exporters._white_bg.white_background(fig)` — the same re-skin the PNG/SVG exporters apply. `PdfPages.savefig` bypasses `save_with_white_bg`, so without it the report embedded the dark on-screen palette while the exported image of the same plot came out light.
*   Plot pages are **portrait A4** like every other page. The renderers lay out for their own proportions, so `_fit_axes_to_band` remaps the union of their axes into `_PLOT_BAND` — proportionally, which keeps the Bode magnitude/phase pair stacked and a colorbar beside its heatmap. Letting a plot fill the sheet would stretch it into a tall strip.
*   Pages carry the standard `_page_header` banner. `white_background(fig, axes=…)` takes the plot axes only, so the re-skin does not white out the banner.

### One histogram, every output format
*   `reports.histogram_options(spec)` is the single place a histogram's `fit` / `group` / `bins` / `zoom` are resolved, and `reports.histogram_spec_for(stim, param)` finds the datasheet's spec for a measurement. Both the standalone figure (`report_service`) and the PDF report page (`pdf_export._add_histograms`) go through them, so a measurement cannot render differently in two formats.
*   `pdf_export` no longer has its own histogram implementation. It draws through `PlotManager.draw_histogram` with `plot_manager.PRINT_THEME` and `tight=False` (its axes are placed by hand, so `tight_layout` would move them), then `_finish_pdf_hist` applies the page's type scale, pins the legend left and adds the μ/σ/Cpk badge. The old `_draw_hist_ax` — which hardcoded `bins="auto"`, no grouping and an always-on zoom — is gone.
*   **`zoom` defaults to True for report figures.** A spec far wider than the spread (a ±10 mV limit on a 60 µV distribution) otherwise collapses every bar into a sliver against a spec-width axis. The PDF always behaved this way; the default makes every format agree. Set `zoom: false` in the plot spec to see distant spec lines instead.

### Equations panel edits
*   The table is a staging area that `▶ Apply` persists — but **removing a row saves immediately**, because a deletion left pending was reverted by the next `reload()` and looked like "I cannot delete this". Add/edit still need Apply.
*   `_dirty` tracks hand edits (`itemChanged`, suppressed while `_loading`), so a `reload()` that drops unsaved work says so rather than reverting silently. `_apply()` returns a bool so a removal knows whether it actually persisted.
*   **Equations only add columns.** `equation_service.apply_scalar_equations` refuses an equation whose name is already a column of the frame (measurement, input, bookkeeping, earlier equation): a replaced measurement kept its simulated pass flags. `MainWindow.reapply_equations` therefore drops the previous `derived_cols` before re-applying, which is exactly the frame as simulated; a removed or newly failing equation would otherwise keep its old column.

### Editing `reports:` from the GUI (`gui_qt/widgets/reports_dialog.py`)
*   Opened by **Reports…** in the Datasheet Editor toolbar, because the block belongs to the datasheet — chipify's paradigm is that a datasheet is fully buildable from the GUI.
*   Persists through `editor_tab.set_document_key("reports", …)`, the same call the equations panel uses: one writer for top-level datasheet keys, and pending form/raw edits are synced first. An empty config renders as `None`, which removes the key instead of leaving an empty husk.
*   The per-plot form is **generated from `reports.PLOT_TYPES`** — a new plot type gets a GUI for free and the dialog never becomes a second place that knows the plot vocabulary. `_FIELD_KIND` maps each option *key* (not each type) to a widget kind, so nine entries cover every type.
*   Dropdown names come from the loaded run when there is one, else from the datasheet parsed through `validate_datasheet` — the block must be configurable before the first simulation.
*   Values a freshly built combo *displays* are committed into the spec (`_commit_displayed`), because otherwise a field the user can plainly see filled in would be absent from the saved YAML and the schema would reject it on the next load.

### Waveform overlay grouping (`plot_manager._OverlayStyle`)
*   The Transient / DC sweep / Bode overlays share one colour-and-legend policy. Ungrouped, colour encodes the signal (or the run index for a single signal) and failing runs are red — unchanged.
*   Passing `group_map` (`run_id -> value`, from `transient_loader.run_group_map`) plus `group_label` switches colour to the **group value**, line style to the signal, and turns pass/fail highlighting **off** — red would pull failing curves out of the grouping. Legend labels use `temp=27`, matching `draw_histogram`'s convention.
*   The Plots tab and the dashboard's `Plots` cell both drive this through the same helpers; the cell's legacy `"Transient"` mode name is migrated in `apply_config`.

### Waveform-curve hover (`uikit/services/curve_hover.py`)
*   Consumes the ``line_map`` (``Line2D -> (run_id, signal)``) the three overlay plotters already return; the Plots tab and the dashboard cell cache one `CurveHoverState` per redraw and hand it over, because `get_state` runs on every mouse move.
*   **Never** `Line2D.contains()` in a loop (~150 ms per mouse move at `RUN_CAP` runs) and **never** `canvas.draw_idle()` per event (~1 s at 500 curves — the scatter hover can afford it, an overlay cannot). Each axes gets a lazily built index of its curves in *scale space*; only the `transLimits + transAxes` affine is re-read per event, so pan/zoom/resize need no rebuild, and the bubble is blitted over a saved background. Measured: 4 ms per hover against 406 ms for one redraw.
*   **Never decimate the index.** Striding through the samples (the old 120k cap) made the hover name the wrong run in dense overlays. `_CurveIndex` keeps every sample and groups them into at most `MAX_INDEX_CHUNKS` chunks with scale-space bounding boxes; a hit tests the boxes, then measures only the segments of the chunks within reach, which gives exactly the brute-force answer.
*   Distance is to the drawn **segment**, in **pixels**, and the *nearest* curve wins: a sparse DC sweep is hoverable between its samples, a log x axis is not measured in decades, and 500 overlapping Monte-Carlo curves do not all answer with the lowest run id.
*   A log axis maps `x <= 0` to the sentinel `-1000`, not `-inf`, so non-positive samples are masked on the raw values before transforming — otherwise a phantom point a thousand decades left wins every hit near the axes edge. Samples either side of a NaN dropout are never joined into a segment, because matplotlib draws a gap there.
*   `line_map` keys are the six-digit ids from the `run_<id>__<tb>.csv` filenames; the results frame's `run_id` reads back from CSV as an int. The run lookup goes through `transient_loader.pad_run_id`, as `run_pass_map` / `run_group_map` do.
*   Both hover managers can share one canvas (the dashboard cell): their `get_state` callbacks are mutually exclusive on the cell's mode, and both are invalidated *before* each redraw, which is what also covers the cell's error branch.
*   The hovered curve is also drawn thicker, opaque and above its neighbours. Unlike the bubble it is an **ordinary artist**, so: the blit pass redraws it over the restored background (the thicker stroke covers the thin one already there, and simply not drawing it is what un-highlights); its original linewidth/alpha/zorder are saved and put back on every exit path — miss, leave, `invalidate()`; and when a real draw catches it mid-highlight, `_on_draw` restores it, *discards* the frame it just saved and schedules one more draw, because a background with the thick curve baked in would smear it until the next redraw.
*   A full draw can run *inside* the hover's own `blit()`: Qt's `blit()` is a synchronous repaint whose paint event first runs any queued draw. `_repaint` notices it and handles the same motion once more (which saves a clean frame and shows the tooltip again); `_on_draw` must not schedule its extra draw for that case, because a draw queued then would run inside the next blit as well. Motions that hide the tooltip are never replayed.
*   The bubble is animated so ordinary draws skip it — but `savefig` draws animated artists, so leaving the canvas hides it (and restores the curve), keeping both out of `Export…`, the toolbar's own Save, and the TeX export.
*   `HoverBubble._place` works in **physical** pixels (`get_width_height(physical=True)`, `OFFSET` converted from points), like events and window extents, and positions the bubble's box explicitly: beside the point where it fits, else clamped inside the canvas. Before measuring it calls `update_positions` and `update_bbox_position_size`, because both the text's offset transform and the rounded box are otherwise laid out only by the next draw.
*   `HoverBubble.show` makes the annotation visible *before* measuring it: `Text.get_window_extent` reports a 1-pixel box for a hidden artist, so the edge-flip silently never fired and a tall tooltip near the top of the canvas was drawn off it.

---

## 4. Architecture Rules (Phase 1 invariants)

1. **The core and the `uikit/` layer** (`uikit/services/`, `uikit/state.py`, `data_loader.py`, `expression.py`, `schema.py`) **never import a GUI toolkit**. This keeps them unit-testable without a display; all Qt code lives under `gui_qt/`.
2. **Tabs never call `simulator.*` directly.** They dispatch through a controller (`SimulationController`).
3. **State is mutated only through `AppState`.** Subscribers receive notifications via `Signal.emit()`; the `QThread` sim worker delivers cross-thread updates as queued Qt signals.

---

## 5. Development Commands

```bash
# Install (engine only)
pip install -e .

# Install with optional fast vectorised evaluation
pip install -e ".[fast]"

# Run tests
pytest

# mypy strict check on typed modules
python -m mypy chipify/expression.py chipify/schema.py chipify/uikit/state.py \
    chipify/uikit/services/ chipify/uikit/widgets/ chipify/data_loader.py \
    chipify/util.py chipify/app_config.py --strict

# Launch GUI (PySide6/Qt)
chipify
```
