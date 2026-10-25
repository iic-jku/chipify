# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Preparation for the initial public release.

### Added
- `chipify-cli --batch DIR --reports` writes one combined PDF,
  `out/reports/<timestamp>/batch_report_<timestamp>.pdf`: a summary page with
  one row per datasheet (runs, valid, passed, yield, time, start page), the
  datasheets' descriptions and failing measurements, then every datasheet's
  own report, with one bookmark per datasheet. The `batch_summary` JSON line
  gains `combined_report`. Title and subtitle come from `name` and
  `description` in `project.yaml`; its new `datasheet_order` key sets the
  batch order (by default, by file name).
- `--report-only` writes the reports (and, with `--batch`, the combined PDF)
  from each datasheet's newest finished run, without simulating again.
- Datasheet `description:` key: one line shown on the report cover and on the
  batch summary page, editable in the Datasheet Editor's form view.
- `reports: plots_per_page: 2` puts two of the configured plots on each
  portrait A4 page of the PDF report, each at reduced height (about 3.7 in
  instead of 5.6 in), with the plot's name above it. Default stays one plot
  per page. Also in the Reports dialog ("Plots per PDF page").
- Hovering a curve in the **Plots** tab (and the dashboard's `Plots` cell)
  draws that curve thicker and fully opaque, and shows the run's details in
  the same tooltip the scatter plot uses: run id, the sample under the cursor,
  PASS/FAIL, and the swept parameters. Works on all three analysis kinds,
  including both panes of the Bode plot. With many runs overlaid the *nearest*
  curve is picked, so pointing at an outlier answers "which run is that?"
  without reading the legend. The highlight is undone as soon as the pointer
  moves away or leaves the plot, so it never reaches an export.
- Modular simulator-engine architecture (`chipify/engines/`): each simulator
  is one `BaseSimulator` class resolved through a registry, mirroring the GUI
  plugin system. New engines can be added as a single built-in module, a
  drop-in plugin file in `~/.chipify/plugins/` (see PLUGINS.md, "Simulator
  engine plugin"), or via `register_engine()`. The datasheet schema, the CLI
  `--simulator` choices, the Settings dialog, and the editor's per-testbench
  engine dropdown all read the registry instead of hardcoded name lists.
  `chipify.simulator` keeps re-exporting the engine API for compatibility.
- `sim_timeout_sec` setting (Settings → Simulation → "Per-run timeout"):
  the per-simulation wall-clock limit was previously hardcoded to 10 s.
- QQ plot, ECDF + Spec Limits, and Yield vs Spec Curve now render like the
  histogram: a measurement selector plots the single chosen output full-size,
  with an "All measurements" checkbox restoring the previous panel grid.
  Plot plugins can opt into the selector via `supports_param = True`
  (see PLUGINS.md → PlotPlugin).
- Custom equations (scalar and transient) are stored **in the datasheet**
  (top-level `equations:` / `transient_equations:` YAML blocks) so they travel
  with the design. The CUSTOM EQUATIONS panel edits the active datasheet;
  equations still in `settings.json` are used as a fallback and migrated out
  on the panel's next Apply.
- Transient-equation results appear as selectable signals in the Multi-Plot
  Dashboard's Transient cells (the main Transient tab already listed them);
  signal lists now refresh immediately when equations change, preserving the
  current selection.
- `PluginContext.specs()` exposes the datasheet's `equations` /
  `transient_equations` blocks; `netlists()` resolves plugin-engine netlist
  extensions.
- Built-in distribution plot modes registered via the PlotPlugin interface
  (`chipify/plot_plugins/`): QQ plot (normality check), ECDF with spec
  limits, and yield-vs-spec curve. User plugins with the same name override
  the built-ins.
- `README.md` with project overview, prerequisites, install, and quick-start.
- Apache-2.0 `LICENSE`.
- `CONTRIBUTING.md` and GitHub issue/PR templates.
- GitHub Actions CI running `pytest` and `mypy` on push and pull request.
- Documented example datasheet under `examples/`.
- Configurable project folder paths (`in_dir`, `out_dir`, `work_dir`, `tb_dir`)
  via `settings.json`, falling back to the default structure when unset.

### Changed
- In `--batch` mode a datasheet's waveforms are written to
  `out/<datasheet>/analysis_data/`, next to its results, instead of the shared
  `out/analysis_data/`. Single-datasheet runs and the GUI are unchanged.
- Run sidecars (`history/run_*.meta.json`) store `analysis_dirs` / `tran_dir`
  relative to the output folder, so a moved or copied `out/` keeps its
  waveforms. Older sidecars with absolute paths still load.
- `pypdf` is now a dependency (it merges the batch report).
- Persisted netlist templates (netlist export and `--templates-dir`) now carry
  absolute host paths for any include that was written relatively. They are
  project- and machine-specific; regenerate them after moving a project or
  upgrading past this version.
- The `work_dir` project folder default was renamed `tmp/` → `work/`: it never
  held temporary data (the RAM-backed scratch dir does) — it is the input
  folder for `*.lib`/`*.mod`/`*.inc` model files staged next to the netlists.
  An explicit `work_dir` in `settings.json` keeps working unchanged.

### Fixed
- VACASK: a declared `transient_signals` / `dc_signals` / `ac_signals` entry
  that was missing from the `.raw` output was written as a column of zeros,
  so the run kept a flat, apparently valid waveform. The capture now fails,
  and its note names the missing signal(s) and the signals the `.raw` does
  contain.
- A declared waveform that was not captured (ngspice or VACASK) was only
  noted in the `<tb>__<kind>_capture` column: `sim_error` stayed `None` and
  the run passed with its curves missing. It now fails that testbench's row
  with `CAPTURE_ERROR`, so the run counts as a crash, no longer passes, and
  the testbench's measurements report ERROR for it.
- A worker process that died abruptly (out-of-memory kill, crash) stalled the
  sweep for good: the pool replaced the worker but never resolved the case it
  was running, so progress stopped one short of the total and only Stop ended
  the run. The lost case is now recorded as `WORKER_LOST` with the worker's
  exit code or signal, and the sweep finishes.
- Two testbenches of one datasheet that declared the same measurement name
  shared one result column: the later testbench overwrote the earlier one's
  value, while the error scope still pointed at the first. A measurement named
  like an input parameter, or any name equal to one of chipify's own result
  columns (`run_id`, `sim_error`, `<measurement>_pass`, ...), overwrote that
  column the same way. Such a datasheet is now rejected with an error naming
  both places.
- A custom equation named like a measurement, an input or a bookkeeping
  column replaced that column's values, while the measurement kept its pass
  flags from the simulation: an out-of-spec replacement still showed PASS and
  100 % yield. Equations can now only add columns. Such an equation (or a
  second one with the same name) is skipped with a ✗ line saying the name
  already exists.
- Removing a custom equation, or changing its expression to one that fails,
  left its old column in the loaded results, still offered by every plot.
  Equations are now re-applied to the results as simulated.
- A run loaded from history (GUI) or by `--report-only` showed the
  datasheet's current limits next to the pass/fail flags saved with the run:
  after tightening a limit, a value outside it still read PASS with zero
  failures. Reloaded runs are now graded against the limits they are shown
  with.
- `chipify-cli` rewrote `simulation_results.csv` without its
  `.meta.json` sidecar, so a sidecar left by an earlier GUI run stayed next
  to the new results: loading them restored that run's datasheet name and
  waveform directories. The CLI now writes the sidecar with the CSV.
- A run whose recorded waveform directory no longer exists (or that never
  captured an analysis kind) silently showed the waveforms of the newest run
  instead. Such a run now has no waveform data; only runs that recorded no
  waveform directories at all (older results) still fall back to the newest.
- Selecting another datasheet in the GUI kept the previous datasheet's run
  loaded, so every tab showed it under the new name, and *Generate Reports*
  then combined the new datasheet's path with the old run's tests. Selecting a
  datasheet now loads its newest run, or shows no results when it has none,
  and reports always name the datasheet the loaded run was parsed from.
- A datasheet block of the wrong type was skipped without a word: a list
  under `tests:` became a plan with no testbenches, which ran every case
  error-free and reported 100 % yield, and limits written as a list
  (`gain: [9, 11]`) meant no limits at all. Wrong block types are now schema
  errors, and a simulation refuses to start when the datasheet has no
  testbench, or a testbench declares no measurement, `measure:` or signals.
  Empty blocks (`tests:` with nothing under it) still load, so a datasheet
  can be built up step by step.
- The Fail Breakdown chart (GUI and report) said "100% Yield!" for a run in
  which every testbench crashed, because crashed runs are not counted as spec
  failures. It now says that no run produced a usable value, and when some
  runs errored it says how many instead of claiming full yield.
- `chipify-cli` (including `--report-only`) never applied the datasheet's
  scalar `equations:`, so a report plot of a derived quantity was skipped as
  a missing measurement while the GUI drew it. The CLI now applies them
  before the summary and the reports, and prints any equation it had to
  skip. Saved result CSVs still hold only the simulated columns.
- Two report plots that resolved to the same file name (two histograms of one
  measurement with different bins, a repeated `name:`, or names differing
  only in case) were reported as two files while the second overwrote the
  first. The later plot is now saved with a `_2` suffix and a warning.
- An image exporter that raised (for example a user exporter plugin) aborted
  the whole report run, so later plots and the PDF / Markdown report were
  never written. The failure is now a warning for that one file.
- `chipify-cli --batch` (with or without `--report-only`) stopped at the first
  datasheet that was not valid YAML or failed the schema: later datasheets
  never ran and no batch summary or combined report was written. That
  datasheet is now recorded as failed with its error, and the batch goes on.
- `range(...)` parameters were turned into floats, so `seed: range(200)`
  rendered `.option SEED = 3.0` and the results stored `seed = 3.0`. `range`
  values are now integers. (ngspice still rejects a seed of `0` in any form
  and then seeds from the clock, so start seed sweeps at 1, e.g.
  `range(1, 201)`.) A fractional argument such as `range(2.5)` was silently
  truncated and is now a schema error pointing at `linspace()`.
- A testbench whose schematic used a symbol xschem could not find was
  simulated anyway: xschem leaves the instance out (`* x1 - <symbol> IS
  MISSING !!!!`) and exits as it does for a clean schematic. The testbench
  now fails before simulation with an error naming the instance and symbol.
  (VACASK's `--spectre` netlists drop such an instance without any marker,
  so this check covers the ngspice path only.)
- ngspice capture injection treated any text `.control` / `.endc` as the
  control block, including a comment such as `* this .control block ...`:
  the `MY_DATA` echo and waveform capture then landed at the comment, giving
  `NO_MY_DATA_FOUND` or missing waveforms, and `set num_threads=1` was
  inserted after every mention. Only real `.control`, `.endc` and
  `quit`/`exit` lines count now (also behind the `* {{ NL }}` prefix).
  Netlists of benches without such comments are unchanged.
- An ngspice crash was reported as the last five lines of its log: an
  out-of-memory kill read as a progress line, and a fatal error near the top
  (a missing library) was cut off. `CRASH` now says how ngspice ended (exit
  code, or the killing signal, with SIGKILL flagged as a likely
  out-of-memory kill) and quotes the log's first error line. Runs whose
  values are missing (`INVALID_OUTPUT(... got 0)`, `NO_MY_DATA_FOUND`, as in
  a shell without the PDK setup) now carry the simulator's first error line
  too, and `CAPTURE_ERROR` rows are classified as such in the error table.
- A datasheet none of whose testbenches could be netlisted (for example all
  moved to another folder) still dispatched every case, each one failing:
  9000 runs, 9000 crashes. The simulation now stops before dispatch with the
  netlist errors. A missing schematic is named as such instead of "xschem
  wrote no netlist". A datasheet where only some testbenches fail still runs
  the others.
- Run sidecars recorded the ngspice version as `******`, the first line of
  the `ngspice --version` banner. They now record `ngspice-<n>`.
- Every click in a Transient, DC sweep or Bode plot made matplotlib test
  each curve for a pick nothing listened to: up to 238 ms per click with 500
  overlaid runs. The overlay curves no longer register for picking.
- Hovering a dense waveform overlay could name the wrong run: past 120,000
  samples per plot the hover index kept only every n-th sample, and the thinned
  curves cut corners (at 500 runs, 476 of 600 cursors placed on a curve got
  another run's tooltip). The index now keeps every sample and checks only
  the curve sections near the cursor, so the hover costs the same per mouse
  move. Building it, once per redraw, takes longer (about 0.4 s instead of
  0.1 s at 500 runs of 5000 samples on the development machine).
- Hover tooltips (scatter and curves) could be drawn partly off the plot: on
  HiDPI screens the placement compared logical canvas size with physical
  mouse positions, and a tooltip too tall for the room above the point was
  flipped below without checking the room there. A tooltip now goes on the
  side where it fits, or is kept inside the canvas when neither side does.
- The waveform hover took the drawing done by a figure export (a PNG at export
  resolution, or an SVG) as its saved background, so if the plot was not
  redrawn afterwards the next hover could restore a picture of the wrong
  size. Export drawings are now ignored. (The mechanism is covered by a test;
  a visible failure in the GUI was not reproduced.)
- Killing `chipify-cli` (or stopping the container it runs in) left its
  worker processes running: each finished its case and went on to the cases
  already queued for it, starting one simulator after another. A worker now
  treats the death of its chipify process like the Stop button: it kills the
  running simulator and starts no further case.
- Runs made with `chipify-cli --batch` never appeared in the GUI's history
  list, which only looked in `out/history`, not in the batch folders
  `out/<stem>/history`. They are now listed with the selected datasheet's
  other runs, as `<stem>/run_<timestamp>.csv`.
- `chipify-cli` could hang after writing a report's PNGs and before the PDF
  existed when `DISPLAY`/`WAYLAND_DISPLAY` was set and PySide6 installed (e.g.
  in IIC-OSIC-TOOLS): matplotlib picked the Qt backend. The CLI now pins Agg.
- Batch datasheets shared one waveform folder per second and restarted their
  run ids at `000000`, so two datasheets on the same testbench could overwrite
  or read each other's waveforms without any error.
- A history run loaded in the GUI (or for a report) showed the *newest* run's
  waveforms: nothing restored the run's own waveform directories from its
  sidecar. `data_loader.load_csv` now does.
- A testbench referencing a macro or model file by a **relative** path failed
  while the same path written absolutely worked. Netlists are re-written into
  the RAM scratch dir before they run (ngspice executes from `FAST_TMP`, vacask
  from a per-worker subdir of it), so a relative `.include` resolved against
  the scratch dir instead of `tb/`. Include-family directives (`.include`,
  `.inc`, `.lib`, and `include`/`ahdl_include`/`load` for vacask) are now
  resolved to absolute paths at netlist-generation time, against the directory
  of the testbench itself. Bare filenames are unchanged — they still come from
  `work/` staging — and a path that doesn't exist next to the testbench now
  fails that testbench with an error naming the path and the directory
  searched, instead of a bare simulator crash.
- Correlation matrix: `run_id` (an index, not data) and the per-run duration
  bookkeeping column no longer appear as correlated parameters (GUI and PDF
  report), and the axis labels are anchored so long names stay visible.
- The correlation matrix in the PDF report now greys out the self-correlation
  diagonal like the GUI, instead of rendering it as deep-red 1.00 cells.
- Multi-Plot Dashboard: a dashboard opened before the first data load lost its
  saved cell selections (and persisted the degraded config on close); saved
  selections are now restored when the data arrives.
- Multi-Plot Dashboard: cells in the right-hand grid columns were cut off at
  the window edge. The per-cell controls row forced an ~830 px minimum cell
  width; controls now wrap onto two rows, combos may shrink (popups still
  show full text), and grid columns share the viewport width evenly.
- Histogram selectors (main tab and dashboard cells) no longer offer input
  parameters as "measurements" — a distribution of an input is just the sweep
  grid. Inputs remain available for grouping and as scatter X/Y axes.
- A Jinja2 template-rendering error (e.g. a parameter-name typo in a
  testbench) now fails only that testbench's row with
  `TEMPLATE_RENDER_ERROR`; previously the exception silently discarded the
  whole worker batch of cases from the results. Engines that raise from
  `run()` are contained the same way (`ENGINE_ERROR`).
- AC waveform extraction from VACASK `.raw` files no longer crashes when the
  parsed bucket lacks the X-axis sentinel (`or`-chaining evaluated numpy
  arrays whose truth value is ambiguous).
- Analysis-capture failure notes now quote the *actual* engine's name and
  log tail; on VACASK runs they previously quoted a stale ngspice log.
- A non-numeric `MY_DATA` token now records NaN plus a failed flag for that
  measurement instead of silently omitting the column.
- An unknown engine name (e.g. a typo'd `simulator_engine` in settings.json)
  now logs a warning before falling back to ngspice.
- Stale *copies* of staged VACASK `.osdi` files are refreshed when the PDK
  changes (symlinked ones always tracked the source already).
- Histogram TeX export: measurement names with SPICE syntax (`v(out)`, …) are
  sanitised for the output filenames, the success dialog shows the paths that
  were actually written, and an all-NaN series reports "nothing exported"
  instead of a false success message.

### Removed
- SSH/remote-dispatch feature (not ready for release); preserved on the
  `feature/remote-dispatch` branch for future work.

## [0.2.0]

- Post-refactor baseline: multiprocessing simulation engine, CustomTkinter GUI,
  sandboxed expression evaluation, plugin system, and PDF/Markdown/LaTeX reports.

## [0.2.1]

- improve stability, fix bugs

## [0.2.2]

- switch to PySide6-Essentials

## [0.2.3]

- Errors are now scoped per testbench (`<tb_path>__error`) instead of a single
  per-row `sim_error`. A testbench that fails no longer hides the results of the
  testbenches that succeeded, and a measurement with no usable run reports
  ERROR rather than a vacuously-true PASS.
- Measurements tab: Errors column, amber ERROR status, and a SIMULATION ERRORS
  section naming the testbench, failing corner and simulator message, plus a
  scrollable log panel and an "Open Log" button.
- The CLI analyzer and the Markdown/PDF reports read the shared measurements
  service, so all four surfaces agree on a verdict and report errors.
- Crashed worker batches are recorded as WORKER_LOST rows instead of silently
  vanishing from the results; `run_sim` raises on unexpected failure and returns
  None only on user abort, so a failed run can no longer end in silence.
- Logging is enabled for CLI runs and the log banner records the running
  chipify version and install path.
- Waveform overlays (Transient / DC sweep / Bode) can be grouped by a swept
  input parameter: curves are coloured by the parameter's value (`temp=-40`,
  `temp=27`, …) with the signal carried by line style. Available in the Plots
  tab and in the Multi-plot dashboard.
- Fixed: waveform overlays matched no files for any run loaded from a results
  CSV. `run_id` parses back as an integer, so the lookup asked for `run_4`
  while the file is `run_000004__<tb>.csv`; only a run still in memory from a
  live simulation ever drew.
- New optional `reports:` block in the datasheet declares figures and reports to
  generate: plot type, axes/signals, grouping and output formats (any
  `ExporterPlugin` extension such as `png`/`svg`, plus `latex` where a pgfplots
  generator exists). Produced by `chipify-cli --reports` or the GUI's
  *Generate Reports* button into `out/reports/<timestamp>/`, with a `.latest`
  pointer. Previously a CLI run produced no figures at all.
- Fixed: the PDF report only ever contained histograms. A configured scatter,
  transient, DC sweep, Bode, correlation or tornado plot was written as an
  image but never reached the report, which instead showed an automatic
  histogram per measurement plus a correlation page nobody asked for. When a
  datasheet declares `reports: plots:`, the report's figure pages are now
  exactly those plots, in order, rendered through the same call that writes the
  standalone image. Datasheets without a `plots:` list keep the automatic
  sections.
- Fixed: the page-header banner was never visible. `axis("off")` also skips an
  axes' background patch, so every report page drew its white banner text on
  white — "Measurements (1 / 1)", "Histograms (1 / 4)" and the rest were in the
  PDF's text layer but invisible on the page.
- Fixed: figures embedded in the PDF kept the dark on-screen palette, while the
  same plot exported as a PNG came out light — the report path bypassed the
  exporters' white-paper re-skin.
- Fixed: custom equations could not be deleted. Removing a row without then
  pressing Apply was silently undone by the next table reload (a mode switch or
  datasheet reload put the equation straight back), and clicking Remove with no
  row selected did nothing and said nothing. Removing now saves immediately,
  falls back to the focused row, reports when there is nothing to remove, and an
  unsaved edit discarded by a reload is announced instead of vanishing.
- Fixed: a measurement rendered differently in different formats. The PDF report
  had its own histogram implementation that ignored the datasheet's `reports:`
  spec, so a plot configured `group: temp` came out grouped as a PNG and
  ungrouped in the PDF. Both now use one renderer and one set of options.
- Report histograms zoom to their data by default. With spec limits far outside
  the spread the bars previously collapsed into slivers against a spec-width
  axis; `zoom: false` restores the wide view.
- The `reports:` block is now editable in the GUI: **Reports…** in the Datasheet
  Editor opens a dialog for the plots, formats and documents, saved into the
  datasheet through the same writer the equations panel uses.
- `Export PDF Report` and `Generate Reports` are unified into one button and one
  code path. PDF output moves from the `out/reports/` root into the run's
  `out/reports/<timestamp>/` folder with everything else; when a datasheet
  declares no `reports:` block the button still offers a one-click PDF.
- Markdown and PDF reports now use the same engineering-unit formatting
  (`373.5 m`, `2.686 G`); they previously disagreed on the same numbers.
  Dimensionless values such as Cpk keep the plain form.
- Internal: run selection for waveform overlays lives once in
  `transient_loader.select_run_ids` instead of three copies with two different
  vocabularies, and a datasheet's `runs:` key now takes the same tokens as the
  GUI (`all_valid`, `failing`, `first:N`).
- The `Transient` tab is now `Plots`, since it has long covered DC sweep and
  Bode as well; the dashboard's waveform cell gained the same analysis-kind
  selector. Dashboards saved with the old `Transient` cell still open.
