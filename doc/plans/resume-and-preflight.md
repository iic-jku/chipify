# Plan: run settings, run records, resumable runs and single-case debugging

Status: implemented on branch `feature/resume-and-preflight` (`9c77734` to `d75ef8d`), based
on `fix/silent-failures-and-robustness` at `be9ea0b`. Line numbers refer to that commit.
Where the implementation differs from this plan, the plan has been corrected; the user-facing
behaviour is in README.md and CHANGELOG.md, the constraints in context.md.

## Goal

Long sweeps (hours, thousands of cases, laptop or compute server) must survive what ends them
early today: host standby, a container stop, an out-of-memory kill, a cancelled run. Today a
datasheet's results are written only after its last case (`simulator.run_sim` collects every row
in memory, lines 836-973, and returns `None` on abort, line 981), so one interrupted case loses
all the others. Around that, four smaller gaps make production runs awkward:

- settings can only be changed by editing the committed `settings.json` (no `--cores`,
  `--timeout`, `--out`), and a batch cannot be narrowed to some datasheets;
- a run's sidecar does not say which datasheet contents, chipify version or settings produced
  it, so history cannot be graded against the limits it ran with, and nothing can check that a
  resumed run still matches its datasheet;
- there is no way to check a datasheet before a sweep, or to run one case and keep its deck;
- the CLI and the GUI save a finished run through two different code paths
  (`cli._run_single` lines 130-160, `SimWorker._persist` lines 112-156), and they already
  differ: only the GUI archives the netlist templates.

Out of scope: comparing runs (a later step that builds on the run record), GUI controls for
resuming, rerunning only failed cases of a finished run.

## 1. One save path for finished runs (`chipify/run_store.py`)

New module, used by both `cli._run_single` and `SimWorker._persist`:

- `save_run(df, stim, *, out_dir, duration_s, yaml_name) -> SavedRun` writes
  `simulation_results.csv`, the `analysis_data/<kind>/.latest` pointers, the history copy
  `history/run_<ts>.csv`, both sidecars, and archives the netlist templates as
  `history/run_<ts>_templates/` (`netlist_export.persist_templates`, lines 56-75).
- Counts come from `data_loader.result_summary` in both callers. The GUI computed valid runs and
  yield by hand (lines 127-133); the numbers are the same, the code is not duplicated any more.
- Behaviour change: CLI runs now archive their templates too, so a CLI history run can export
  per-sample netlists and be re-run with `--templates-dir` like a GUI run.
- The GUI keeps adding its `simulation_duration_s_total` column before saving.

## 2. Effective run settings, CLI overrides, batch filter

Today `run_sim` reads the config once (line 806), but every case in every worker reads it again
for the timeout (`_simulate_single_case`, line 309), and the VACASK engine reads its binary,
netlist source and PDK directory the same way. An override held in the parent would not reach
the workers, and an edit of `settings.json` during a sweep changes the timeout halfway through.

- `app_config.RUN_KEYS`: the keys that shape results (`num_cores`, `simulator_engine`,
  `sim_timeout_sec`, `vacask_binary`, `vacask_netlist_source`, `ng2vc_binary`,
  `vacask_pdk_dir`, `process_start_method`, `chunk_size`).
- `run_sim(..., overrides=None)` takes the config once, applies the overrides, and keeps that
  snapshot as the run's settings. Each worker batch carries the snapshot, and
  `simulate_case_batch` pins it with `app_config.set_run_overrides()`, so every
  `load_config()` in a worker returns the run's values. The parent never pins: in the GUI a
  global pin would leak into the settings dialog and be saved back to `settings.json`.
- CLI flags: `--cores N`, `--timeout SEC` (both validated, > 0), `--out DIR` (output root for
  this invocation: results, history, waveforms, reports, the combined batch PDF and
  `chipify.log`), `--only PATTERN` (repeatable, `fnmatch` on the datasheet file name or stem;
  only with `--batch`; no match is an error that lists the datasheets).
- Precedence: CLI flag > `settings.json` > `project.yaml` defaults > built-in defaults.

## 3. Run record: what produced a run

`run_meta.write_meta` (lines 134-192) records the datasheet *name*, host, Python, simulator
versions and `git_commit`, which is `git rev-parse` in the working directory, i.e. the design
project's commit, not chipify's. Added, all optional for readers:

- `chipify` (version) and `chipify_path` (the package directory: tells a container's
  preinstalled chipify from a working tree);
- `datasheet_sha256` and `datasheet_snapshot`, a copy of the datasheet text as parsed
  (`history/run_<ts>.datasheet.yaml`). `Stimuli._load_from_yaml` (util.py line 55) keeps the
  text it parsed, so a datasheet edited during a long run is not mistaken for the one that ran;
- `settings`: the run settings of section 2; `engines`: `{testbench: engine}` as resolved.

`schema_version` stays 1: the fields are additive and every reader uses `.get()`.

## 4. Run journal and `--resume`

The run's on-disk record starts when the sweep starts, not when it ends.

- Run timestamp: one per run, taken in `run_sim` and used for the analysis directories
  (`_create_analysis_dirs`, line 705) and the history files (today the history copy is named
  after the end of the run, line 155 of `cli.py`, and the waveform folders after its start).
- Before the first case is dispatched, `run_sim` writes `history/run_<ts>.meta.json` with
  `"complete": false` and `planned_runs`, the datasheet snapshot, the templates archive and the
  run settings; then every finished worker batch is appended to `history/run_<ts>.rows.jsonl`
  (one line per batch, `{"t": <unix time>, "rows": [...]}`, flushed and fsynced; the time lets
  a resumed run's duration leave out the pause before the resume). Waveform CSVs are already
  written per case by the workers (`_persist_analyses`).
- `save_run` finishes the record: history CSV, `"complete": true`, latest CSV, pointers; then
  deletes the journal. A run that ends any other way leaves sidecar, snapshot, templates and
  journal behind. History listings only show `run_*.csv` (`data_loader.list_history_runs`,
  line 465), so an unfinished run never appears as a finished one.
- `--resume`: finds the datasheet's newest unfinished run, refuses when the datasheet text no
  longer matches its hash or the case count differs, and dispatches only the cases that have no
  row yet plus the rows lost with a worker (`WORKER_LOST`). It reuses the run's archived
  templates (same netlists as the first part), analysis directories, run timestamp and run
  settings (CLI overrides given now win), and appends to the same journal. Run ids stay
  stable because cases come from the same datasheet in the same order (`generate_cases`,
  lines 647-650).
- Batches: `--batch` keeps `batches/batch_<ts>.json` in the output root, listing the selected
  datasheets in order with each one's state and run timestamp. `--batch DIR --resume` continues
  the newest unfinished batch of `DIR`: finished datasheets are loaded from their recorded run
  (for the summary and the combined report), the interrupted one is resumed, the rest run.
- An abort prints how many cases finished and the command that continues the run.

## 5. Preflight and single-case runs

- `--preflight`: per datasheet (single or `--batch`, with `--only`), load and validate it, count
  the cases, resolve each testbench's engine, generate every netlist (missing symbols, missing
  testbenches and include paths surface here, through the existing checks), and render the
  first case of every testbench to catch undefined placeholders. Starts no pool and writes
  nothing to history. Exit code 1 when anything fails; `--json` prints one line per datasheet.
- `--case K` (index or run id): run that single case in-process, with the run settings, and
  keep everything under `<out>/debug/<stem>/case_<id>/`: each testbench's deck exactly as the
  simulator received it, its simulator log (`engine.run_log_tail`) and its captured waveforms.
  Prints the row (values, verdicts, errors). Writes no history. `--batch DIR --case 0` is the
  one-case-per-datasheet smoke test.
- No simulator commands are added: the testbench stays the source of truth.

## 6. Compatibility

- Results CSVs: unchanged columns. Sidecars: additive fields. New files: datasheet snapshots,
  journals (deleted on success), batch journals, CLI template archives.
- `run_sim` keeps its signature; `overrides` and the resume state are new keyword arguments.
  Plugin engines are untouched (the engine API does not change).
- History file names take the run's start time instead of its end time.

## 7. Tests

pytest without simulators, as `tests/test_cli.py` does (`run_sim` replaced) or with a fake
engine registered for the run (journal, resume, `--case`): overrides reach a worker case;
a mid-run `settings.json` edit does not; `--only` selection and errors; run record fields;
journal rows survive an abort and a killed worker; resume runs only the missing cases and
refuses a changed datasheet; batch resume skips finished datasheets; preflight reports a
missing testbench and an undefined placeholder; `--case` keeps the deck and writes no
history. Real-tool check in the container: interrupt a sweep, resume it, compare with an
uninterrupted run.

## 8. Docs

README (CLI flags, resume, preflight, smoke test), CHANGELOG `[Unreleased]`, context.md
(run lifecycle and record, run settings pinned in workers).

## 9. Order of work

1. `run_store.save_run` for CLI and GUI.
2. Run settings, `--cores/--timeout/--out/--only`.
3. Run record fields and datasheet snapshot.
4. Journal and `--resume` for single datasheets.
5. Batch journal and `--batch --resume`.
6. `--preflight` and `--case`.
