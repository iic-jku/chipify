# Plan: one combined PDF report for `--batch`

Status: proposal, nothing implemented. Written against chipify 0.2.3 (commit 939cd35).
Line numbers refer to that commit.

## Goal

`chipify-cli --batch DIR --reports` already runs every datasheet in `DIR` and writes each one's
report. In addition it should write **one combined PDF**: a summary page with one row per
datasheet, then every datasheet's report pages, with page numbers and (optionally) bookmarks.
The same combined PDF should be rebuildable from finished runs without simulating again.

A prototype of the summary page was built outside chipify (matplotlib page + ghostscript merge
of the per-datasheet PDFs) for four macros of a bias chain (11 + 5 + 5 + 3 datasheets, about
2900 runs). It is the reference for the layout in section 4.

## 1. CLI (`chipify/cli.py`)

Today (lines 241-291): `--batch DIR` globs `DIR/*.yaml` (DIR relative to the working
directory, `-c` ignored), runs each through `_run_single(..., out_dir=OUT_DIR/<stem>,
reports=args.reports)`, prints one JSON line per datasheet plus `{"batch_summary": [...]}`,
and exits 1 only when a run crashes or aborts.

Changes:

- `_run_single` (line 55) returns only the summary dict, so the loop loses each run's
  DataFrame and `Stimuli` (the Markdown branch re-reads the CSV to get them back). Make it
  return a small `RunEntry` (`yaml, stem, df, stim, duration_s, out_dir, reports_dir,
  meta_path, summary`), or return it alongside the summary.
- After the loop, when `--reports` is set, call the new
  `report_service.generate_batch_report(entries, OUT_DIR, title=...)` (section 3) and add
  `"combined_report": "<path>"` to the `batch_summary` JSON line.
- Datasheet order: sorted glob as today, overridden by `datasheet_order` from `project.yaml`
  when present (section 5).
- Optional `--strict`: exit 1 when any datasheet has a yield below 100 %.
- **Force the Agg backend at the top of `main()`** (line 132): `matplotlib.use("Agg")`.
  `pdf_export.py:26` and `plot_manager.py:8` import `pyplot` without choosing a backend. In a
  container with `DISPLAY`/`WAYLAND_DISPLAY` set and PySide6 installed, matplotlib picks
  QtAgg, `plt.figure` brings up Qt and the CLI hangs in `poll()` after the PNGs are written
  and before the PDF exists. Observed in the IIC-OSIC-TOOLS image. The tests already call
  `matplotlib.use("Agg")` (test_report_service.py:312, 352, 496), which is why CI never sees it.

## 2. Data isolation (needed before a report spans several datasheets)

Batch mode isolates `simulation_results.csv`, `history/` and `reports/` per datasheet, but not
the waveforms:

- `run_sim` writes waveforms to the shared `OUT_DIR/analysis_data/<kind>/<ts>`, and
  `write_analysis_pointers` (simulator.py:547-566) keeps one `.latest` per project.
- Run ids restart at `000000` for every datasheet (simulator.py:632) and the folder timestamp
  has one-second resolution, so two datasheets that start in the same second share a folder.
  Datasheets on the same testbench (`run_<id>__<tb>.csv`) can then overwrite or read each
  other's waveforms. A wrong match is silent.
- In batch mode `out_root` is `OUT_DIR/<stem>` (report_service.py:224) while the waveforms
  sit under `OUT_DIR/analysis_data`, so only the `df.attrs["analysis_dirs"]` lookup works.
  Rebuilding a report from a history CSV falls back to `.latest`, which belongs to whichever
  datasheet ran last.
- `report_service._render_waveform` calls `resolve_analysis_dir(df, out_root, kind)`
  (line 107) without `meta=`, and nothing restores `analysis_dirs` into `df.attrs` when a
  history CSV is loaded.
- The run meta stores `analysis_dirs` as absolute paths, which break when `out/` is moved or
  the project is copied.

Fixes: write waveforms under `OUT_DIR/<stem>/analysis_data/` in batch mode (or name the folder
`<stem>_<ts>`), keep `.latest` per output root, pass `meta=` in `_render_waveform`, restore
`df.attrs` when loading history, and store `analysis_dirs` relative to the output root.

## 3. PDF refactor (`chipify/pdf_export.py`, `chipify/report_service.py`)

- Split `generate_pdf_report` (lines 704-746) into
  `render_report_pages(pdf, df, stim, yaml_path, *, sim_duration_sec, reports_config, out_root)
  -> SectionInfo` that draws into an already open `PdfPages`, and a thin wrapper that opens the
  file inside `mpl.rc_context(_PDF_RCPARAMS)` as now. The page builders (`_add_cover`,
  `_add_table`, `_add_errors`, `_add_report_plots`, `_add_histograms`, `_add_correlation`,
  `_add_plugin_pages`) already take an open `pdf` and stay unchanged.
- New `pdf_export.generate_combined_pdf(entries, out_path, title, descriptions)`:
  1. dry pass with a counting stub for `savefig` (like `_Stub` in test_report_service.py:349)
     to get each section's page count, so the summary can print start pages without a new
     dependency;
  2. summary page(s), then each section via `render_report_pages`;
  3. page numbers through a small `PdfPages` proxy whose `savefig()` stamps a footer (this
     also covers plugin `render_pdf` hooks, which only call `savefig`);
  4. optional bookmarks: if `pypdf` is importable (CI already installs it), add one outline
     entry per datasheet in a post-pass; otherwise skip silently. `PdfPages` cannot write
     outlines itself.
- New `report_service.generate_batch_report(entries, out_dir, *, title=None)` owns
  `out_dir/reports/<ts>/batch_report_<ts>.pdf` and `reports/.latest`, the same way
  `generate_reports` does for one datasheet. It must not call `generate_reports`, which owns
  its own timestamped folder.

## 4. Summary page

- Header: title (from `project.yaml`, else the project folder name), generation time, chipify
  version, simulator, totals (runs, valid, passed, overall yield, total time).
- Table, one row per datasheet: datasheet, runs, crashes, valid, passed, yield (green >= 99 %,
  amber >= 80 %, red below, the cover's thresholds at `_add_cover` line 211), measurements
  PASS / FAIL / ERROR, duration, start page.
- Use the same numbers as the per-datasheet cover: `data_loader.result_summary(
  prepare_results(df))` and `measurements.measurement_rows(...)`. The `--json` line
  (`cli._json_summary`, lines 14-26) has no measurement counts, so compute them in process.
- Below the table: each datasheet's one-line description and the names of failing / errored
  measurements.
- Continue on further pages when the table does not fit.

## 5. Configuration

- Datasheet `description:` (string). `schema.validate_datasheet` (schema.py:363) drops
  unknown top-level keys today. Add the key to the schema and to `Stimuli`. Show it on the
  summary page and as a subtitle on the datasheet's own cover.
- `project.yaml` (`chipify/project_config.py`, `_ALLOWED_KEYS` already has `name` and
  `description`): add `report_title`, `datasheet_order` (list of file names) and
  `batch_report: {pdf: true, per_datasheet: true}`.
- Optional per datasheet: `reports: {include_auto: true}` to keep the automatic
  per-measurement histograms and the correlation page even when `plots:` is declared
  (pdf_export.py:734-743 drops them by design, locked in by test_report_service.py:429).

## 6. Output layout

```
out/
  <stem>/simulation_results.csv, history/, analysis_data/, reports/<ts>/report_<ts>.pdf
  reports/<ts>/batch_report_<ts>.pdf      # combined
  reports/.latest
```

## 7. Report from finished runs: `--batch DIR --report-only`

Rebuild every datasheet's report and the combined PDF from each datasheet's newest
`history/run_*.meta.json` and CSV, without simulating. The API already supports it:
`report_service.generate_reports(df, util.Stimuli(yaml), yaml, out_dir, duration_s=...)` with
`df.attrs["analysis_dirs"]` taken from the meta. The prototype used exactly this to relabel
plots across 24 datasheets without rerunning their simulations. It needs the section 2 fixes
(meta lookup, relative paths) to be reliable.

## 8. Tests (pytest, no simulations)

- `tests/test_report_service.py` (model: `test_documents_and_latest_pointer`, line 195):
  combined page count equals summary pages plus all sections; the start pages printed on the
  summary match the dry pass; summary numbers equal `result_summary`; a `description:` shows
  up; bookmarks exist when pypdf is installed.
- New `tests/test_cli.py`: `--batch --reports` with `simulator.run_sim` monkeypatched to
  return canned DataFrames (pattern of test_simulator_engine.py:195-203): JSON lines,
  `combined_report` path, exit codes, `--report-only`, and `matplotlib.get_backend()` is Agg
  after `main()`.
- Isolation test: two datasheets on the same testbench do not see each other's waveforms.

## 9. Docs and changelog

- README "Headless CLI" (lines 98-113): document the combined report under `--batch`,
  `--report-only`, and `--reports`, which is missing there today.
- `CHANGELOG.md` `[Unreleased]`: Added (combined batch report, `description:`,
  `--report-only`), Fixed (Agg backend, waveform isolation, relative analysis paths, `meta=`
  lookup). Version in `chipify/__init__.py:4`.

## 10. Small fixes worth bundling (verified in the code)

- `bins: auto` (lowercase) raises in `int("auto")` (plot_manager.py:230, report_service.py:165);
  compare case-insensitively.
- Units are parsed (schema.py:465) but never shown in the PDF or Markdown table; pass `r.unit`
  to `fmt_eng`.
- One NaN turns a whole measurement row into ERROR (measurements.py:212); report "n missing"
  and keep the statistics.
- The cover's "CPU Cores" comes from the current settings (pdf_export.py around line 182), not
  from the run; take it from the run meta.

## 11. Order of work

1. Agg backend and the data-isolation fixes (sections 1 and 2), as a bug-fix release on their own.
2. PDF refactor (section 3) with unchanged behaviour and green tests.
3. Summary page and configuration (sections 4 and 5).
4. Combined report in `--batch`, then `--report-only` (sections 1 and 7).
5. Tests and docs (sections 8 and 9).
6. GUI later: a "Run batch" action through a controller (CONTRIBUTING: tabs must not call the
   simulator directly) that reuses the batch loop and `generate_batch_report`.

## Open decisions

- Combined report only with `--batch --reports`, or always in batch mode?
- `pypdf` as an optional dependency for bookmarks, or no bookmarks?
- Descriptions in each datasheet, in `project.yaml`, or both?
- Waveforms per datasheet under `out/<stem>/analysis_data/` changes where single-datasheet GUI
  users find them; keep the shared folder outside batch mode?
