# Copyright (c) 2026 Santiago Hofwimmer
"""Interrupted runs keep what they simulated and can be continued.

Results used to be written only after the last case, so a host standby, a
container stop, an out-of-memory kill or a cancelled run lost every case of
the datasheet. A run's record now starts with the sweep (``run_store``), every
finished worker batch goes into its journal, and ``--resume`` simulates only
what is missing.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

simulator = pytest.importorskip("chipify.simulator")

from chipify import app_config, data_loader, engines, run_meta, run_store  # noqa: E402
from chipify.util import Stimuli  # noqa: E402

_WATCHDOG_SEC = 60


def _stim(source=b"parameters: {k: [0, 1, 2]}\n", yaml_path="sweep.yaml"):
    test = SimpleNamespace(tb_path="tb", engine="ngspice", template_str="* {{ k }}\n")
    return SimpleNamespace(tests=[test], source_bytes=source, yaml_path=yaml_path)


def _row(run_id, value=1.0, error="None"):
    return {"run_id": run_id, "k": int(run_id), "sim_error": error, "tb__error": error,
            "v": value, "v_pass": True, "tb_overall_pass": error == "None"}


def _begin(out, stim=None, planned=3):
    stim = stim or _stim()
    ts = run_store.new_run_ts(out)
    journal = run_store.begin_run(stim, out, ts, planned_runs=planned,
                                  analysis_dirs={}, run_settings={"sim_timeout_sec": 60.0,
                                                                  "num_cores": 8},
                                  engines={"tb": "ngspice"})
    return ts, journal


# ── The record of a run in progress ──────────────────────────────────────────

def test_a_run_in_progress_has_a_record_but_no_results_yet(tmp_path):
    ts, journal = _begin(tmp_path)
    journal.append([_row("000000"), _row("000001")])
    journal.close()

    meta = run_meta.read_meta(tmp_path / "history" / f"run_{ts}.csv")
    assert meta["complete"] is False and meta["planned_runs"] == 3
    assert meta["yaml"] == "sweep.yaml"
    assert (tmp_path / meta["datasheet_snapshot"]).read_bytes() == _stim().source_bytes
    assert (Path(meta["templates_dir"]) / "tb.spice").is_file()
    # Nothing lists it as a finished run.
    assert data_loader.list_history_runs(tmp_path, yaml_name="sweep.yaml") == []
    run = run_store.find_unfinished(tmp_path, "sweep.yaml")
    assert run.ts == ts and sorted(run.done_rows()) == ["000000", "000001"]


def test_the_journal_keeps_the_last_row_of_a_case_and_skips_a_torn_line(tmp_path):
    ts, journal = _begin(tmp_path)
    journal.append([_row("000000"), _row("000001", error="tb: WORKER_LOST: killed")])
    journal.append([_row("000001", value=2.0)])         # the lost case, rerun
    journal.append([_row("000002", error="tb: WORKER_LOST: killed")])
    journal.close()
    with open(tmp_path / "history" / f"run_{ts}.rows.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"t": 1, "rows": [{"run_id": "0000')    # killed mid-write

    run = run_store.find_unfinished(tmp_path, "sweep.yaml")
    assert run.rows()["000001"]["v"] == 2.0
    # A case lost with its worker never ran: it is simulated again.
    assert sorted(run.done_rows()) == ["000000", "000001"]


def test_numpy_values_keep_their_type_in_the_journal(tmp_path):
    import numpy as np

    ts, journal = _begin(tmp_path)
    journal.append([{"run_id": "000000", "v": np.float64(1.5), "v_pass": np.bool_(True),
                     "n": np.int64(3), "gone": float("nan")}])
    journal.close()
    row = run_store.find_unfinished(tmp_path, "sweep.yaml").rows()["000000"]
    assert (row["v"], row["v_pass"], row["n"]) == (1.5, True, 3)
    assert row["v_pass"] is True and row["gone"] != row["gone"]       # NaN


def test_only_the_newest_run_of_a_datasheet_is_offered(tmp_path):
    _begin(tmp_path)                                   # interrupted, then given up
    later = run_store.save_run(pd.DataFrame([_row("000000")]), _stim(),
                               out_dir=tmp_path, duration_s=1.0, yaml_name="sweep.yaml")
    assert later.history_csv.is_file()
    assert run_store.find_unfinished(tmp_path, "sweep.yaml") is None
    _begin(tmp_path, _stim(yaml_path="other.yaml"))
    assert run_store.find_unfinished(tmp_path, "sweep.yaml") is None
    assert run_store.find_unfinished(tmp_path, "other.yaml") is not None


def test_a_changed_datasheet_cannot_continue_the_run(tmp_path):
    _begin(tmp_path)
    run = run_store.find_unfinished(tmp_path, "sweep.yaml")
    assert run.problem_with(_stim()) == ""
    problem = run.problem_with(_stim(source=b"parameters: {k: [0, 1, 2, 3]}\n"))
    assert "changed" in problem and ".datasheet.yaml" in problem


def test_saving_completes_the_run_the_record_started(tmp_path):
    ts, journal = _begin(tmp_path)
    journal.close()
    df = pd.DataFrame([_row("000000"), _row("000001"), _row("000002")])
    df.attrs["run_ts"] = ts
    saved = run_store.save_run(df, _stim(), out_dir=tmp_path, duration_s=5.0,
                               yaml_name="sweep.yaml")

    assert saved.history_csv == tmp_path / "history" / f"run_{ts}.csv"
    meta = run_meta.read_meta(saved.history_csv)
    assert meta["complete"] is True and meta["planned_runs"] == 3
    assert (tmp_path / meta["datasheet_snapshot"]).is_file()
    assert not (tmp_path / "history" / f"run_{ts}.rows.jsonl").exists()
    assert run_store.find_unfinished(tmp_path, "sweep.yaml") is None
    assert data_loader.list_history_runs(tmp_path, yaml_name="sweep.yaml")[1:] == [
        f"run_{ts}.csv"]


def test_the_time_between_two_parts_does_not_count(tmp_path):
    ts, journal = _begin(tmp_path)
    journal.close()
    run = run_store.find_unfinished(tmp_path, "sweep.yaml")
    start = run_store._unix_time(run.meta["timestamp"])
    run.meta["resumed"] = [_iso(start + 3600)]            # resumed an hour later
    lines = [{"t": start + 10, "rows": [_row("000000")]},
             {"t": start + 20, "rows": [_row("000001")]},   # killed after this
             {"t": start + 3600 + 5, "rows": [_row("000002")]}]
    run.journal.write_text("".join(json.dumps(line) + "\n" for line in lines),
                           encoding="utf-8")
    assert run.elapsed_s() == pytest.approx(25.0)


def _iso(unix_time):
    import datetime
    return datetime.datetime.fromtimestamp(unix_time).isoformat(timespec="seconds")


def test_a_record_that_cannot_be_written_does_not_stop_the_run(tmp_path):
    (tmp_path / "history").write_text("not a folder", encoding="utf-8")
    journal = run_store.begin_run(_stim(), tmp_path, "20260101_000000", planned_runs=3,
                                  analysis_dirs={}, run_settings={}, engines={})
    journal.append([_row("000000")])                  # a no-op, not an error
    journal.close()


# ── A real sweep, interrupted and resumed ─────────────────────────────────────

_ENGINE_SRC = textwrap.dedent(
    """
    import os
    import time
    from pathlib import Path

    from chipify.engines import BaseSimulator
    from chipify.engines.abort import abort_simulation

    class Interruptible(BaseSimulator):
        name = "interruptible"

        def generate_test_template(self, test):
            return "{{ k }}"

        def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
            k = int(netlist.strip())
            box = Path(os.environ["CHIPIFY_TEST_BOX"])
            with open(box / "ran.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{k}\\n")
            marker = box / "aborted"
            if k == 3 and not marker.exists():
                time.sleep(0.5)      # the parent collects cases 0-2 meanwhile
                marker.write_text("1", encoding="utf-8")
                abort_simulation()   # like the Stop button or Ctrl-C
                time.sleep(1.0)      # ... seen before this case returns
            return f"MY_DATA: {k * 10}", None
    """
)

_DATASHEET = """\
parameters:
  k: [0, 1, 2, 3, 4, 5]
tests:
  tb:
    engine: interruptible
    v: {min: -1}
"""


def _ran(box) -> list[int]:
    path = box / "ran.txt"
    return [int(x) for x in path.read_text(encoding="utf-8").split()] if path.exists() else []


@pytest.mark.parametrize("start_method", ["spawn"] + (
    ["forkserver"] if sys.platform.startswith("linux") else []))
def test_an_interrupted_sweep_resumes_with_only_the_missing_cases(
        tmp_path, monkeypatch, start_method):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "interruptible.py").write_text(_ENGINE_SRC, encoding="utf-8")
    box = tmp_path / "box"
    box.mkdir()
    # Workers inherit the environment: the same engine, the same box.
    monkeypatch.setenv("CHIPIFY_PLUGINS", str(plugins))
    monkeypatch.setenv("CHIPIFY_TEST_BOX", str(box))
    engines.reload_engines()
    cfg = {"process_start_method": start_method, "num_cores": 1, "chunk_size": 1,
           "sim_timeout_sec": 30}
    monkeypatch.setattr(app_config, "load_config", lambda: dict(cfg))
    monkeypatch.setattr(simulator, "stage_files_to_ram", lambda _engines: None)
    datasheet = tmp_path / "sweep.yaml"
    datasheet.write_text(_DATASHEET, encoding="utf-8")
    out = tmp_path / "out"

    watchdog = threading.Timer(_WATCHDOG_SEC, simulator.abort_simulation)
    watchdog.start()
    try:
        assert simulator.run_sim(Stimuli(datasheet), out_dir=out) is None   # aborted
        assert _ran(box) == [0, 1, 2, 3]
        run = run_store.find_unfinished(out, "sweep.yaml")
        assert sorted(run.done_rows()) == ["000000", "000001", "000002"]

        stim = Stimuli(datasheet)
        assert run.problem_with(stim) == ""
        df = simulator.run_sim(stim, out_dir=out, resume=run)
    finally:
        watchdog.cancel()
        engines.reload_engines()

    assert _ran(box) == [0, 1, 2, 3, 3, 4, 5]          # 0-2 were not simulated again
    assert list(df["run_id"]) == [f"{i:06d}" for i in range(6)]
    assert list(df["v"]) == [k * 10.0 for k in range(6)]
    assert list(df["sim_error"]) == ["None"] * 6
    assert df.attrs["run_ts"] == run.ts

    saved = run_store.save_run(df, stim, out_dir=out, duration_s=1.0, yaml_name="sweep.yaml")
    assert saved.history_csv.name == f"run_{run.ts}.csv"
    meta = run_meta.read_meta(saved.history_csv)
    assert meta["complete"] is True and len(meta["resumed"]) == 1
    assert run_store.find_unfinished(out, "sweep.yaml") is None


def test_a_resume_refuses_another_case_count(tmp_path, monkeypatch):
    stim = Stimuli()
    stim.params = {"k": [0, 1]}
    stim.tests = [SimpleNamespace(tb_path="tb", value_lst=[object()], engine="ngspice")]
    run = run_store.UnfinishedRun(tmp_path, "20260101_000000", {"planned_runs": 3})
    with pytest.raises(ValueError, match="planned 3 cases"):
        simulator.run_sim(stim, out_dir=tmp_path, resume=run)
