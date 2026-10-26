# Copyright (c) 2026 Santiago Hofwimmer
"""``--preflight`` and ``--case``: check a datasheet, or run one case, before a sweep.

A datasheet whose testbenches had moved ran 9000 cases into 9000 crashes, and
debugging a testbench meant rebuilding the rendered deck by hand: a sweep keeps
it only in a scratch file that the next case overwrites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

simulator = pytest.importorskip("chipify.simulator")

from chipify import app_config, cli, engines, settings  # noqa: E402
from chipify.util import Stimuli  # noqa: E402


class _EchoEngine(simulator.BaseSimulator):
    """Doubles ``temp``; its testbench name says what goes wrong."""
    name = "echo_case"

    def generate_test_template(self, test):
        if test.tb_path == "missing":
            raise FileNotFoundError("tb/missing.sch not found")
        if test.tb_path == "typo":
            return "* temp={{ temperatur }}\n"            # undefined placeholder
        return "* temp={{ temp }}\n"

    def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
        self._log = f"ran {netlist.strip()!r} within {timeout_sec}"
        return f"MY_DATA: {float(netlist.split('=')[1]) * 2}", None

    def run_log_tail(self, n_lines=25):
        return getattr(self, "_log", "")


_GOOD = """\
parameters:
  temp: [-40, 27, 100]
tests:
  tb:
    engine: echo_case
    v: {min: 0}
"""

_BROKEN = """\
parameters:
  temp: [27]
tests:
  missing:
    engine: echo_case
    v1: {min: 0}
  typo:
    engine: echo_case
    v2: {min: 0}
  tb:
    engine: echo_case
    v: {min: 0}
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    engines.register_engine(_EchoEngine)
    datasheets = tmp_path / "datasheets"
    datasheets.mkdir()
    (datasheets / "good.yaml").write_text(_GOOD, encoding="utf-8")
    (datasheets / "broken.yaml").write_text(_BROKEN, encoding="utf-8")
    monkeypatch.setattr(settings, "IN_DIR", datasheets)
    monkeypatch.setattr(settings, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(app_config, "setup_logging", lambda **_k: None)
    monkeypatch.setattr(simulator, "stage_files_to_ram", lambda _engines: None)
    yield {"root": tmp_path, "datasheets": datasheets, "out": tmp_path / "out"}
    engines.reload_engines()


def _cli(monkeypatch, capsys, *argv) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["chipify-cli", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code or 0
    return code, capsys.readouterr().out


# ── preflight ─────────────────────────────────────────────────────────────────

def test_a_ready_datasheet_passes_preflight(project):
    report = simulator.preflight(Stimuli(project["datasheets"] / "good.yaml"))
    assert report.ok
    assert (report.cases, report.sizes) == (3, {"temp": 3})
    assert report.testbenches == {"tb": {"engine": "echo_case", "error": ""}}


def test_preflight_names_each_testbench_that_would_fail(project):
    report = simulator.preflight(Stimuli(project["datasheets"] / "broken.yaml"))
    assert not report.ok
    errors = {tb: info["error"] for tb, info in report.testbenches.items()}
    assert "netlist generation failed" in errors["missing"]
    assert "TEMPLATE_RENDER_ERROR" in errors["typo"] and "temperatur" in errors["typo"]
    assert errors["tb"] == ""


def test_preflight_reports_a_plan_that_checks_nothing(project):
    stim = Stimuli()
    stim.params = {"temp": [27]}
    report = simulator.preflight(stim)
    assert not report.ok and "no testbenches" in report.plan_error


def test_the_cli_preflight_writes_nothing(project, monkeypatch, capsys):
    code, out = _cli(monkeypatch, capsys, "-c", "good.yaml", "--preflight", "--json")
    assert code == 0
    assert "3 case(s) (3 temp)" in out
    line = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    assert line["ok"] is True and line["cases"] == 3
    assert not (project["out"] / "history").exists()


def test_a_batch_preflight_fails_when_one_datasheet_would(project, monkeypatch, capsys):
    code, out = _cli(monkeypatch, capsys, "--batch", str(project["datasheets"]),
                     "--preflight")
    assert code == 1
    assert "[-] missing [echo_case]" in out and "[+] tb [echo_case]: ok" in out
    code, _ = _cli(monkeypatch, capsys, "--batch", str(project["datasheets"]),
                   "--preflight", "--only", "good")
    assert code == 0


# ── one case ──────────────────────────────────────────────────────────────────

def test_one_case_keeps_its_deck_and_log(project, tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "stale.log").write_text("from an earlier run", encoding="utf-8")
    row = simulator.run_case(Stimuli(project["datasheets"] / "good.yaml"), 1, keep,
                             overrides={"sim_timeout_sec": 77})
    assert row["run_id"] == "000001" and row["temp"] == 27
    assert row["v"] == 54.0 and row["sim_error"] == "None"
    assert (keep / "tb.spice").read_text(encoding="utf-8") == "* temp=27\n"
    assert (keep / "tb.log").read_text(encoding="utf-8") == "ran '* temp=27' within 77.0"
    assert not (keep / "stale.log").exists()


def test_a_case_that_does_not_exist_is_refused(project, tmp_path):
    with pytest.raises(ValueError, match="0 to 2"):
        simulator.run_case(Stimuli(project["datasheets"] / "good.yaml"), 3, tmp_path)


def test_the_cli_case_writes_no_history(project, monkeypatch, capsys):
    code, out = _cli(monkeypatch, capsys, "-c", "good.yaml", "--case", "2")
    assert code == 0
    assert "Case 000002 of good.yaml: temp=100" in out and "v = 200  PASS" in out
    keep = project["out"] / "debug" / "good" / "case_000002"
    assert (keep / "tb.spice").read_text(encoding="utf-8") == "* temp=100\n"
    assert not (project["out"] / "history").exists()
    assert not (project["out"] / "simulation_results.csv").exists()


def test_a_batch_smoke_test_runs_one_case_per_datasheet(project, monkeypatch, capsys):
    code, out = _cli(monkeypatch, capsys, "--batch", str(project["datasheets"]),
                     "--case", "0", "--json")
    lines = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    assert code == 1                               # broken.yaml's testbenches fail
    assert [(l["yaml"], l["ok"]) for l in lines] == [("broken.yaml", False),
                                                     ("good.yaml", True)]
    assert (project["out"] / "debug" / "good" / "case_000000" / "tb.spice").is_file()


@pytest.mark.parametrize("argv", [
    ("--case", "0", "--reports"), ("--case", "0", "--resume"),
    ("--preflight", "--report-only"), ("--preflight", "--case", "0"),
    ("--case", "-1"),
])
def test_flags_that_do_not_combine(project, monkeypatch, capsys, argv):
    code, _ = _cli(monkeypatch, capsys, "-c", "good.yaml", *argv)
    assert code == 2
