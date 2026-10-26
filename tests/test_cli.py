# Copyright (c) 2026 Santiago Hofwimmer
"""``chipify-cli --batch``: per-datasheet isolation, the combined report and
``--report-only``. ``simulator.run_sim`` is replaced by canned results, so no
simulator is needed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from chipify import cli

_DATASHEET = """
description: {description}
parameters:
  temp: [-40, 27, 100]
tests:
  tb:
    gain: {{min: 9, max: 11, unit: dB}}
    transient_signals: [outp]
reports:
  pdf: true
"""


def _results() -> pd.DataFrame:
    return pd.DataFrame({
        "temp": [-40, 27, 100],
        "run_id": ["000000", "000001", "000002"],
        "sim_error": ["None"] * 3,
        "tb__error": ["None"] * 3,
        "tb_overall_pass": [True] * 3,
        "gain": [10.0, 10.2, 9.8],
        "gain_pass": [True] * 3,
    })


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Two datasheets in tmp_path/datasheets, output in tmp_path/out."""
    from chipify import app_config, project_config, run_meta, settings, simulator

    batch = tmp_path / "datasheets"
    batch.mkdir()
    for name in ("alpha", "beta"):
        (batch / f"{name}.yaml").write_text(
            _DATASHEET.format(description=f"{name} checks"), encoding="utf-8")
    monkeypatch.setattr(settings, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(project_config, "PROJECT_FILE", tmp_path / "project.yaml")
    # Keep the real out/chipify.log and the version probes out of the test.
    monkeypatch.setattr(app_config, "setup_logging", lambda **_k: None)
    for probe in ("_ngspice_version", "_vacask_version", "_git_commit"):
        monkeypatch.setattr(run_meta, probe, lambda: "")

    calls = []

    def fake_run_sim(stim, *, out_dir=None, **_kw):
        calls.append(Path(out_dir))
        df = _results()
        df.attrs["analysis_dirs"] = simulator._create_analysis_dirs(stim, out_dir)
        return df

    monkeypatch.setattr(simulator, "run_sim", fake_run_sim)
    return {"root": tmp_path, "batch": batch, "out": tmp_path / "out",
            "calls": calls}


def _main(monkeypatch, capsys, *argv) -> tuple[int, list[dict]]:
    """Run the CLI; return its exit code and the JSON lines it printed."""
    monkeypatch.setattr(sys, "argv", ["chipify-cli", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code or 0
    lines = capsys.readouterr().out.splitlines()
    return code, [json.loads(l) for l in lines if l.startswith("{")]


def _outline_titles(pdf_path) -> list[str]:
    from pypdf import PdfReader
    return [item.title for item in PdfReader(str(pdf_path)).outline]


def test_batch_reports_write_one_combined_pdf(project, monkeypatch, capsys):
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--reports", "--json")
    assert code == 0
    batch = lines[-1]
    assert [s["yaml"] for s in batch["batch_summary"]] == ["alpha.yaml", "beta.yaml"]
    combined = Path(batch["combined_report"])
    assert combined.is_file()
    assert combined.parent.parent == project["out"] / "reports"
    assert _outline_titles(combined) == ["Summary", "alpha", "beta"]
    # Every datasheet still has its own report next to its results.
    for s in batch["batch_summary"]:
        assert Path(s["reports_dir"]).parent.parent.parent == project["out"]


def test_each_datasheet_keeps_its_own_waveforms(project, monkeypatch, capsys):
    """Run ids restart at 000000 per datasheet, so a shared waveform folder
    let two datasheets on one testbench read each other's curves."""
    from chipify import run_meta

    _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    assert project["calls"] == [project["out"] / "alpha", project["out"] / "beta"]
    for name in ("alpha", "beta"):
        root = project["out"] / name
        pointer = root / "analysis_data" / "transient" / ".latest"
        assert Path(pointer.read_text(encoding="utf-8")).parent.parent.parent == root
        hist = next((root / "history").glob("run_*.csv"))
        stored = run_meta.read_meta(hist)["analysis_dirs"]["transient"]
        assert stored.startswith("analysis_data/transient/")  # relative to root
    assert not (project["out"] / "analysis_data").exists()


def test_cli_runs_archive_their_netlist_templates(project, monkeypatch, capsys):
    """Only GUI runs kept the templates they simulated, so a CLI history run
    could not export a sample's netlist or be re-run with --templates-dir."""
    from chipify import run_meta, simulator

    def run_sim_with_templates(stim, *, out_dir=None, **_kw):
        for test in stim.tests:
            test.engine = "ngspice"
            test.template_str = f"* {test.tb_path} {{{{ temp }}}}\n"
        df = _results()
        df.attrs["analysis_dirs"] = simulator._create_analysis_dirs(stim, out_dir)
        return df

    monkeypatch.setattr(simulator, "run_sim", run_sim_with_templates)
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    assert code == 0
    for name in ("alpha", "beta"):
        hist = next((project["out"] / name / "history").glob("run_*.csv"))
        templates = Path(run_meta.read_meta(hist)["templates_dir"])
        assert (templates / "tb.spice").read_text(encoding="utf-8") == "* tb {{ temp }}\n"


def test_latest_results_get_their_own_sidecar(project, monkeypatch, capsys):
    """The CLI rewrote simulation_results.csv but left the sidecar of an
    earlier GUI run next to it, so the new results resolved to that run's
    datasheet and waveforms."""
    from chipify import data_loader as _dl
    from chipify import run_meta

    root = project["out"] / "alpha"
    root.mkdir(parents=True)
    (root / "simulation_results.meta.json").write_text(json.dumps({
        "schema_version": 1, "yaml": "other.yaml",
        "analysis_dirs": {"transient": "analysis_data/transient/19990101_000000"},
    }), encoding="utf-8")

    _main(monkeypatch, capsys, "--batch", str(project["batch"]))

    latest = root / "simulation_results.csv"
    hist = next((root / "history").glob("run_*.csv"))
    assert run_meta.read_meta(latest)["yaml"] == "alpha.yaml"
    assert (_dl.load_csv(str(latest)).attrs["analysis_dirs"]
            == _dl.load_csv(str(hist)).attrs["analysis_dirs"])


def test_a_failed_datasheet_fails_the_batch_but_stays_in_the_report(
        project, monkeypatch, capsys):
    from chipify import simulator

    real = simulator.run_sim

    def flaky(stim, **kw):
        if Path(kw["out_dir"]).name == "alpha":
            raise RuntimeError("netlist failed")
        return real(stim, **kw)

    monkeypatch.setattr(simulator, "run_sim", flaky)
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--reports", "--json")
    assert code == 1
    assert lines[-1]["batch_summary"][0] == {"yaml": "alpha.yaml", "error": "no data"}
    assert _outline_titles(lines[-1]["combined_report"]) == ["Summary", "beta"]


@pytest.mark.parametrize("mode", [[], ["--report-only"]])
@pytest.mark.parametrize("bad_text", [
    "tests: {tb: [unclosed",                       # not YAML
    "tests: [gain]",                               # YAML, but not a datasheet
])
def test_an_unreadable_datasheet_does_not_end_the_batch(
        project, monkeypatch, capsys, mode, bad_text):
    """Parsing ran outside the per-datasheet guard: the first broken YAML
    ended the batch, so beta never ran and no summary was printed."""
    if mode:
        _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    (project["batch"] / "alpha.yaml").write_text(bad_text, encoding="utf-8")
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--reports", *mode, "--json")
    assert code == 1
    alpha, beta = lines[-1]["batch_summary"]
    assert alpha["yaml"] == "alpha.yaml" and alpha["error"]
    assert beta["yaml"] == "beta.yaml" and beta["total"] == 3
    assert _outline_titles(lines[-1]["combined_report"]) == ["Summary", "beta"]


def test_report_only_rebuilds_from_finished_runs(project, monkeypatch, capsys):
    from chipify import simulator

    _main(monkeypatch, capsys, "--batch", str(project["batch"]))

    def must_not_run(*_a, **_k):
        raise AssertionError("--report-only must not simulate")

    monkeypatch.setattr(simulator, "run_sim", must_not_run)
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--report-only", "--json")
    assert code == 0
    per_datasheet, batch = lines[:-1], lines[-1]
    assert [s["total"] for s in per_datasheet] == [3, 3]
    assert all(Path(s["reports_dir"]).is_dir() for s in per_datasheet)
    assert _outline_titles(batch["combined_report"]) == ["Summary", "alpha", "beta"]


def test_report_only_grades_against_the_current_limits(project, monkeypatch, capsys):
    """The report prints the edited datasheet's limits, so the saved run's
    verdicts must follow them: 10.2 dB used to stay PASS under a 10.1 max."""
    from chipify import simulator

    _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    alpha = project["batch"] / "alpha.yaml"
    alpha.write_text(alpha.read_text(encoding="utf-8").replace("max: 11", "max: 10.1"),
                     encoding="utf-8")

    def must_not_run(*_a, **_k):
        raise AssertionError("--report-only must not simulate")

    monkeypatch.setattr(simulator, "run_sim", must_not_run)
    _code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                         "--report-only", "--json")
    passed = {s["yaml"]: s["passed"] for s in lines[:-1]}
    assert passed == {"alpha.yaml": 2, "beta.yaml": 3}


_EQUATION_DATASHEET = """
parameters:
  temp: [-40, 27, 100]
tests:
  tb:
    gain: {min: 9, max: 11, unit: dB}
equations:
  gain_x2: gain * 2
reports:
  formats: [png]
  plots:
    - {type: histogram, param: gain_x2, name: derived}
"""


@pytest.mark.parametrize("mode", [[], ["--report-only"]])
def test_reports_can_plot_datasheet_equations(project, monkeypatch, capsys, mode):
    """Only the GUI applied scalar equations, so a CLI report skipped the
    histogram of gain_x2 as a missing measurement."""
    for name in ("alpha", "beta"):
        (project["batch"] / f"{name}.yaml").write_text(_EQUATION_DATASHEET,
                                                       encoding="utf-8")
    if mode:
        _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    _code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                         "--reports", *mode, "--json")
    for summary in lines[:-1]:
        assert (Path(summary["reports_dir"]) / "derived.png").is_file()
    # Saved results stay as simulated: the derived column is not persisted.
    latest = project["out"] / "alpha" / "simulation_results.csv"
    assert "gain_x2" not in pd.read_csv(latest).columns


def test_report_only_without_a_finished_run_fails(project, monkeypatch, capsys):
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--report-only", "--json")
    assert code == 1
    assert lines[-1]["batch_summary"][0]["error"] == "no finished run"


def test_datasheet_order_from_project_yaml(project, monkeypatch, capsys):
    (project["root"] / "project.yaml").write_text(
        "name: bias chain\ndatasheet_order: [beta.yaml]\n", encoding="utf-8")
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--reports", "--json")
    assert code == 0
    assert [s["yaml"] for s in lines[-1]["batch_summary"]] == ["beta.yaml", "alpha.yaml"]
    assert _outline_titles(lines[-1]["combined_report"]) == ["Summary", "beta", "alpha"]


def _recording_run_sim(project, monkeypatch):
    """Replace run_sim; return the list of the overrides each call received."""
    from chipify import simulator

    seen = []

    def run_sim(stim, *, out_dir=None, overrides=None, **_kw):
        project["calls"].append(Path(out_dir))
        seen.append(overrides)
        df = _results()
        df.attrs["analysis_dirs"] = simulator._create_analysis_dirs(stim, out_dir)
        return df

    monkeypatch.setattr(simulator, "run_sim", run_sim)
    return seen


def test_cores_and_timeout_override_the_settings_for_one_run(project, monkeypatch, capsys):
    seen = _recording_run_sim(project, monkeypatch)
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                    "--cores", "3", "--timeout", "600")
    assert code == 0
    assert seen == [{"num_cores": 3, "sim_timeout_sec": 600.0}] * 2


def test_without_overrides_the_settings_apply(project, monkeypatch, capsys):
    seen = _recording_run_sim(project, monkeypatch)
    _main(monkeypatch, capsys, "--batch", str(project["batch"]))
    assert seen == [{"num_cores": None, "sim_timeout_sec": None}] * 2


@pytest.mark.parametrize("argv", [
    ("--cores", "0"), ("--cores", "two"), ("--cores", "2.5"),
    ("--timeout", "0"), ("--timeout", "-5"), ("--timeout", "nan"), ("--timeout", "inf"),
])
def test_invalid_overrides_are_rejected_before_anything_runs(
        project, monkeypatch, capsys, argv):
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]), *argv)
    assert code == 2
    assert project["calls"] == []


def test_only_runs_the_selected_datasheets(project, monkeypatch, capsys):
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--only", "al*", "--reports", "--json")
    assert code == 0
    assert project["calls"] == [project["out"] / "alpha"]
    assert [s["yaml"] for s in lines[-1]["batch_summary"]] == ["alpha.yaml"]
    assert _outline_titles(lines[-1]["combined_report"]) == ["Summary", "alpha"]


def test_only_takes_several_patterns_and_keeps_the_batch_order(project, monkeypatch, capsys):
    (project["batch"] / "gamma.yaml").write_text(
        _DATASHEET.format(description="gamma checks"), encoding="utf-8")
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                    "--only", "gamma.yaml", "--only", "alpha")
    assert code == 0
    assert project["calls"] == [project["out"] / "alpha", project["out"] / "gamma"]


def test_only_without_a_match_names_the_datasheets(project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["chipify-cli", "--batch", str(project["batch"]),
                                      "--only", "Alpha"])  # case-sensitive
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "there are: alpha, beta" in capsys.readouterr().out
    assert project["calls"] == []


def test_only_needs_batch(project, monkeypatch, capsys):
    code, _ = _main(monkeypatch, capsys, "--only", "alpha")
    assert code == 2


def test_out_moves_everything_a_batch_writes(project, monkeypatch, capsys):
    elsewhere = (project["root"] / "elsewhere").resolve()
    code, lines = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                        "--out", str(elsewhere), "--reports", "--json")
    assert code == 0
    assert project["calls"] == [elsewhere / "alpha", elsewhere / "beta"]
    assert Path(lines[-1]["combined_report"]).parent.parent == elsewhere / "reports"
    assert (elsewhere / "alpha" / "history").is_dir()
    assert not (project["out"] / "alpha").exists()

    # --report-only finds those runs only where they were written.
    from chipify import simulator

    def must_not_run(*_a, **_k):
        raise AssertionError("--report-only must not simulate")

    monkeypatch.setattr(simulator, "run_sim", must_not_run)
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]),
                    "--report-only", "--out", str(elsewhere))
    assert code == 0
    code, _ = _main(monkeypatch, capsys, "--batch", str(project["batch"]), "--report-only")
    assert code == 1


def test_out_applies_to_a_single_run(project, monkeypatch, capsys):
    from chipify import settings

    monkeypatch.setattr(settings, "IN_DIR", project["batch"])
    elsewhere = (project["root"] / "elsewhere").resolve()
    code, _ = _main(monkeypatch, capsys, "-c", "alpha.yaml", "--out", str(elsewhere))
    assert code == 0
    assert project["calls"] == [elsewhere]
    assert (elsewhere / "simulation_results.csv").is_file()


def test_order_datasheets_accepts_names_and_stems(tmp_path, caplog):
    from chipify import project_config

    paths = [tmp_path / n for n in ("c.yaml", "a.yaml", "b.yaml")]
    ordered = project_config.order_datasheets(paths, ["b", "c.yaml", "zzz.yaml"])
    assert [p.name for p in ordered] == ["b.yaml", "c.yaml", "a.yaml"]
    assert "zzz.yaml" in caplog.text


def test_cli_pins_the_agg_backend():
    """With DISPLAY set and PySide6 installed matplotlib picks QtAgg, and the
    CLI hung in Qt before the PDF was written. A fresh interpreter, because
    other tests have already selected Agg in this one."""
    code = textwrap.dedent("""
        import sys, matplotlib
        from chipify import cli
        sys.argv = ["chipify-cli", "--version"]
        try:
            cli.main()
        except SystemExit:
            pass
        print("BACKEND=" + matplotlib.get_backend())
    """)
    env = dict(os.environ, MPLBACKEND="QtAgg")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=120,
                         cwd=Path(__file__).resolve().parent.parent)
    assert "backend=agg" in out.stdout.lower(), out.stdout + out.stderr
