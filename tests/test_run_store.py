# Copyright (c) 2026 Santiago Hofwimmer
"""Saving a finished run (``chipify.run_store``), shared by the CLI and the GUI."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from chipify import run_meta, run_store


def _results(analysis_dirs=None) -> pd.DataFrame:
    df = pd.DataFrame({
        "run_id": ["000000", "000001"],
        "sim_error": ["None", "tb: CRASH: ngspice exited with code 1: Error"],
        "tb__error": ["None", "tb: CRASH: ngspice exited with code 1: Error"],
        "gain": [10.0, float("nan")],
        "gain_pass": [True, False],
        "tb_overall_pass": [True, False],
    })
    if analysis_dirs:
        df.attrs["analysis_dirs"] = analysis_dirs
    return df


def _stim(template="* tb {{ temp }}\n"):
    return SimpleNamespace(tests=[
        SimpleNamespace(tb_path="tb", engine="ngspice", template_str=template)])


def test_a_saved_run_has_latest_history_templates_and_sidecars(tmp_path):
    wave = tmp_path / "analysis_data" / "transient" / "20260101_000000"
    wave.mkdir(parents=True)
    saved = run_store.save_run(_results({"transient": str(wave)}), _stim(),
                               out_dir=tmp_path, duration_s=12.5, yaml_name="amp.yaml")

    assert saved.latest_csv == tmp_path / "simulation_results.csv"
    assert saved.history_csv.parent == tmp_path / "history"
    assert (Path(saved.templates_dir) / "tb.spice").read_text(
        encoding="utf-8") == "* tb {{ temp }}\n"
    for csv in (saved.latest_csv, saved.history_csv):
        meta = run_meta.read_meta(csv)
        assert meta["yaml"] == "amp.yaml" and meta["duration_s"] == 12.5
        assert (meta["total_runs"], meta["valid_runs"], meta["global_yield"]) == (2, 1, 50.0)
        assert meta["templates_dir"] == saved.templates_dir
        assert meta["analysis_dirs"] == {
            "transient": "analysis_data/transient/20260101_000000"}
    assert (tmp_path / "analysis_data" / "transient" / ".latest").read_text(
        encoding="utf-8") == str(wave)


def test_a_failed_history_copy_still_leaves_the_latest_results(tmp_path):
    (tmp_path / "history").write_text("not a folder", encoding="utf-8")
    saved = run_store.save_run(_results(), _stim(), out_dir=tmp_path,
                               duration_s=1.0, yaml_name="amp.yaml")
    assert saved.history_csv is None and saved.history_error
    assert len(pd.read_csv(saved.latest_csv)) == 2
    assert run_meta.read_meta(saved.latest_csv)["yaml"] == "amp.yaml"


def test_a_run_records_what_produced_it(tmp_path):
    """The sidecar named the datasheet file but not its contents, chipify's
    version or the settings, so a run could not be traced to what made it."""
    import hashlib

    import chipify

    source = b"\xef\xbb\xbfparameters:\r\n  temp: [27]\r\n"   # BOM and CRLF kept
    stim = _stim()
    stim.source_bytes = source
    df = _results()
    df.attrs["run_settings"] = {"sim_timeout_sec": 600.0, "num_cores": 2}
    df.attrs["engines"] = {"tb": "ngspice"}
    saved = run_store.save_run(df, stim, out_dir=tmp_path, duration_s=1.0,
                               yaml_name="amp.yaml")

    for csv in (saved.latest_csv, saved.history_csv):
        meta = run_meta.read_meta(csv)
        assert meta["datasheet_sha256"] == hashlib.sha256(source).hexdigest()
        assert (tmp_path / meta["datasheet_snapshot"]).read_bytes() == source
        assert meta["settings"] == {"sim_timeout_sec": 600.0, "num_cores": 2}
        assert meta["engines"] == {"tb": "ngspice"}
        assert meta["chipify"] == chipify.__version__
        assert Path(meta["chipify_path"]) == Path(chipify.__file__).resolve().parent
    assert run_meta.read_meta(saved.history_csv)["datasheet_snapshot"] == (
        saved.history_csv.with_suffix(".datasheet.yaml").relative_to(tmp_path).as_posix())


def test_a_datasheet_keeps_the_bytes_it_was_parsed_from(tmp_path):
    """Read as bytes for the record; a BOM and CRLF line ends parse as before."""
    from chipify.util import Stimuli

    text = ("description: |\n  two\n  lines\nparameters:\n  corner:\n    - tt\n    - ss\n"
            "tests:\n  tb:\n    gain: {min: 9, max: 11}\n")
    lf = tmp_path / "lf.yaml"
    lf.write_bytes(text.encode("utf-8"))
    crlf = tmp_path / "crlf.yaml"
    crlf.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))

    a, b = Stimuli(lf), Stimuli(crlf)
    assert a.source_bytes == lf.read_bytes() and b.source_bytes == crlf.read_bytes()
    assert (a.description, a.params) == (b.description, b.params) == (
        "two lines", {"corner": ["tt", "ss"]})
    assert [t.tb_path for t in a.tests] == [t.tb_path for t in b.tests] == ["tb"]


def test_a_datasheet_built_in_code_records_no_snapshot(tmp_path):
    saved = run_store.save_run(_results(), _stim(), out_dir=tmp_path,
                               duration_s=1.0, yaml_name="")
    meta = run_meta.read_meta(saved.history_csv)
    assert meta["datasheet_sha256"] == "" and meta["datasheet_snapshot"] == ""
    assert not list((tmp_path / "history").glob("*.datasheet.yaml"))


def test_a_run_without_templates_archives_nothing(tmp_path):
    saved = run_store.save_run(_results(), _stim(template=""), out_dir=tmp_path,
                               duration_s=1.0, yaml_name="amp.yaml")
    assert saved.templates_dir == ""
    assert not list((tmp_path / "history").glob("*_templates"))
