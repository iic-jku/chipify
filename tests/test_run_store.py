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


def test_a_run_without_templates_archives_nothing(tmp_path):
    saved = run_store.save_run(_results(), _stim(template=""), out_dir=tmp_path,
                               duration_s=1.0, yaml_name="amp.yaml")
    assert saved.templates_dir == ""
    assert not list((tmp_path / "history").glob("*_templates"))
