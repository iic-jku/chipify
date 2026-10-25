# Copyright (c) 2026 Santiago Hofwimmer
"""
tests/test_run_meta.py

Unit tests for chipify.run_meta.update_meta (notes/tags annotation backend).
"""
from __future__ import annotations

import os

from chipify import run_meta


def test_update_meta_merges_into_existing(tmp_path) -> None:
    csv = os.path.join(str(tmp_path), "run_20260611_120000.csv")
    open(csv, "w", encoding="utf-8").close()
    run_meta.write_meta(csv, yaml_name="amp.yaml", total_runs=10)

    meta = run_meta.update_meta(csv, notes="hello", tags=["x"])
    assert meta["notes"] == "hello"
    assert meta["tags"] == ["x"]
    # Pre-existing fields survive the merge.
    assert meta["yaml"] == "amp.yaml"
    assert meta["total_runs"] == 10
    assert run_meta.read_meta(csv)["notes"] == "hello"


def test_update_meta_creates_minimal_sidecar(tmp_path) -> None:
    csv = os.path.join(str(tmp_path), "run_20260611_130000.csv")
    open(csv, "w", encoding="utf-8").close()

    meta = run_meta.update_meta(csv, notes="legacy run", tags=[])
    assert meta["notes"] == "legacy run"
    assert meta["schema_version"] == 1
    # No datasheet attribution is invented — filtering stays unaffected.
    assert "yaml" not in meta


def test_ngspice_version_is_the_version_not_the_banner_border(monkeypatch) -> None:
    """The first line of `ngspice --version` is a row of asterisks."""
    import subprocess

    banner = ("******\n"
              "** ngspice-47 : Circuit level simulation program\n"
              "** Compiled with KLU Direct Linear Solver\n")
    monkeypatch.setattr(run_meta.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, stdout=banner, stderr=""))
    from tests.conftest import REAL_NGSPICE_VERSION   # conftest stubs the probe
    assert REAL_NGSPICE_VERSION() == "ngspice-47"


# ── Waveform directories: relative to the output root ─────────────────────────

def _history_run(root, wave_kind="transient"):
    """A history CSV plus a waveform dir under *root*, like a finished run."""
    wave = root / "analysis_data" / wave_kind / "20260611_120000"
    wave.mkdir(parents=True)
    hist = root / "history" / "run_20260611_120000.csv"
    hist.parent.mkdir(parents=True)
    hist.write_text("run_id,sim_error,gain\n0,None,1.0\n", encoding="utf-8")
    return hist, wave


def test_analysis_dirs_are_stored_relative_to_the_output_root(tmp_path) -> None:
    hist, wave = _history_run(tmp_path / "out" / "amp")
    run_meta.write_meta(hist, analysis_dirs={"transient": str(wave)},
                        tran_dir=str(wave))
    meta = run_meta.read_meta(hist)
    assert meta["analysis_dirs"] == {
        "transient": "analysis_data/transient/20260611_120000"}
    assert meta["tran_dir"] == "analysis_data/transient/20260611_120000"
    assert run_meta.analysis_dirs_for(hist) == {"transient": str(wave)}


def test_a_moved_output_folder_still_finds_its_waveforms(tmp_path) -> None:
    hist, wave = _history_run(tmp_path / "out")
    run_meta.write_meta(hist, analysis_dirs={"transient": str(wave)})
    moved = tmp_path / "copied_out"
    (tmp_path / "out").rename(moved)
    moved_hist = moved / "history" / hist.name
    assert run_meta.analysis_dirs_for(moved_hist) == {
        "transient": str(moved / "analysis_data" / "transient" / "20260611_120000")}


def test_older_sidecars_with_absolute_paths_still_resolve(tmp_path) -> None:
    import json

    hist, wave = _history_run(tmp_path / "out")
    hist.with_suffix(".meta.json").write_text(
        json.dumps({"schema_version": 1, "tran_dir": str(wave)}), encoding="utf-8")
    # Legacy transient-only key fills in for a missing analysis_dirs entry.
    assert run_meta.analysis_dirs_for(hist) == {"transient": str(wave)}


def test_load_csv_restores_the_runs_own_waveform_dirs(tmp_path) -> None:
    """A reloaded history run must not fall back to the newest run's data."""
    from chipify import data_loader
    from chipify.uikit.services import transient_loader

    root = tmp_path / "out"
    hist, wave = _history_run(root)
    run_meta.write_meta(hist, analysis_dirs={"transient": str(wave)})
    newer = root / "analysis_data" / "transient" / "20260612_090000"
    newer.mkdir()
    (root / "analysis_data" / "transient" / ".latest").write_text(
        str(newer), encoding="utf-8")

    df = data_loader.load_csv(str(hist))
    assert df.attrs["analysis_dirs"] == {"transient": str(wave)}
    assert transient_loader.resolve_analysis_dir(df, root, "transient") == str(wave)


def _newer_run_with_pointer(root, kind="transient"):
    newer = root / "analysis_data" / kind / "20260612_090000"
    newer.mkdir(parents=True)
    (root / "analysis_data" / kind / ".latest").write_text(str(newer), encoding="utf-8")
    return newer


def test_a_deleted_waveform_dir_is_not_replaced_by_a_newer_runs(tmp_path) -> None:
    """The run's own directory is gone: report no data instead of quietly
    showing the curves of the run behind the .latest pointer."""
    import shutil
    from chipify import data_loader
    from chipify.uikit.services import transient_loader

    root = tmp_path / "out"
    hist, wave = _history_run(root)
    run_meta.write_meta(hist, analysis_dirs={"transient": str(wave)})
    _newer_run_with_pointer(root)
    shutil.rmtree(wave)

    df = data_loader.load_csv(str(hist))
    assert transient_loader.resolve_analysis_dir(df, root, "transient") == ""


def test_a_kind_the_run_never_captured_is_not_borrowed(tmp_path) -> None:
    from chipify import data_loader
    from chipify.uikit.services import transient_loader

    root = tmp_path / "out"
    hist, wave = _history_run(root)
    run_meta.write_meta(hist, analysis_dirs={"transient": str(wave)})
    _newer_run_with_pointer(root, kind="ac")

    df = data_loader.load_csv(str(hist))
    assert transient_loader.resolve_analysis_dir(df, root, "ac") == ""


def test_a_run_without_recorded_dirs_still_uses_the_pointer(tmp_path) -> None:
    """Legacy CSVs carry no waveform directories: the pointer is all there is."""
    import pandas as pd
    from chipify.uikit.services import transient_loader

    newer = _newer_run_with_pointer(tmp_path)
    assert transient_loader.resolve_analysis_dir(
        pd.DataFrame(), tmp_path, "transient") == str(newer)


def test_resolve_analysis_dir_takes_relative_meta_paths_from_out_dir(tmp_path) -> None:
    import pandas as pd
    from chipify.uikit.services import transient_loader

    _, wave = _history_run(tmp_path, wave_kind="ac")
    meta = {"analysis_dirs": {"ac": "analysis_data/ac/20260611_120000"}}
    assert transient_loader.resolve_analysis_dir(
        pd.DataFrame(), tmp_path, "ac", meta=meta) == str(wave)
