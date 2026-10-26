# Copyright (c) 2026 Santiago Hofwimmer
"""A run's settings are taken once and pinned in its workers.

Workers read the config again for every case (the timeout, the VACASK
engine's binary and paths). Without the pin, an override held by the parent
(the CLI's ``--timeout``) never reached them, and an edit of settings.json
during a sweep changed the timeout halfway through.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import threading

import pytest

simulator = pytest.importorskip("chipify.simulator")

from chipify import app_config, engines  # noqa: E402
# Alias: pytest must not try to collect chipify's Test domain class.
from chipify.util import Stimuli, Test as TbTest, Value  # noqa: E402

_WATCHDOG_SEC = 60


def test_a_worker_batch_keeps_the_run_settings_when_settings_json_changes(
        tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"sim_timeout_sec": 10}), encoding="utf-8")
    monkeypatch.setattr(app_config, "CONFIG_PATH", config)
    monkeypatch.setattr(app_config, "_config_cache", None)
    seen = []

    class TimeoutEcho(simulator.BaseSimulator):
        name = "timeout_echo_inproc"

        def generate_test_template(self, test):
            return ""

        def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
            seen.append(timeout_sec)
            # Someone edits settings.json while the sweep runs.
            config.write_text(json.dumps({"sim_timeout_sec": 99}), encoding="utf-8")
            return "MY_DATA: 1.0", None

    engines.register_engine(TimeoutEcho)
    try:
        test = TbTest("tb", [Value("v", None, None, None)])
        test.engine = "timeout_echo_inproc"
        test.template_str = "* {{ k }}"
        batch = [({"k": 0}, "000000"), ({"k": 1}, "000001")]
        rows = simulator.simulate_case_batch(
            (batch, [test], {}, "", {"sim_timeout_sec": 600.0}))
    finally:
        engines.reload_engines()

    assert seen == [600.0, 600.0]
    assert [r["sim_error"] for r in rows] == ["None", "None"]
    # The pin ends with the batch: this process reads settings.json again.
    monkeypatch.setattr(app_config, "_config_cache", None)
    assert app_config.load_config()["sim_timeout_sec"] == 99


def test_the_pin_beats_settings_json_and_project_defaults(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"num_cores": 8, "theme": "light"}), encoding="utf-8")
    monkeypatch.setattr(app_config, "CONFIG_PATH", config)
    monkeypatch.setattr(app_config, "_config_cache", None)
    previous = app_config.set_run_overrides({"num_cores": 2})
    try:
        cfg = app_config.load_config()
        assert (cfg["num_cores"], cfg["theme"]) == (2, "light")
        cfg["num_cores"] = 5                     # a caller's copy, not the pin
        assert app_config.load_config()["num_cores"] == 2
    finally:
        app_config.set_run_overrides(previous)
    assert app_config.load_config()["num_cores"] == 8


_ENGINE_SRC = textwrap.dedent(
    """
    import os
    from chipify.engines import BaseSimulator

    class TimeoutEcho(BaseSimulator):
        name = "timeoutecho"

        def generate_test_template(self, test):
            return "* {{ k }}"

        def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
            return f"MY_DATA: {timeout_sec} {os.getpid()}", None
    """
)

_START_METHODS = ["spawn"] + (
    ["forkserver"] if sys.platform.startswith("linux") else []
)


@pytest.mark.parametrize("start_method", _START_METHODS)
def test_overrides_reach_every_worker_process(tmp_path, monkeypatch, start_method):
    """``--timeout`` and ``--cores`` on a real pool: the workers are separate
    processes that read the config themselves."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "timeout_echo.py").write_text(_ENGINE_SRC, encoding="utf-8")
    # Workers inherit the environment, so they discover the same engine.
    monkeypatch.setenv("CHIPIFY_PLUGINS", str(plugins))
    engines.reload_engines()
    cfg = {"process_start_method": start_method, "num_cores": 4, "chunk_size": 1,
           "sim_timeout_sec": 10}
    monkeypatch.setattr(app_config, "load_config", lambda: dict(cfg))
    monkeypatch.setattr(simulator, "stage_files_to_ram", lambda _engines: None)

    test = TbTest("tb", [Value("timeout", None, None, None),
                         Value("pid", None, None, None)])
    test.engine = "timeoutecho"
    stim = Stimuli()
    stim.params = {"k": [0, 1, 2, 3]}
    stim.tests = [test]

    watchdog = threading.Timer(_WATCHDOG_SEC, simulator.abort_simulation)
    watchdog.start()
    try:
        df = simulator.run_sim(stim, out_dir=tmp_path / "out",
                               overrides={"sim_timeout_sec": 123.0, "num_cores": 1})
    finally:
        watchdog.cancel()
        engines.reload_engines()

    assert df is not None
    assert list(df["sim_error"]) == ["None"] * 4
    assert list(df["timeout"]) == [123.0] * 4
    assert df["pid"].nunique() == 1           # --cores 1: one worker process
    assert int(df["pid"].iloc[0]) != os.getpid()
    # The settings the workers used travel to the run's sidecar.
    settings = df.attrs["run_settings"]
    assert (settings["sim_timeout_sec"], settings["num_cores"]) == (123.0, 1)
    assert (settings["process_start_method"], settings["chunk_size"]) == (start_method, 1)
    assert df.attrs["engines"] == {"tb": "timeoutecho"}
