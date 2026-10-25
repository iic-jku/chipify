# Copyright (c) 2026 Santiago Hofwimmer
"""Regression test: a pool worker that dies abruptly must not stall a sweep.

``multiprocessing.Pool`` replaces a dead worker but never resolves the task it
was running, so ``run_sim`` waited forever for that case: mocking an
AsyncResult whose ``get()`` raises never exercised this. Here a real pool runs
a plugin engine that exits its worker process on one case; the replacement
worker simulates the other case, and the sweep must finish in bounded time
with an explicit WORKER_LOST row for the lost one.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

simulator = pytest.importorskip("chipify.simulator")

from chipify import app_config, engines  # noqa: E402
# Alias: pytest must not try to collect chipify's Test domain class.
from chipify.util import Stimuli, Test as TbTest, Value  # noqa: E402

_ENGINE_SRC = textwrap.dedent(
    """
    import os
    from chipify.engines import BaseSimulator

    class DieOnceEngine(BaseSimulator):
        name = "dieonce"

        def generate_test_template(self, test):
            return "{{ mode }}"

        def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
            if netlist.strip() == "die":
                os._exit(7)
            return "MY_DATA: 1.0", None
    """
)

_WATCHDOG_SEC = 60

_START_METHODS = ["spawn"] + (
    ["forkserver"] if sys.platform.startswith("linux") else []
)


@pytest.mark.parametrize("start_method", _START_METHODS)
def test_dead_worker_is_recorded_and_the_sweep_finishes(
    tmp_path, monkeypatch, start_method,
) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "die_once.py").write_text(_ENGINE_SRC, encoding="utf-8")
    # Workers inherit the environment, so they discover the same engine.
    monkeypatch.setenv("CHIPIFY_PLUGINS", str(plugins))
    engines.reload_engines()
    cfg = {"process_start_method": start_method, "num_cores": 1, "chunk_size": 1}
    monkeypatch.setattr(app_config, "load_config", lambda: cfg)
    monkeypatch.setattr(simulator, "stage_files_to_ram", lambda _engines: None)

    test = TbTest("tb_x", [Value("v", None, None, None)])
    test.engine = "dieonce"
    stim = Stimuli()
    stim.params = {"mode": ["die", "ok"]}
    stim.tests = [test]

    # Undetected, the lost case would hang the suite: abort it instead.
    watchdog = threading.Timer(_WATCHDOG_SEC, simulator.abort_simulation)
    watchdog.start()
    started = time.monotonic()
    try:
        df = simulator.run_sim(stim, out_dir=tmp_path / "out")
    finally:
        watchdog.cancel()
        engines.reload_engines()

    assert df is not None, "the sweep stalled until the watchdog aborted it"
    assert time.monotonic() - started < _WATCHDOG_SEC
    rows = df.set_index("mode")
    assert "WORKER_LOST" in rows.loc["die", "sim_error"]
    assert "exited with code 7" in rows.loc["die", "sim_error"]
    assert rows.loc["ok", "sim_error"] == "None"
    assert rows.loc["ok", "v"] == pytest.approx(1.0)


# ── Workers of a chipify process that died ────────────────────────────────────
# Module level: spawned processes import these by name.

def _worker(flag_dir: str) -> None:
    """Pool-worker stand-in: polls is_aborted() as the engines do."""
    from chipify.engines import abort

    Path(flag_dir, "started").write_text("1", encoding="utf-8")
    deadline = time.monotonic() + _WATCHDOG_SEC
    while time.monotonic() < deadline:
        if abort.is_aborted():
            Path(flag_dir, "noticed").write_text(
                "parent" if abort.parent_gone() else "flag", encoding="utf-8")
            return
        time.sleep(0.05)


def _chipify_process(flag_dir: str, start_method: str) -> None:
    """Stand-in for chipify-cli: start a worker, then die without cleanup."""
    multiprocessing.get_context(start_method).Process(
        target=_worker, args=(flag_dir,)).start()
    deadline = time.monotonic() + _WATCHDOG_SEC
    while not Path(flag_dir, "started").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    os._exit(0)        # like kill -9: no atexit handler, no pool shutdown


@pytest.mark.parametrize("start_method", _START_METHODS)
def test_a_worker_stops_when_its_chipify_process_dies(tmp_path, start_method) -> None:
    """A killed chipify-cli left its workers starting the cases still queued
    for them. A worker now reads its parent's death as an abort, which kills
    the running simulator and starts no further case."""
    simulator.clear_abort_flag()
    parent = multiprocessing.get_context("spawn").Process(
        target=_chipify_process, args=(str(tmp_path), start_method))
    parent.start()
    parent.join(_WATCHDOG_SEC)

    noticed = tmp_path / "noticed"
    deadline = time.monotonic() + _WATCHDOG_SEC
    while not noticed.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert noticed.exists(), "the worker kept running after its parent died"
    assert noticed.read_text(encoding="utf-8") == "parent"


def test_the_main_process_has_no_parent_to_lose() -> None:
    from chipify.engines import abort
    assert abort.parent_gone() is False
