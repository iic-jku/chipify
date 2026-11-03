# Copyright (c) 2026 Santiago Hofwimmer
"""Suite-wide fixtures."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from chipify import run_meta


@pytest.fixture
def tmp_path(tmp_path):
    """``tmp_path``, moved to a whitespace-free location when it has whitespace.

    pytest names its temporary root after the user, and in the IIC-OSIC-TOOLS
    container USER is "1000\\ndesigner": the newline went into every path, so
    tests that compare netlist text or messages containing a path failed
    unless run with --basetemp. Quoting a path with a space has its own test
    in test_netlist_paths; a newline cannot appear in a netlist line at all.
    """
    if not any(ch.isspace() for ch in str(tmp_path)):
        yield tmp_path
        return
    clean = Path(tempfile.mkdtemp(prefix="chipify-tests-"))
    try:
        yield clean
    finally:
        shutil.rmtree(clean, ignore_errors=True)

#: The real probe, for the test that checks its parsing with a stubbed
#: subprocess.run.
REAL_NGSPICE_VERSION = run_meta._ngspice_version


@pytest.fixture(autouse=True)
def _no_tool_version_probes(monkeypatch):
    """Keep tests from launching ngspice, vacask or git for a run sidecar.

    ``run_meta.write_meta`` records their versions. On Windows ``ngspice`` is
    the GUI build, so every test that wrote a sidecar opened an ngspice window
    until the 5 s probe timeout killed it.
    """
    for probe in ("_ngspice_version", "_vacask_version", "_git_commit"):
        monkeypatch.setattr(run_meta, probe, lambda: "")


@pytest.fixture(autouse=True)
def _fresh_forkserver():
    """Stop the forkserver after each test.

    Pool workers are forked from one forkserver per process, so they see its
    environment from when it started, not the current test's. Tests hand their
    workers a plugin engine through CHIPIFY_PLUGINS; the first forkserver test's
    directory reached every later one ("Unknown simulator engine 'timeoutecho';
    available: ngspice, vacask, interruptible", Linux CI only).
    """
    yield
    if sys.platform.startswith("linux"):
        from multiprocessing import forkserver
        forkserver._forkserver._stop()
