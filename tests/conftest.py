# Copyright (c) 2026 Santiago Hofwimmer
"""Suite-wide fixtures."""
from __future__ import annotations

import pytest

from chipify import run_meta

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
