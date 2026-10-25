# Copyright (c) 2026 Santiago Hofwimmer
"""run_xschem: a netlist with missing symbols is rejected before simulation.

xschem 3.4.8RC exits with rc=10 for clean schematics as well as broken ones,
so the exit code says nothing; the netlist's own ``IS MISSING`` comment is the
signal. The netlists below are what xschem actually wrote in the container.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from chipify import settings
from chipify.engines import xschem as xschem_mod

_MISSING = (
    "** sch_path: /tmp/b18/tb.sch\n"
    "**.subckt tb\n"
    "*  x1 -  source_follower  IS MISSING !!!!\n"
    "R1 net1 net2 1k\n"
    "**.ends\n"
    ".end\n"
)
_CLEAN = "** sch_path: /tmp/b18/tb.sch\nR1 net1 net2 1k\n.end\n"


def test_missing_symbols_names_instance_and_symbol() -> None:
    assert xschem_mod.missing_symbols(_MISSING) == ["x1 (source_follower)"]
    assert xschem_mod.missing_symbols(_CLEAN) == []


def _fake_xschem(netlist: str):
    class _Process:
        """Writes *netlist* where xschem would, then exits with rc=10."""

        def __init__(self, cmd, **_kw) -> None:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            (out_dir / f"{Path(cmd[-1]).stem}.spice").write_text(
                netlist, encoding="utf-8")
            self.stdin = io.StringIO()
            self.pid = 4242
            self.returncode = 10

        def poll(self):
            return self.returncode

    return _Process


@pytest.mark.parametrize("netlist, ok", [(_CLEAN, True), (_MISSING, False)])
def test_run_xschem_rejects_a_netlist_with_missing_symbols(
    tmp_path, monkeypatch, netlist, ok,
) -> None:
    monkeypatch.setattr(settings, "FAST_TMP", str(tmp_path))
    monkeypatch.setattr(settings, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(xschem_mod.subprocess, "Popen", _fake_xschem(netlist))
    (tmp_path / "tb.sch").write_text("v {xschem version=3.4.5}\n", encoding="utf-8")

    if ok:
        xschem_mod.run_xschem(tmp_path / "tb.sch")      # rc=10 alone is fine
        assert (tmp_path / "tb.spice").exists()
    else:
        with pytest.raises(RuntimeError, match=r"x1 \(source_follower\)"):
            xschem_mod.run_xschem(tmp_path / "tb.sch")


def test_run_xschem_names_a_missing_schematic(tmp_path, monkeypatch) -> None:
    """xschem reported a missing .sch only as 'wrote no netlist'."""
    def _must_not_start(*_a, **_k):
        raise AssertionError("xschem must not run for a missing schematic")

    monkeypatch.setattr(settings, "FAST_TMP", str(tmp_path))
    monkeypatch.setattr(xschem_mod.subprocess, "Popen", _must_not_start)
    with pytest.raises(RuntimeError, match=r"tb_gone\.sch does not exist"):
        xschem_mod.run_xschem(tmp_path / "tb_gone.sch")
