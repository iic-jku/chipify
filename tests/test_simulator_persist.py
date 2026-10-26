# Copyright (c) 2026 Santiago Hofwimmer
"""Tests for chipify.simulator analysis-capture diagnostics.

Covers _persist_analyses (writes per-run CSVs and, crucially, records a reason on
the result row when an analysis produced no data — the silent Bode/AC "no data"
symptom) and the ngspice-error extraction helper.
"""
from __future__ import annotations

import os

import pytest

# simulator.py imports heavy runtime deps (tqdm, jinja2, …); skip cleanly if the
# environment doesn't have them installed. CI installs the package, so it runs there.
simulator = pytest.importorskip("chipify.simulator")


class _FakeAnalysis:
    """Stand-in for an Analysis: optionally writes the dest CSV on persist."""

    def __init__(self, kind: str, write_csv: bool = True) -> None:
        self.kind = kind
        self.write_csv = write_csv
        self.persisted: tuple[str, str] | None = None

    def persist_to_csv(self, src_tab: str, dest_csv: str) -> None:
        self.persisted = (src_tab, dest_csv)
        if self.write_csv:
            with open(dest_csv, "w", encoding="utf-8") as fh:
                fh.write("frequency,out_mag,out_phase\n1,2,3\n")


def test_note_when_tab_missing(tmp_path) -> None:
    """ngspice produced no tab → reason recorded on the row, persist not called."""
    an = _FakeAnalysis("ac")
    missing_tab = str(tmp_path / "run_ac.tab")          # never created
    dest_dir = str(tmp_path / "out_ac")
    os.makedirs(dest_dir, exist_ok=True)
    sample: dict = {}

    simulator._persist_analyses(
        [an], {"ac": missing_tab}, {"ac": dest_dir},
        run_id=1, tb_safe="tb_x", tb_path="tb_x", sample=sample,
    )

    assert an.persisted is None
    assert "tb_x__ac_capture" in sample
    assert "no data" in sample["tb_x__ac_capture"]


def test_note_when_tab_present_but_no_csv(tmp_path) -> None:
    """Tab exists but persist writes nothing (empty/unreadable) → reason recorded."""
    an = _FakeAnalysis("ac", write_csv=False)
    tab = tmp_path / "run_ac.tab"
    tab.write_text("* comment only\n", encoding="utf-8")  # exists, but yields no CSV
    dest_dir = str(tmp_path / "out_ac")
    os.makedirs(dest_dir, exist_ok=True)
    sample: dict = {}

    simulator._persist_analyses(
        [an], {"ac": str(tab)}, {"ac": dest_dir},
        run_id=1, tb_safe="tb_x", tb_path="tb_x", sample=sample,
    )

    assert an.persisted is not None                      # persist was attempted
    assert "tb_x__ac_capture" in sample                  # but no CSV → reason recorded
    assert "empty" in sample["tb_x__ac_capture"]


def test_writes_csv_when_tab_present(tmp_path) -> None:
    """An existing tab that persists to a real CSV → no failure note."""
    an = _FakeAnalysis("ac", write_csv=True)
    tab = tmp_path / "run_ac.tab"
    tab.write_text("1 2 3\n", encoding="utf-8")
    dest_dir = str(tmp_path / "out_ac")
    os.makedirs(dest_dir, exist_ok=True)
    sample: dict = {}

    failures = simulator._persist_analyses(
        [an], {"ac": str(tab)}, {"ac": dest_dir},
        run_id=7, tb_safe="tb_x", tb_path="tb_x", sample=sample,
    )

    assert an.persisted is not None
    assert os.path.exists(os.path.join(dest_dir, "run_7__tb_x.csv"))
    assert "tb_x__ac_capture" not in sample
    assert failures == []


class _ReasonEngine:
    """Engine stand-in that knows why it wrote no tab."""

    name = "vacask"

    def capture_error(self, kind: str) -> str:
        return "signal(s) ['v(typo)'] not in the .raw output"

    def run_log_tail(self, n_lines: int = 25) -> str:
        return "Error: unrelated line from a successful run"

    @staticmethod
    def extract_error(log_text: str) -> str:
        return log_text


def test_engine_capture_reason_replaces_generic_note(tmp_path) -> None:
    """The engine's own reason names the missing signal; the generic 'does the
    testbench run its analysis?' guess and the log tail would mislead."""
    an = _FakeAnalysis("transient")
    sample: dict = {}

    failures = simulator._persist_analyses(
        [an], {"transient": str(tmp_path / "never_written.tab")},
        {"transient": str(tmp_path)},
        run_id=1, tb_safe="tb_x", tb_path="tb_x", sample=sample,
        engine=_ReasonEngine(),
    )

    note = sample["tb_x__transient_capture"]
    assert "v(typo)" in note
    assert "does the testbench run" not in note
    assert "unrelated" not in note
    assert failures == [note]


def test_skips_analysis_without_output_dir(tmp_path) -> None:
    """If no output dir was set up for the kind, skip silently (no note)."""
    an = _FakeAnalysis("ac")
    sample: dict = {}

    simulator._persist_analyses(
        [an], {}, {},
        run_id=1, tb_safe="tb_x", tb_path="tb_x", sample=sample,
    )

    assert an.persisted is None
    assert sample == {}


def test_extract_ngspice_error() -> None:
    """Picks the error line, else the last non-empty line; '' on empty input."""
    txt = "Doing analysis at TEMP = 27\nError: vector out not available\nrun simulation done"
    assert "out not available" in simulator._extract_ngspice_error(txt)
    assert simulator._extract_ngspice_error("") == ""
    assert simulator._extract_ngspice_error("foo\nbar\n") == "bar"


def test_inject_capture_before_quit() -> None:
    """Capture must be spliced before a terminating `quit` (which exits ngspice)."""
    netlist = (
        ".control\n"
        "ac dec 10 1 1e9\n"
        "echo MY_DATA:1 2\n"
        "quit\n"
        ".endc\n"
    )
    out = simulator._inject_capture(netlist, "WRDATA_HERE")
    assert "WRDATA_HERE" in out
    assert out.index("WRDATA_HERE") < out.index("quit") < out.index(".endc")


def test_inject_capture_handles_exit() -> None:
    netlist = ".control\nrun\nexit\n.endc\n"
    out = simulator._inject_capture(netlist, "CAP")
    assert out.index("CAP") < out.index("exit")


def test_inject_capture_before_endc_when_no_quit() -> None:
    """Without a quit/exit, capture goes before .endc (unchanged behavior)."""
    netlist = ".control\nac dec 10 1 1e9\n.endc\n"
    out = simulator._inject_capture(netlist, "WRDATA_HERE")
    assert out.index("WRDATA_HERE") < out.index(".endc")


def test_inject_capture_ignores_comments_mentioning_control() -> None:
    """A comment spelling .control / .endc / quit used to anchor the injection,
    which then landed in the comment's position, outside the real block."""
    netlist = (
        "* the .endc below closes the chipify .control block; quit is last\n"
        "V1 in 0 1\n"
        ".control\n"
        "op\n"
        "quit\n"
        ".endc\n"
    )
    out = simulator._inject_capture(netlist, "CAP")
    assert out.index(".control\n") < out.index("CAP") < out.index("quit\n")
    assert out.splitlines()[0] == netlist.splitlines()[0]   # comment untouched


def test_inject_capture_honours_the_jinja_newline_prefix() -> None:
    """`* {{ NL }}` keeps a line inert in xschem and makes it a command for
    chipify, so it counts as the directive itself."""
    netlist = ".param x=1\n* {{ NL }}.control\nop\n* {{ NL }}quit\n* {{ NL }}.endc\n"
    out = simulator._inject_capture(netlist, "CAP")
    assert out.index("* {{ NL }}.control") < out.index("CAP") < out.index("* {{ NL }}quit")


def test_finalize_sets_threads_only_in_real_control_blocks() -> None:
    from chipify.engines.ngspice import NgspiceSimulator

    class _T:
        value_lst: list = []
        analyses: list = []

    netlist = "* see .control below\n.controlled_source x\n.control\nop\n.endc\n"
    out = NgspiceSimulator()._finalize_netlist(netlist, _T())
    assert out.count("set num_threads=1") == 1
    assert out.index(".control\n") < out.index("set num_threads=1")
    assert "* see .control below\n.controlled_source x\n" in out


def test_log_capture_failures_dedupes(caplog) -> None:
    """Distinct capture notes on rows are surfaced once each from the main process."""
    rows = [
        {"tb_x__ac_capture": "ac produced no data - empty tab"},
        {"tb_x__ac_capture": "ac produced no data - empty tab"},  # dup
        {"other": 1},
    ]
    with caplog.at_level("WARNING"):
        simulator._log_capture_failures(rows)
    hits = [r for r in caplog.records if "capture failed" in r.getMessage()]
    assert len(hits) == 1
    assert "tb_x__ac" in hits[0].getMessage()


# ── MY_DATA echo auto-injection (generate_test_template) ──────────────────────

class _StubValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubAnalysis:
    """Analysis stub whose ngspice_inject is a recognisable marker."""
    kind = "transient"

    def ngspice_inject(self) -> str:
        return "WRDATA_MARK"


class _StubTest:
    def __init__(self, value_names, analyses=()) -> None:
        self.tb_path = "tb_op"
        self.value_lst = [_StubValue(n) for n in value_names]
        self.analyses = list(analyses)


def _gen_template(monkeypatch, tmp_path, spice: str, test) -> str:
    """Drive NgspiceSimulator.generate_test_template against an on-disk netlist,
    stubbing out the xschem subprocess."""
    from chipify import settings
    from chipify.engines import ngspice as ngspice_mod

    monkeypatch.setattr(settings, "FAST_TMP", str(tmp_path))
    monkeypatch.setattr(settings, "TB_DIR", str(tmp_path))
    monkeypatch.setattr(ngspice_mod, "run_xschem", lambda *a, **k: None)
    (tmp_path / "tb_op.spice").write_text(spice, encoding="utf-8")
    return simulator.NgspiceSimulator().generate_test_template(test)


def test_generate_template_injects_my_data_echo(tmp_path, monkeypatch) -> None:
    """Chipify emits `echo MY_DATA:$&<name>` in value_lst order, before quit, and
    strips any stale hand-written echo so only one MY_DATA line survives."""
    spice = (
        ".control\n"
        "op\n"
        "let ve = (v(outp)+v(outn))/2\n"
        "let vd = v(outp)-v(outn)\n"
        "echo MY_DATA:$&vd $&ve\n"      # stale, wrong order — must be dropped
        "quit\n"
        ".endc\n"
    )
    out = _gen_template(monkeypatch, tmp_path, spice, _StubTest(["ve", "vd"]))

    assert out.count("MY_DATA:") == 1
    assert "echo MY_DATA:$&ve $&vd" in out
    assert out.index("MY_DATA:") < out.index("quit")
    assert "$&vd $&ve" not in out          # the stale echo is gone


def test_generate_template_echo_precedes_wrdata(tmp_path, monkeypatch) -> None:
    """When both scalars and a waveform analysis are present, the scalar echo is
    spliced before the wrdata/setplot capture (so $&<name> resolves in the plot
    the testbench's own meas left current, before setplot switches it)."""
    spice = ".control\nac dec 10 1 1e9\nquit\n.endc\n"
    test = _StubTest(["gain"], analyses=[_StubAnalysis()])
    out = _gen_template(monkeypatch, tmp_path, spice, test)

    assert out.index("MY_DATA:") < out.index("WRDATA_MARK") < out.index("quit")


def test_generate_template_no_echo_without_scalars(tmp_path, monkeypatch) -> None:
    """A transient-only testbench (no value_lst) gets the waveform capture but no
    MY_DATA echo."""
    spice = ".control\ntran 1u 1m\nquit\n.endc\n"
    test = _StubTest([], analyses=[_StubAnalysis()])
    out = _gen_template(monkeypatch, tmp_path, spice, test)

    assert "MY_DATA:" not in out
    assert "WRDATA_MARK" in out


# ── VACASK scalar extraction from .raw ────────────────────────────────────────

def _vacask_test(value_names, measure=None):
    from chipify.util import Test, Value
    t = Test("tb_v", [Value(n, None, None, None) for n in value_names])
    t.measure = measure or {}
    t.analyses = []
    return t


def _patch_raw(monkeypatch, bucket) -> None:
    from chipify.engines import vacask as vacask_mod
    # _vacask_extract_results guards on Path(raw_file).exists(); make any path
    # report as present so the in-memory bucket below is what gets exercised.
    monkeypatch.setattr(vacask_mod.Path, "exists", lambda _self: True)
    monkeypatch.setattr(vacask_mod, "_read_raw_file", lambda _f: {"dc": bucket})


def test_vacask_extract_finds_node_voltages(monkeypatch) -> None:
    _patch_raw(monkeypatch, {"outp": [0.9], "outn": [0.5], "__x__": [0.0]})
    line, err = simulator._vacask_extract_results("x.raw", _vacask_test(["outp", "outn"]), {})
    assert err is None
    assert line.replace("MY_DATA:", "").split() == ["0.9", "0.5"]


def test_vacask_extract_reports_available_signals_when_no_match(monkeypatch) -> None:
    """A run where no requested value matches must surface the .raw's actual
    signal names (not a silent all-NaN row)."""
    _patch_raw(monkeypatch, {"outp": [0.9], "outn": [0.5], "__x__": [0.0]})
    line, err = simulator._vacask_extract_results("x.raw", _vacask_test(["vgain", "pm"]), {})
    assert line is None
    assert "NO_MATCHING_SIGNALS" in err
    assert "outp" in err and "outn" in err


def test_vacask_missing_waveform_signal_is_reported_not_zero_filled(
    monkeypatch, tmp_path,
) -> None:
    """A declared transient signal absent from the .raw writes no tab and hands
    the reason back to the engine, instead of a zero-filled waveform."""
    from chipify.analyses import TransientAnalysis
    from chipify.engines import vacask as vacask_mod
    x = [0.0, 1.0]
    monkeypatch.setattr(vacask_mod.Path, "exists", lambda _self: True)
    monkeypatch.setattr(vacask_mod, "_read_raw_file",
                        lambda _f: {"transient": {"__x__": x, "time": x}})
    test = _vacask_test([])
    test.analyses = [TransientAnalysis(["v(out)"])]
    tab = tmp_path / "tran.tab"
    errors: dict = {}

    simulator._vacask_extract_results(
        "x.raw", test, {"transient": str(tab)}, capture_errors=errors,
    )

    # Path.exists is patched above, so ask the OS directly.
    assert not os.path.exists(tab)
    assert "v(out)" in errors["transient"]


def _patch_ac_raw(monkeypatch, bucket) -> None:
    from chipify.engines import vacask as vacask_mod
    monkeypatch.setattr(vacask_mod.Path, "exists", lambda _self: True)
    monkeypatch.setattr(vacask_mod, "_read_raw_file", lambda _f: {"ac": bucket})


_AC_BUCKET = {"__x__": [1.0, 1e3], "frequency": [1.0, 1e3], "vout": [1 + 0j, 0.5 - 0.5j]}


def test_vacask_measure_on_ac_data_uses_the_magnitude(monkeypatch) -> None:
    """An AC-only bench's measure: saw float-cast vectors: 0.5, not 0.7071."""
    _patch_ac_raw(monkeypatch, _AC_BUCKET)
    test = _vacask_test(["gain"], measure={"gain": "last(abs(vout))"})
    line, err = simulator._vacask_extract_results("x.raw", test, {})
    assert err is None
    assert float(line.replace("MY_DATA:", "")) == pytest.approx(0.7071067811865476)


def test_vacask_complex_value_without_a_reduction_is_an_error(monkeypatch) -> None:
    """A value named after a complex AC vector used to report its real part."""
    _patch_ac_raw(monkeypatch, _AC_BUCKET)
    line, err = simulator._vacask_extract_results("x.raw", _vacask_test(["vout"]), {})
    assert line is None
    assert "MEASURE_ERROR(vout)" in err and "last(abs(vout))" in err


def test_vacask_complex_measure_result_is_an_error(monkeypatch) -> None:
    _patch_ac_raw(monkeypatch, _AC_BUCKET)
    test = _vacask_test(["vlast"], measure={"vlast": "last(vout)"})
    line, err = simulator._vacask_extract_results("x.raw", test, {})
    assert line is None
    assert "MEASURE_ERROR(vlast)" in err and "complex value" in err


def test_vacask_extract_partial_match_is_not_an_error(monkeypatch) -> None:
    """If at least one value resolves, the row is kept (missing ones → nan)."""
    _patch_raw(monkeypatch, {"outp": [0.9], "__x__": [0.0]})
    line, err = simulator._vacask_extract_results("x.raw", _vacask_test(["outp", "outn"]), {})
    assert err is None
    vals = line.replace("MY_DATA:", "").split()
    assert vals[0] == "0.9" and vals[1] == "nan"
