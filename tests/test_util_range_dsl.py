# Copyright (c) 2026 Santiago Hofwimmer
"""
tests/test_util_range_dsl.py

Unit tests for chipify.schema._parse_range_dsl.

Covers:
- range(N), range(a, b), range(a, b, step)
- linspace(start, stop, N)
- logspace(start, stop, N)
- Negative arguments
- Rejection of dangerous expressions (os.system, imports, attribute chains)
- Rejection of unknown function names
- Rejection of non-constant arguments
"""
from __future__ import annotations

import pytest
from chipify.schema import _parse_range_dsl, SchemaError, validate_parameters


# ── range() ───────────────────────────────────────────────────────────────────

def test_range_single_arg() -> None:
    result = _parse_range_dsl("range(5)")
    assert result == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_range_two_args() -> None:
    result = _parse_range_dsl("range(2, 7)")
    assert result == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_range_three_args_step() -> None:
    result = _parse_range_dsl("range(0, 10, 2)")
    assert result == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_range_negative_start() -> None:
    result = _parse_range_dsl("range(-3, 3)")
    assert result == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]


def test_range_renders_integer_seeds() -> None:
    """As floats, `seed: range(200)` rendered `.option SEED = 3.0` and stored
    `seed = 3.0` in the results."""
    from jinja2 import Template
    seeds = validate_parameters({"seed": "range(1, 4)"})["seed"]
    assert seeds == [1, 2, 3] and all(type(s) is int for s in seeds)
    assert Template(".option SEED = {{ seed }}").render(seed=seeds[2]) == ".option SEED = 3"
    # A whole-number float literal is still a valid range argument.
    assert _parse_range_dsl("range(2.0)") == [0, 1]


@pytest.mark.parametrize("expr", ["range(2.5)", "range(0, 10, 0.5)", "range(-1.5, 3)"])
def test_range_rejects_fractional_arguments(expr) -> None:
    """int() truncated them silently: range(2.5) swept 0 and 1."""
    with pytest.raises(ValueError, match="whole numbers"):
        _parse_range_dsl(expr)


# ── linspace() ────────────────────────────────────────────────────────────────

def test_linspace_basic() -> None:
    result = _parse_range_dsl("linspace(0, 1, 5)")
    assert len(result) == 5
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_linspace_float_bounds() -> None:
    result = _parse_range_dsl("linspace(0.5, 1.5, 3)")
    assert result[0] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.5)


# ── logspace() ────────────────────────────────────────────────────────────────

def test_logspace_basic() -> None:
    result = _parse_range_dsl("logspace(0, 2, 3)")
    assert len(result) == 3
    assert result[0] == pytest.approx(1.0)     # 10^0 = 1
    assert result[2] == pytest.approx(100.0)    # 10^2 = 100


# ── Security: rejection of dangerous expressions ──────────────────────────────

def test_rejects_os_system() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("os.system('rm -rf /')")


def test_rejects_import() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("__import__('os')")


def test_rejects_attribute_access() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("os.path.join('a', 'b')")


def test_rejects_unknown_function() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("arange(10)")


def test_rejects_non_constant_arg() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("range(n)")  # variable, not a literal


def test_rejects_binary_expression() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("1 + 2")


def test_rejects_string_arg() -> None:
    with pytest.raises(ValueError):
        _parse_range_dsl("range('hello')")


# ── validate_parameters integration ──────────────────────────────────────────

def test_validate_parameters_expands_range() -> None:
    result = validate_parameters({"n_steps": "range(4)"})
    assert result["n_steps"] == [0.0, 1.0, 2.0, 3.0]


def test_validate_parameters_passes_list_unchanged() -> None:
    result = validate_parameters({"temp": [27, 85, 125]})
    assert result["temp"] == [27, 85, 125]


def test_validate_parameters_wraps_numeric_scalar() -> None:
    # Scalars must come back as one-element lists: generate_cases builds the
    # sweep grid with itertools.product, which crashes on a bare float.
    result = validate_parameters({"vdd": 1.8})
    assert result["vdd"] == [pytest.approx(1.8)]


def test_validate_parameters_wraps_string_scalar() -> None:
    # A bare string would otherwise be swept character by character.
    result = validate_parameters({"corner": "tt"})
    assert result["corner"] == ["tt"]


def test_validate_parameters_raises_on_unsafe_string() -> None:
    with pytest.raises(SchemaError):
        validate_parameters({"key": "os.system('bad')"})


# ── measurement bound validation ──────────────────────────────────────────────

def test_validate_datasheet_rejects_non_numeric_bound() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_x": {"gain": {"min": "abc", "max": 80}}},
    }
    with pytest.raises(SchemaError, match="tb_x.gain"):
        validate_datasheet(data)


def test_validate_datasheet_allows_missing_bounds() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_x": {"gain": {"min": 40}}},
    }
    stim = validate_datasheet(data)
    val = stim.tests[0].value_lst[0]
    assert val.vmin == 40.0 and val.vmax is None


def test_validate_datasheet_parses_optional_unit() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_x": {
            "gain": {"min": 40, "unit": "dB"},
            "offset": {"max": 5},          # no unit → None
        }},
    }
    stim = validate_datasheet(data)
    gain, offset = stim.tests[0].value_lst
    assert gain.unit == "dB"
    assert offset.unit is None


def test_validate_datasheet_parses_per_testbench_engine() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {
            "tb_v": {"gain": {"min": 1}, "engine": "VACASK"},   # normalised
            "tb_n": {"ve": {"max": 1}},                          # default → None
        },
    }
    stim = validate_datasheet(data)
    by = {t.tb_path: t.engine for t in stim.tests}
    assert by["tb_v"] == "vacask"
    assert by["tb_n"] is None


def test_validate_datasheet_rejects_unknown_engine() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_x": {"engine": "hspice", "gain": {"min": 1}}},
    }
    with pytest.raises(SchemaError, match="engine"):
        validate_datasheet(data)


def test_validate_datasheet_parses_per_testbench_source() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {
            "tb_i": {"gain": {"min": 1}, "source": "netlist"},   # imported deck
            "tb_s": {"ve": {"max": 1}},                          # default → xschem
        },
    }
    stim = validate_datasheet(data)
    by = {t.tb_path: t.netlist_source for t in stim.tests}
    assert by["tb_i"] == "netlist"
    assert by["tb_s"] == "xschem"
    # The source: key must not be mistaken for a measurement.
    tb_i = next(t for t in stim.tests if t.tb_path == "tb_i")
    assert [v.name for v in tb_i.value_lst] == ["gain"]


def test_validate_datasheet_rejects_unknown_source() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_x": {"source": "magic", "gain": {"min": 1}}},
    }
    with pytest.raises(SchemaError, match="source"):
        validate_datasheet(data)


# ── block types ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("data, where", [
    # A list where the testbench mapping belongs became a plan with no tests.
    ({"parameters": {"temp": [27]}, "tests": [{"tb_x": {"gain": {"min": 1}}}]},
     r"^tests: expected a mapping of testbenches, got list"),
    ({"parameters": [27], "tests": {"tb_x": {"gain": {"min": 1}}}},
     r"^parameters: expected a mapping"),
    ({"tests": {"tb_x": ["gain"]}}, r"tests\.tb_x: expected a mapping"),
    # Limits as a list used to mean "no limits": always PASS.
    ({"tests": {"tb_x": {"gain": [9, 11]}}}, r"tests\.tb_x\.gain: expected a mapping"),
    ({"tests": {"tb_x": {"transient_signals": "v(out)"}}},
     r"tests\.tb_x\.transient_signals: expected a list"),
    ({"tests": {"tb_x": {"measure": "gain * 2"}}}, r"tests\.tb_x\.measure: expected a mapping"),
])
def test_validate_datasheet_rejects_wrong_block_types(data, where) -> None:
    from chipify.schema import validate_datasheet
    with pytest.raises(SchemaError, match=where):
        validate_datasheet(data)


def test_validate_datasheet_reads_empty_blocks_as_empty() -> None:
    """`tests:` or `tb_x:` with nothing under it is a datasheet being written."""
    from chipify.schema import validate_datasheet
    assert validate_datasheet({"parameters": None, "tests": None}).tests == []
    stim = validate_datasheet({"tests": {"tb_x": None, "tb_y": {"gain": None,
                                                                "dc_signals": None}}})
    assert [t.tb_path for t in stim.tests] == ["tb_x", "tb_y"]
    assert stim.tests[1].value_lst[0].vmin is None
    assert stim.tests[1].analyses == []


# ── one results column per name ───────────────────────────────────────────────

def test_validate_datasheet_rejects_measurement_shared_by_two_testbenches() -> None:
    """Results are keyed by name: the second testbench's a0_db overwrote the
    first one's value while its error scope still pointed at the first."""
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {
            "tb_lpf_opamp": {"a0_db": {"min": 60}},
            "tb_lpf_ac": {"a0_db": {"min": -1}},
        },
    }
    with pytest.raises(SchemaError, match=r"tb_lpf_ac\.a0_db.*'tb_lpf_opamp'"):
        validate_datasheet(data)


def test_validate_datasheet_rejects_measure_shared_by_two_testbenches() -> None:
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {
            "tb_a": {"gain": {"min": 1}},
            "tb_b": {"bw": {"min": 1}, "measure": {"gain": "bw * 2"}},
        },
    }
    with pytest.raises(SchemaError, match=r"tb_b\.gain"):
        validate_datasheet(data)


def test_validate_datasheet_allows_measure_for_its_own_value() -> None:
    """A measure: entry computing a value of the same testbench is one column."""
    from chipify.schema import validate_datasheet
    data = {
        "parameters": {"temp": [27]},
        "tests": {"tb_v": {"gain": {"min": 1}, "measure": {"gain": "v_out_"}}},
    }
    stim = validate_datasheet(data)
    assert [v.name for v in stim.tests[0].value_lst] == ["gain"]


@pytest.mark.parametrize("data, where", [
    ({"parameters": {"vdd": [1.8]},
      "tests": {"tb_x": {"vdd": {"min": 1}}}}, r"tb_x\.vdd.*input parameter"),
    ({"parameters": {"temp": [27]},
      "tests": {"tb_x": {"run_id": {"min": 1}}}}, r"tb_x\.run_id"),
    ({"parameters": {"temp": [27]},
      "tests": {"tb_x": {"gain": {"min": 1}, "gain_pass": {"min": 1}}}},
     r"tb_x\.gain_pass.*verdict column of 'gain'"),
    ({"parameters": {"sim_error": [0]},
      "tests": {"tb_x": {"gain": {"min": 1}}}}, r"parameters\.sim_error"),
    ({"parameters": {"tb_x__error": [0]},
      "tests": {"tb_x": {"gain": {"min": 1}}}}, r"error column of testbench 'tb_x'"),
])
def test_validate_datasheet_rejects_names_of_other_columns(data, where) -> None:
    from chipify.schema import validate_datasheet
    with pytest.raises(SchemaError, match=where):
        validate_datasheet(data)
