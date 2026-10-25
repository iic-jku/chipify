# Copyright (c) 2026 Santiago Hofwimmer
"""
schema.py – TypedDicts and validation helpers for the Chipify datasheet YAML.

TypedDicts define the expected structure of a parsed datasheet.yaml so that
editors and mypy can provide type-checked access to the data.

``validate_datasheet(data)`` is the single entry-point: it converts a raw
``yaml.safe_load()`` dict into validated Python objects, raising ``SchemaError``
with a precise error path on any structural or semantic problem.

Note: The heavy model classes (Stimuli, Test, Value) still live in util.py.
schema.py imports from util to build those objects; util.py must NOT import
from schema.py to avoid circular imports.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import numpy as np

log = logging.getLogger("chipify.schema")


# ── Exceptions ────────────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Raised when a datasheet.yaml violates the expected schema."""


def _supported_engines() -> tuple[str, ...]:
    """Valid per-testbench ``engine:`` values, from the engine registry.

    ``chipify.engines`` is import-light (no simulator modules are loaded just
    to list names), so datasheet validation stays cheap while still accepting
    drop-in engine plugins. Falls back to the built-in pair if the registry
    can't load for any reason.
    """
    try:
        from chipify.engines import engine_names
        return engine_names()
    except Exception:
        return ("ngspice", "vacask")


# ── Allowed range DSL functions ───────────────────────────────────────────────

_ALLOWED_CALLS: dict[str, Any] = {
    "range": range,
    "linspace": np.linspace,
    "logspace": np.logspace,
}


def _parse_range_dsl(value: str) -> list[int] | list[float]:
    """
    Parse a YAML parameter value that encodes a sequence via a safe DSL.

    Allowed forms
    -------------
    - ``range(N)``                 → list(range(N)), integers
    - ``range(start, stop)``       → list(range(start, stop)), integers
    - ``range(start, stop, step)`` → list(range(start, stop, step)), integers
    - ``linspace(start, stop, N)`` → list of N evenly spaced floats
    - ``logspace(start, stop, N)`` → list of N log-spaced floats

    Any other expression is rejected with ValueError (no eval() is used).

    Parameters
    ----------
    value:
        The raw string from the YAML file, e.g. ``"range(10)"`` or
        ``"linspace(0, 1, 5)"``.

    Returns
    -------
    list[int] | list[float]
        Evaluated sequence as a Python list. ``range`` stays integer, as its
        arguments are: as floats, ``seed: range(200)`` rendered
        ``.option SEED = 3.0`` and stored ``seed = 3.0`` in the results.

    Raises
    ------
    ValueError
        If the expression is not one of the allowed forms, uses unsupported
        node types, or contains non-numeric arguments.
    """
    try:
        tree = ast.parse(value.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in range DSL {value!r}: {exc}") from exc

    body = tree.body
    if not isinstance(body, ast.Call):
        raise ValueError(
            f"Range DSL {value!r}: top-level must be a function call "
            f"(range, linspace, logspace)."
        )

    # Resolve function name
    func_node = body.func
    if not isinstance(func_node, ast.Name):
        raise ValueError(
            f"Range DSL {value!r}: function must be a plain name, not an attribute."
        )
    func_name = func_node.id
    if func_name not in _ALLOWED_CALLS:
        raise ValueError(
            f"Range DSL {value!r}: function {func_name!r} is not allowed. "
            f"Use one of: {sorted(_ALLOWED_CALLS)!r}"
        )

    # No keyword args (starred args are rejected by the constant-only
    # argument check below).
    if body.keywords:
        raise ValueError(f"Range DSL {value!r}: keyword arguments are not supported.")

    # Extract numeric arguments — only constants allowed
    args: list[float | int] = []
    for i, arg in enumerate(body.args):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
            args.append(arg.value)
        elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            # Allow negation of constants: range(-5, 5)
            inner = arg.operand
            if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)):
                args.append(-inner.value)
            else:
                raise ValueError(
                    f"Range DSL {value!r}: argument {i} is not a numeric constant."
                )
        else:
            raise ValueError(
                f"Range DSL {value!r}: argument {i} is not a numeric constant."
            )

    # Call the allowed function
    fn = _ALLOWED_CALLS[func_name]
    if func_name == "range":
        # int() used to truncate silently: range(2.5) swept 0 and 1.
        fractional = [a for a in args if not float(a).is_integer()]
        if fractional:
            raise ValueError(
                f"Range DSL {value!r}: range() takes whole numbers, got "
                f"{fractional}; use linspace() for fractional steps."
            )
        try:
            return list(fn(*[int(a) for a in args]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Range DSL {value!r}: failed to evaluate range({args!r}): {exc}"
            ) from exc
    try:
        # linspace / logspace return numpy arrays.
        return [float(x) for x in fn(*args).tolist()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Range DSL {value!r}: failed to evaluate {func_name}({args!r}): {exc}"
        ) from exc


def _parse_scalar_value(raw: Any) -> float | None:
    """Convert a YAML scalar (str/int/float/None) to float or None.

    Raises SchemaError for values that are present but not numeric, so a typo
    like ``min: abc`` fails loudly instead of silently dropping the bound.
    """
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"expected a number, got {raw!r}") from exc


def validate_parameters(params_raw: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalise the ``parameters:`` block.

    Sequences expressed as range-DSL strings are expanded here. Scalars
    (numbers and non-DSL strings) are wrapped in one-element lists so that
    every parameter value is a sweepable sequence — ``generate_cases`` builds
    the case grid with ``itertools.product``, which would otherwise crash on
    a bare number or silently sweep a bare string character by character.

    Returns a dict mapping parameter names to lists of values.
    Raises SchemaError on malformed entries.
    """
    result: dict[str, Any] = {}
    for key, value in params_raw.items():
        if not isinstance(key, str):
            raise SchemaError(f"Parameter key must be a string, got {key!r}")
        if isinstance(value, str):
            # Only attempt DSL parsing for strings that look like function calls
            stripped = value.strip()
            if stripped and stripped[0].isalpha() and "(" in stripped:
                try:
                    result[key] = _parse_range_dsl(stripped)
                except ValueError as exc:
                    raise SchemaError(f"Parameter {key!r}: {exc}") from exc
            else:
                result[key] = [value]
        elif isinstance(value, list):
            result[key] = value
        else:
            result[key] = [value]
    return result


def _validate_description(data: dict[str, Any]) -> str:
    """The datasheet's optional one-line ``description:``.

    Shown on the report cover and on the batch report's summary page, so it is
    collapsed to a single line: a YAML block scalar would otherwise carry its
    line breaks into a table cell.
    """
    raw = data.get("description")
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        raise SchemaError(f"description: expected text, got {type(raw).__name__}")
    return " ".join(str(raw).split())


def _validate_equations(data: dict[str, Any], key: str) -> list[dict[str, str]]:
    """Validate a top-level equations block into ``[{"name", "expr"}]`` form.

    Accepts the natural YAML mapping (``name: expr``) as well as the legacy
    settings.json list-of-dicts shape. Names must be Python identifiers —
    they become DataFrame column names / plot signal names.
    """
    raw = data.get(key)
    if raw is None:
        return []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise SchemaError(f"{key}[{i}]: expected a mapping, got {entry!r}")
            pairs.append((entry.get("name", ""), entry.get("expr", "")))
    else:
        raise SchemaError(
            f"{key}: expected a mapping of name → expression, got {type(raw).__name__}"
        )

    out: list[dict[str, str]] = []
    for name, expr in pairs:
        name_s = str(name).strip()
        expr_s = str(expr).strip() if expr is not None else ""
        if not name_s.isidentifier():
            raise SchemaError(
                f"{key}.{name_s or '<empty>'}: equation name must be a valid "
                f"identifier (it becomes a result column)."
            )
        if not expr_s:
            raise SchemaError(f"{key}.{name_s}: expression must not be empty.")
        out.append({"name": name_s, "expr": expr_s})
    return out


def _validate_reports(data: dict[str, Any]) -> "Any":
    """Validate the top-level ``reports:`` block into a ``ReportsConfig``.

    Absent or empty yields an empty config, so datasheets that declare nothing
    are unaffected. Every error names the offending path, because a typo in a
    plot spec would otherwise only surface as a missing file much later.
    """
    from chipify.reports import (COMMON_KEYS, LATEX_FORMAT, PLOT_TYPES,
                                 PlotSpec, ReportsConfig)

    raw = data.get("reports")
    if raw is None:
        return ReportsConfig()
    if not isinstance(raw, dict):
        raise SchemaError(
            f"reports: expected a mapping, got {type(raw).__name__}."
        )

    def _formats(value: Any, path: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise SchemaError(f"{path}: expected a list of formats, "
                              f"got {type(value).__name__}.")
        known = _known_formats()
        out: list[str] = []
        for fmt in value:
            name = str(fmt).strip().lower().lstrip(".")
            if name not in known:
                raise SchemaError(
                    f"{path}: unknown format {name!r}. Available: "
                    f"{', '.join(sorted(known))}."
                )
            if name not in out:
                out.append(name)
        return out

    per_page = raw.get("plots_per_page")
    if per_page is None:
        per_page = 1
    # bool is an int subclass: `plots_per_page: true` must not read as 1.
    if isinstance(per_page, bool) or per_page not in (1, 2):
        raise SchemaError(
            f"reports.plots_per_page: expected 1 or 2, got {per_page!r}."
        )

    cfg = ReportsConfig(
        formats=_formats(raw.get("formats"), "reports.formats"),
        pdf=bool(raw.get("pdf", False)),
        markdown=bool(raw.get("markdown", False)),
        plots_per_page=int(per_page),
    )

    # Only an absent/null key defaults to "no plots": `plots: {}` is a typo for
    # a list and must be reported, not silently read as an empty one.
    plots_raw = raw.get("plots")
    if plots_raw is None:
        plots_raw = []
    if not isinstance(plots_raw, list):
        raise SchemaError(
            f"reports.plots: expected a list, got {type(plots_raw).__name__}."
        )

    for i, entry in enumerate(plots_raw):
        path = f"reports.plots[{i}]"
        if not isinstance(entry, dict):
            raise SchemaError(f"{path}: expected a mapping, got {entry!r}.")

        ptype = str(entry.get("type", "")).strip()
        if not ptype:
            raise SchemaError(f"{path}: missing required key 'type'.")
        if ptype not in PLOT_TYPES:
            raise SchemaError(
                f"{path}.type: unknown plot type {ptype!r}. Available: "
                f"{', '.join(sorted(PLOT_TYPES))}."
            )
        spec_type = PLOT_TYPES[ptype]

        allowed = set(COMMON_KEYS) | set(spec_type.required) | set(spec_type.optional)
        for key in entry:
            if key not in allowed:
                raise SchemaError(
                    f"{path}: unknown key {key!r} for plot type {ptype!r}. "
                    f"Allowed: {', '.join(sorted(allowed))}."
                )
        for key in spec_type.required:
            if not str(entry.get(key, "")).strip():
                raise SchemaError(
                    f"{path}: plot type {ptype!r} requires {key!r}."
                )

        # Validate the run selection against the same vocabulary the GUI uses,
        # so a typo is caught at load time rather than silently plotting all runs.
        if "runs" in entry:
            from chipify.uikit.services.transient_loader import parse_run_mode
            try:
                parse_run_mode(entry["runs"])
            except ValueError as exc:
                raise SchemaError(f"{path}.runs: {exc}") from None

        formats = _formats(entry.get("formats"), f"{path}.formats")
        if LATEX_FORMAT in formats and not spec_type.supports_latex:
            # Not fatal: report_service warns and still writes the other
            # formats. Failing the whole datasheet load over it would be
            # disproportionate for a presentational request.
            log.warning("%s: LaTeX export is not available for plot type %r; "
                        "it will be skipped.", path, ptype)

        options = {k: v for k, v in entry.items() if k not in COMMON_KEYS}
        cfg.plots.append(PlotSpec(
            type=ptype,
            name=str(entry.get("name", "")).strip(),
            formats=formats,
            options=options,
        ))

    return cfg


def _known_formats() -> set[str]:
    """Image extensions from the exporter registry, plus ``latex``."""
    from chipify.reports import LATEX_FORMAT
    try:
        from chipify.plugin_loader import get_exporter_plugins
        exts = {e.extension.lstrip(".").lower() for e in get_exporter_plugins()}
    except Exception:  # noqa: BLE001 — a broken user plugin must not block loading
        exts = {"png", "svg"}
    return exts | {LATEX_FORMAT}


def validate_datasheet(data: dict[str, Any]) -> "Any":  # returns util.Stimuli
    """
    Build and return a validated ``util.Stimuli`` object from a raw YAML dict.

    This is the authoritative entry-point for loading a datasheet.yaml.
    It replaces the ad-hoc parsing in ``Stimuli._load_from_yaml``.

    Raises
    ------
    SchemaError
        On structural violations (wrong types, missing required keys, bad
        range-DSL expressions).
    """
    from chipify.util import Stimuli, Test, Value  # local import to avoid circular dep

    if not isinstance(data, dict):
        raise SchemaError(f"Datasheet root must be a mapping, got {type(data).__name__}.")

    stim = Stimuli()
    stim.description = _validate_description(data)

    # ── Parameters ────────────────────────────────────────────────────────────
    params_raw = _mapping_block(data, ("parameters", "params", "sweep"),
                                "a mapping of parameter names to values")
    stim.params = validate_parameters(params_raw)

    # ── Custom equations (scalar + waveform) ──────────────────────────────────
    stim.equations = _validate_equations(data, "equations")
    stim.transient_equations = _validate_equations(data, "transient_equations")

    # ── Auto-generated plots / reports ────────────────────────────────────────
    stim.reports = _validate_reports(data)

    # ── Tests / Testbenches ───────────────────────────────────────────────────
    tests_raw = _mapping_block(data, ("tests", "testbenches", "measurements"),
                               "a mapping of testbenches")

    from chipify.analyses import SCHEMA_KEY_TO_CLASS, Analysis

    for tb_path, measurements in tests_raw.items():
        # An empty entry (`tb_x:` with nothing under it) is a testbench still
        # being written; run_sim refuses to dispatch one that declares nothing.
        if measurements is None:
            measurements = {}
        if not isinstance(measurements, dict):
            raise SchemaError(
                f"tests.{tb_path}: expected a mapping of measurements and "
                f"settings, got {type(measurements).__name__}"
            )

        analyses: list[Analysis] = []
        measure: dict[str, str] = {}
        engine: str | None = None
        netlist_source: str = "xschem"
        value_lst: list[Value] = []

        for val_name, bounds in measurements.items():
            if val_name in SCHEMA_KEY_TO_CLASS:
                if bounds is not None and not isinstance(bounds, list):
                    raise SchemaError(
                        f"tests.{tb_path}.{val_name}: expected a list of signals, "
                        f"got {type(bounds).__name__}"
                    )
                if bounds:
                    cls = SCHEMA_KEY_TO_CLASS[val_name]
                    analyses.append(cls(signals=[str(s) for s in bounds]))
                continue

            if val_name == "measure":
                if bounds is not None and not isinstance(bounds, dict):
                    raise SchemaError(
                        f"tests.{tb_path}.measure: expected a mapping of names "
                        f"to expressions, got {type(bounds).__name__}"
                    )
                measure = {k: str(v) for k, v in (bounds or {}).items()}
                continue

            if val_name == "engine":
                if bounds is not None:
                    eng = str(bounds).strip().lower()
                    supported = _supported_engines()
                    if eng and eng not in supported:
                        raise SchemaError(
                            f"tests.{tb_path}.engine: unknown engine {eng!r}; "
                            f"use one of {sorted(supported)}"
                        )
                    engine = eng or None
                continue

            if val_name == "source":
                if bounds is not None:
                    src = str(bounds).strip().lower()
                    allowed = {"xschem", "netlist"}
                    if src and src not in allowed:
                        raise SchemaError(
                            f"tests.{tb_path}.source: unknown source {src!r}; "
                            f"use one of {sorted(allowed)}"
                        )
                    netlist_source = src or "xschem"
                continue

            # Measurement boundary spec. Anything but a mapping used to mean "no
            # limits": `gain: [9, 11]` was a measurement that always passed.
            if bounds is None:
                bounds = {}
            if not isinstance(bounds, dict):
                raise SchemaError(
                    f"tests.{tb_path}.{val_name}: expected a mapping such as "
                    f"{{min: ..., max: ...}}, got {type(bounds).__name__}"
                )
            try:
                vmin = _parse_scalar_value(bounds.get("min", bounds.get("vmin")))
                vmax = _parse_scalar_value(bounds.get("max", bounds.get("vmax")))
                vtyp = _parse_scalar_value(bounds.get("typ", bounds.get("vtyp")))
            except SchemaError as exc:
                raise SchemaError(
                    f"tests.{tb_path}.{val_name}: {exc}"
                ) from exc
            unit_raw = bounds.get("unit", bounds.get("units"))
            unit = str(unit_raw).strip() or None if unit_raw is not None else None
            value_lst.append(
                Value(name=str(val_name), vmin=vmin, vmax=vmax, vtyp=vtyp, unit=unit)
            )

        t = Test(tb_path, value_lst)
        t.analyses = analyses
        t.measure = measure
        t.engine = engine
        t.netlist_source = netlist_source
        stim.addTest(t)

    _check_result_columns(stim)
    return stim


def _mapping_block(data: dict[str, Any], keys: tuple[str, ...],
                   expected: str) -> dict[str, Any]:
    """The first of *keys* present in *data*, which must be a mapping.

    An empty block (``tests:`` with nothing under it) reads as an empty
    mapping. Any other type used to be skipped without a word, so a list where
    the testbench mapping belongs became a plan with no testbenches at all.
    """
    for key in keys:
        if key not in data:
            continue
        block = data[key]
        if block is None:
            return {}
        if not isinstance(block, dict):
            raise SchemaError(f"{key}: expected {expected}, got {type(block).__name__}")
        return block
    return {}


def _check_result_columns(stim: Any) -> None:
    """Reject datasheet names that would share one column of the results.

    Results are stored by name alone, one ``sample[name]`` per run. Two
    testbenches declaring the same measurement shared a column: the later one
    overwrote the earlier value while its error scope still pointed at the
    first testbench. A measurement named like an input parameter, or a name
    equal to one of chipify's own columns, overwrote that column the same way.
    """
    from chipify.data_loader import RESERVED_COLUMNS, TB_PASS_SUFFIX, tb_error_col

    owners: dict[str, str] = {}
    for test in stim.tests:
        # A measure: entry may compute a value of the same testbench.
        names = dict.fromkeys([v.name for v in test.value_lst] + list(test.measure))
        for name in names:
            owner = owners.setdefault(name, test.tb_path)
            if owner != test.tb_path:
                raise SchemaError(
                    f"tests.{test.tb_path}.{name}: testbench {owner!r} already "
                    f"declares {name!r}. Results are stored by measurement name, "
                    f"so each testbench needs its own (for example with a "
                    f"testbench prefix)."
                )

    generated = {col: "a column chipify writes itself" for col in RESERVED_COLUMNS}
    for test in stim.tests:
        generated[tb_error_col(test.tb_path)] = (
            f"the error column of testbench {test.tb_path!r}")
        generated[test.tb_path + TB_PASS_SUFFIX] = (
            f"the verdict column of testbench {test.tb_path!r}")
        for val in test.value_lst:
            generated[f"{val.name}_pass"] = f"the verdict column of {val.name!r}"

    for name, tb_path in owners.items():
        if name in stim.params:
            raise SchemaError(
                f"tests.{tb_path}.{name}: same name as the input parameter {name!r}."
            )
        if name in generated:
            raise SchemaError(f"tests.{tb_path}.{name}: same name as {generated[name]}.")
    for name in stim.params:
        if name in generated:
            raise SchemaError(f"parameters.{name}: same name as {generated[name]}.")
