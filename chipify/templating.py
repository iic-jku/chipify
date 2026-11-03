# Copyright (c) 2026 Santiago Hofwimmer
"""
templating.py – The one way chipify renders a netlist template for a sample.

The simulation workers and the per-sample netlist export render the same
templates, so they share this function rather than two Jinja2 configurations
that could drift apart.
"""
from __future__ import annotations

from typing import Any, Mapping

from jinja2 import StrictUndefined, Template


def render_netlist(template: str, values: Mapping[str, Any]) -> str:
    """Render a netlist *template* with one sample's *values*.

    ``StrictUndefined``: a placeholder the sample does not define raises
    instead of rendering as empty text. ``keep_trailing_newline``: Jinja2
    drops a template's final newline by default, and VACASK rejects a deck
    whose last line has no newline with a syntax error at end of file.
    """
    # Jinja2 types Template() as Any, so the annotation keeps strict mypy's
    # no-any-return quiet (newer mypy reports it, 1.14 did not).
    rendered: str = Template(
        template, undefined=StrictUndefined, keep_trailing_newline=True,
    ).render(dict(values))
    return rendered
