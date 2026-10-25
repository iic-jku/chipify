# Copyright (c) 2026 Santiago Hofwimmer
"""
abort.py – File-based abort flag shared by the orchestrator and all engines.

The flag lives in FAST_TMP (namespaced per project, see settings.py) so the
GUI/CLI process can stop worker processes it cannot signal directly: workers
poll :func:`is_aborted` between progress ticks and while their simulator
subprocess runs.
"""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

from chipify import settings

log = logging.getLogger("chipify.engines.abort")

ABORT_FLAG_PATH = Path(settings.FAST_TMP) / "abort.flag"


def parent_gone() -> bool:
    """True in a worker process whose chipify process has died.

    A killed ``chipify-cli`` (or a stopped container) left its pool workers
    running: they finished their case and went on to the cases already queued
    for them, each starting another simulator. Workers poll this through
    :func:`is_aborted`, so it takes the same path as the abort flag: the
    running simulator is killed and no further case starts. Always False in
    the main process, which has no multiprocessing parent.
    """
    parent = multiprocessing.parent_process()
    return parent is not None and not parent.is_alive()


def is_aborted() -> bool:
    return ABORT_FLAG_PATH.exists() or parent_gone()


def clear_abort_flag() -> None:
    try:
        ABORT_FLAG_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def abort_simulation() -> None:
    log.info("abort_simulation() called – writing abort flag.")
    try:
        ABORT_FLAG_PATH.write_text("abort", encoding="utf-8")
    except OSError as exc:
        log.error("Could not write abort flag: %s", exc)
