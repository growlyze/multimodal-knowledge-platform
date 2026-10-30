"""Shared logging infrastructure for data_tools.

Every public function in this package is wrapped with @logged. The decorator
records the call, its timing and a compact summary of inputs/outputs so that a
sandbox run can be audited after the fact.

Failures inside a tool are expected to be handled *by the tool* (return an
empty DataFrame / a blank Figure and log a warning) - see the "graceful errors"
requirement. The decorator only logs-and-reraises genuine programming errors so
they are visible instead of silently swallowed.
"""

import functools
import logging
import time
from uuid import uuid4

import pandas as pd

logger = logging.getLogger("data_tools")

MAX_REPR = 200


def logged(func):
    """Log every call to `func` with timing and input/output summaries."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        call_id = uuid4().hex[:8]
        name = func.__name__

        logger.info(
            "[%s] CALL %s | args=%s kwargs=%s",
            call_id, name, _summarize_args(args), _summarize_args(kwargs),
        )

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "[%s] FAIL %s | %.2fs | %s: %s",
                call_id, name, elapsed, type(exc).__name__, exc,
            )
            raise
        elapsed = time.perf_counter() - start
        logger.info(
            "[%s] OK %s | %.2fs | result=%s",
            call_id, name, elapsed, _summarize(result),
        )
        return result

    return wrapper


def _summarize(obj):
    """Compact, cheap string form of an object for log lines."""
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame({len(obj)} rows, cols={list(obj.columns)[:5]})"
    if isinstance(obj, pd.Series):
        return f"Series({len(obj)} values, name={obj.name})"
    if isinstance(obj, dict):
        return f"dict(keys={list(obj.keys())[:5]})"
    if isinstance(obj, (list, tuple, set)):
        return f"{type(obj).__name__}(len={len(obj)})"
    return str(obj)[:MAX_REPR]


def _summarize_args(args):
    if isinstance(args, tuple):
        return tuple(_summarize(a) for a in args)
    if isinstance(args, dict):
        return {k: _summarize(v) for k, v in args.items()}
    return _summarize(args)


def missing_columns(df, required, where=""):
    """Return the required columns absent from `df`, logging a warning if any.

    Used by tools to bail out gracefully instead of raising KeyError. Also
    treats a non-DataFrame or empty frame as "everything missing".
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("%s: expected a non-empty DataFrame, got %s", where, _summarize(df))
        return list(required)
    absent = [c for c in required if c not in df.columns]
    if absent:
        logger.warning("%s: missing required columns %s", where, absent)
    return absent
