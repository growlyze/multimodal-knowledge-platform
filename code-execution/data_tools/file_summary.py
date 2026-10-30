"""describe_file() / format_file_summary(): every fact a file summary may state.

Measured the same day the spec behind this module was written: most replies
about a real client file missed its routine closures, several called files
"clean" with no check behind the word, and loader mis-parses were described
as facts. Leaving those checks to the model's judgement failed, so they are
computed here, from the data only, and printed for the model to quote:

  F1 every number comes from this function's output;
  F2 no "clean"/"no gaps" unless the check that would disprove it printed 0;
  F3 every defect - gaps of ANY kind - is listed in `problems` as a plain
     sentence;
  F4 a file the loader could not read correctly is reported as unreadable,
     not described;
  F5 "columns" means the raw header including the time column, and every
     time says its zone.
"""

import os
import re

import numpy as np
import pandas as pd

from .loader import load_file
from .logging_decorator import logged

GAP_KINDS = ("short_gap", "long_gap")
# A gap is every row spacing above the median. Classification is purely by
# duration: under 24h is a short gap, 24h or more is a long gap (a closure, a
# weekend, missing days). No wall-clock schedule is assumed - this module has
# no idea when whatever produced the data was open.
_LONG_GAP = pd.Timedelta("24h")

# load_issues sentences describe_file() states itself, from its own counts -
# repeated verbatim they would double every defect in `problems`.
_RESTATED = (
    "repeat a timestamp", "not in chronological order", "could not be read as",
)
# load_issues that mean the file was NOT read correctly (F4).
_UNREADABLE = (
    "file not found", "file could not be read", "could not be decompressed",
    "could not be parsed as datetime", "implausible date range",
    "no timestamp unit fits", "clock times only",
)


def _human_interval(seconds):
    s = int(round(seconds))
    for unit, size in (("D", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        if s >= size and s % size == 0:
            return "%d%s" % (s // size, unit)
    return "%ds" % s


def _human_span(seconds):
    """'1 day 22 hours 20 minutes' - the whole span, every non-zero part."""
    s = int(round(seconds))
    parts = []
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        k, s = divmod(s, size)
        if k:
            parts.append("%d %s%s" % (k, unit, "" if k == 1 else "s"))
    return " ".join(parts) or "0 minutes"


def _human_minutes(minutes):
    if minutes >= 2 * 1440 and minutes % 1440 == 0:
        return "%d days" % (minutes // 1440)
    if minutes >= 180:
        return ("%.1f h" % (minutes / 60)).replace(".0 h", " h")
    return "%g min" % minutes


def _gaps(index):
    """(median interval, gap list) for a sorted, de-duplicated index."""
    try:
        diffs = index.to_series().diff().dropna()
    except OverflowError:
        # A span wider than ~292 years does not fit a nanosecond timedelta;
        # the implausible-range warning already says the times are wrong.
        return None, []
    if not len(diffs):
        return None, []
    med = diffs.median()
    spots = [(nxt - d, nxt, d) for nxt, d in diffs[diffs > med].items()]
    gaps = [{"start": prev.isoformat(), "end": nxt.isoformat(),
             "minutes": d.total_seconds() / 60,
             "missing_minutes": (d - med).total_seconds() / 60,
             "kind": "long_gap" if d >= _LONG_GAP else "short_gap"}
            for prev, nxt, d in spots]
    return med, gaps


def _gap_sentence(gaps):
    """One plain sentence naming every gap's kind, count, where and how long."""
    parts = []
    by_kind = {k: [g for g in gaps if g["kind"] == k] for k in GAP_KINDS}

    def when(items, limit=4):
        shown = ", ".join("%s (%s)" % (g["start"][:16].replace("T", " "),
                                        _human_minutes(g["minutes"])) for g in items[:limit])
        return shown + (" and %d more" % (len(items) - limit) if len(items) > limit else "")

    labels = {"short_gap": "short gaps (under 24h)",
              "long_gap": "long gaps (24h or more)"}
    for kind in GAP_KINDS:
        items = by_kind[kind]
        if items:
            label = labels[kind] if len(items) > 1 else labels[kind].replace("gaps", "gap")
            parts.append("%d %s at %s" % (len(items), label, when(items)))
    return "%d gap%s: %s" % (len(gaps), "" if len(gaps) == 1 else "s", "; ".join(parts))


_TIME_NAME_RE = re.compile(r"time|date", re.I)
# Enough of the column must parse for its type to be claimed; below that it
# is text. Chosen so a mostly-numeric column with a few corrupt cells is
# still typed "numbers" WITH the corrupt cells counted, never silently typed
# "text" (which would read as "these were never numbers").
_MAJORITY = 0.5


def _num(x):
    """A number as the summary prints it: up to 2 decimals, no trailing zeros."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return ("%.2f" % float(x)).rstrip("0").rstrip(".")


def _profile_column(col, s):
    """One column's entry: type, missing, stats, and corrupt-cell count.

    type is "numbers", "dates" or "text"; `unparseable` counts non-null
    values that did not fit the claimed type (the corrupt rows F3 requires to
    be surfaced).
    """
    entry = {"name": str(col), "missing": int(s.isna().sum()), "unparseable": 0}
    non_null = s.notna()

    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = s
        entry.update(type="dates", first=parsed.min().isoformat(),
                     last=parsed.max().isoformat())
        return entry

    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        entry.update(type="numbers",
                     min=float(v.min()) if v.notna().any() else None,
                     max=float(v.max()) if v.notna().any() else None,
                     mean=float(v.mean()) if v.notna().any() else None,
                     sum=float(v.sum()))
        return entry

    # Object columns: infer from the values, not the name alone.
    coerced = pd.to_numeric(s, errors="coerce")
    if non_null.any() and coerced[non_null].notna().mean() >= _MAJORITY:
        entry.update(type="numbers",
                     min=float(coerced.min()), max=float(coerced.max()),
                     mean=float(coerced.mean()), sum=float(coerced.sum()))
        entry["unparseable"] = int((coerced.isna() & non_null).sum())
        return entry

    if _TIME_NAME_RE.search(str(col)):
        parsed = pd.to_datetime(s, errors="coerce")
        if non_null.any() and parsed[non_null].notna().mean() >= _MAJORITY:
            entry.update(type="dates", first=parsed.min().isoformat(),
                         last=parsed.max().isoformat())
            entry["unparseable"] = int((parsed.isna() & non_null).sum())
            return entry

    counts = s.astype(str).where(non_null).value_counts()
    entry.update(type="text", distinct=int(counts.size),
                 top={str(k): int(v) for k, v in counts.head(5).items()})
    return entry


@logged
def describe_file(path_or_df):
    """Every fact a file summary may state, computed from the data only.

    Args:
        path_or_df: a file path (read with load_file) or an already-loaded
            DataFrame (optionally with a DatetimeIndex).

    Returns a dict: file, readable, kind ("series" when rows are spaced in
    time, else "table"), rows, raw_columns, n_columns, columns (per-column
    type/missing/stats/unparseable), first_ts, last_ts, tz, interval,
    interval_seconds, gaps (start/end/minutes/missing_minutes/kind with kind
    "short_gap" < 24h or "long_gap" >= 24h), gap_counts_by_kind,
    span_seconds (first row to last row), expected_rows (rows a gap-free span
    would have: span / interval + 1, only for regular intervals, else None),
    rows_missing_in_gaps, duplicates, sorted, comments, load_issues, problems
    (one plain sentence per defect). Print format_file_summary(info) and
    quote it; never describe a file from anything else. Gaps are every row
    spacing above the median - listed with their kind, not hidden. Never
    raises.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
        name = (df.attrs.get("source") or {}).get("name") or None
    else:
        df = load_file(path_or_df)
        name = os.path.basename(str(path_or_df))
    issues = list(df.attrs.get("load_issues") or [])
    dated = isinstance(df.index, pd.DatetimeIndex) and len(df) > 0
    kind = "series" if dated else "table"

    # load_file() records the file's own header in attrs["raw_columns"];
    # load_dataset() does not, so this fallback reads the post-load frame -
    # which may carry columns the loader added and named in load_issues.
    raw_columns = df.attrs.get("raw_columns") or (
        ([str(df.index.name or "index")] if dated else [])
        + [str(c) for c in df.columns])

    info = {"file": name, "readable": True, "kind": kind, "rows": int(len(df)),
            "raw_columns": list(raw_columns), "n_columns": len(raw_columns),
            "columns": [],
            "first_ts": None, "last_ts": None, "tz": None,
            "interval": None, "interval_seconds": None,
            "gaps": [], "gap_counts_by_kind": {},
            "duplicates": 0, "sorted": True,
            "span_seconds": None, "expected_rows": None, "rows_missing_in_gaps": None,
            "comments": list(df.attrs.get("file_comments") or []),
            "load_issues": issues, "problems": []}
    problems = info["problems"]

    if any(marker in i for i in issues for marker in _UNREADABLE):
        info["readable"] = False
    if not len(df):
        if info["readable"]:
            problems.append("the file has no data rows, so nothing about its "
                            "contents can be stated")
        problems.extend(i for i in issues if "no rows" not in i)
        return info

    frame = df.sort_index(kind="mergesort") if dated else df
    zone = df.attrs.get("file_tz")
    if dated and zone is not None and frame.index.tz is not None:
        # Times in the file's own wall clock: a fixed-offset file parsed to
        # UTC otherwise shows its own 09:30 as 13:30.
        frame = frame.copy()
        frame.index = frame.index.tz_convert(zone)

    for col in frame.columns:
        entry = _profile_column(col, frame[col])
        info["columns"].append(entry)
        if entry["missing"]:
            problems.append("%d missing value(s) in %r" % (entry["missing"], str(col)))
        if entry["unparseable"]:
            problems.append("%d value(s) in %r could not be read as %s - "
                            "corrupt rows" % (entry["unparseable"], str(col),
                                              "numbers" if entry["type"] == "numbers" else "dates"))

    if dated:
        index = frame.index
        info["first_ts"], info["last_ts"] = index[0].isoformat(), index[-1].isoformat()
        info["duplicates"] = int(index.duplicated().sum())
        info["sorted"] = bool(df.attrs.get("file_order_sorted", True)
                              and df.index.is_monotonic_increasing)
        med, gaps = _gaps(index.unique())
        info["tz"] = (str(zone) + " (the file's own wall clock)" if zone is not None
                      else str(index.tz) if index.tz is not None
                      else "none stated in the file (naive times)")
        if med is not None:
            info["interval_seconds"] = med.total_seconds()
            info["interval"] = _human_interval(med.total_seconds())
        info["gaps"] = gaps
        for g in gaps:
            info["gap_counts_by_kind"][g["kind"]] = info["gap_counts_by_kind"].get(g["kind"], 0) + 1
        # The span and the gap-free row count are printed so they are copied,
        # not recomputed by hand. The count only where it means something:
        # regular intervals whose every spacing is a whole number of them. On
        # irregular spacing "span / interval + 1" counts nothing.
        try:
            info["span_seconds"] = (index[-1] - index[0]).total_seconds()
        except (OverflowError, ValueError):   # wider than a Timedelta holds
            info["span_seconds"] = (index[-1].to_pydatetime()
                                    - index[0].to_pydatetime()).total_seconds()
        if med is not None and pd.Timedelta(0) < med < pd.Timedelta("1D"):
            uniq = index.unique()
            steps = uniq.to_series().diff().dropna() / med
            if len(steps) and np.allclose(steps, np.round(steps)):
                info["expected_rows"] = int(round((uniq[-1] - uniq[0]) / med)) + 1
                info["rows_missing_in_gaps"] = int(round(sum(g["missing_minutes"] for g in gaps)
                                                         * 60 / med.total_seconds()))
        if info["duplicates"]:
            problems.append("%d duplicate timestamp(s): rows that repeat a time another "
                            "row already has" % info["duplicates"])
    else:
        info["duplicates"] = int(frame.duplicated().sum())
        if info["duplicates"]:
            problems.append("%d row(s) are exact duplicates of an earlier row"
                            % info["duplicates"])
        if any("no recognisable time column" in i for i in issues):
            issues.remove(next(i for i in issues if "no recognisable time column" in i))
            info["load_issues"] = issues

    if not info["sorted"]:
        problems.append("the rows were not in time order in the file (they were sorted on load)")
    if dated and all(e["missing"] == info["rows"] for e in info["columns"]) and info["columns"]:
        info["readable"] = False
        problems.append("every value is missing - the file was not read correctly, "
                        "so none of its contents can be described")

    # ---- problems: one sentence per defect (F3). Gaps of every kind count.
    if info["gaps"]:
        problems.append(_gap_sentence(info["gaps"]))
    problems.extend(i for i in issues if not any(m in i for m in _RESTATED))
    return info


def format_file_summary(info):
    """Render describe_file()'s dict as compact text for a model to quote.

    Prints every check WITH its result, zeros included, so an absence ("no
    gaps") is only ever stated as the outcome of a check that ran (F2).
    """
    lines = []
    name = info.get("file") or "(data frame)"
    if not info.get("readable", True):
        lines.append("FILE COULD NOT BE READ CORRECTLY: %s - do not describe its contents; "
                     "report the problems below instead." % name)
    else:
        lines.append("File: %s" % name)
    if info.get("kind") == "table":
        lines.append("Kind: table (no time column - nothing time-based is stated below)")
    cols = info.get("raw_columns") or []
    # Every name: "... (71 more)" hid most of a wide export from the model.
    shown = ", ".join(cols)
    lines.append("Rows: %d | Columns in the file (raw header, time column included): %d - %s"
                 % (info.get("rows", 0), info.get("n_columns", len(cols)), shown))
    if info.get("comments"):
        lines.append("Comment lines skipped before the header: %s" % " | ".join(info["comments"]))
    if info.get("first_ts"):
        lines.append("Time range: %s to %s | Time zone: %s"
                     % (info["first_ts"], info["last_ts"], info.get("tz")))
        if info.get("span_seconds"):          # 0 on a one-row file: nothing to state
            lines.append("Time span: %s (%s hours, first row to last row)" % (
                _human_span(info["span_seconds"]), _num(info["span_seconds"] / 3600)))
        lines.append("Row interval: %s (median spacing between rows)"
                     % (info.get("interval") or "n/a (one row)"))
        exp = info.get("expected_rows")
        if exp is not None:
            unique_rows = info.get("rows", 0) - info.get("duplicates", 0)
            fewer = exp - unique_rows
            n_gaps = len(info.get("gaps") or [])
            where = (", all inside the %d gap%s reported below" % (n_gaps, "" if n_gaps == 1 else "s")
                     if fewer > 0 and fewer == info.get("rows_missing_in_gaps") else "")
            lines.append("Rows: %d %s | %d if no row in the span were missing "
                         "(span / %s + 1) | %d fewer%s" % (
                             unique_rows,
                             "unique timestamps in the file (%d rows)" % info.get("rows", 0)
                             if info.get("duplicates") else "in the file",
                             exp, info.get("interval"), fewer, where))
    gaps = info.get("gaps") or []
    if info.get("rows") and info.get("first_ts"):
        if gaps:
            kinds = ", ".join("%s %d" % (k, n) for k, n in info["gap_counts_by_kind"].items())
            lines.append("Gaps (spacing above the %s median): %d - %s"
                         % (info.get("interval"), len(gaps), kinds))
            listed = sorted(gaps, key=lambda g: -g["minutes"])[:20]
            for g in listed:
                lines.append("  - %s -> %s  %s  %s" % (g["start"], g["end"],
                                                       _human_minutes(g["minutes"]), g["kind"]))
            if len(listed) < len(gaps):
                lines.append("  (%d more not listed here, counted by kind above; every one is in "
                             "describe_file()['gaps'])" % (len(gaps) - len(listed)))
        else:
            lines.append("Gaps (spacing above the median): 0 - no gaps")
    elif info.get("rows"):
        lines.append("Gaps: 0 gaps checked - there are no timestamps to check")
    lines.append("Columns:")
    for c in info.get("columns") or []:
        if c["type"] == "numbers":
            desc = "numbers, min %s, max %s, mean %s, sum %s" % (
                _num(c["min"]), _num(c["max"]), _num(c["mean"]), _num(c["sum"]))
        elif c["type"] == "dates":
            desc = "dates, %s to %s" % (c["first"], c["last"])
        else:
            desc = "text, %d distinct: %s" % (c["distinct"],
                                              ", ".join("%s %d" % kv for kv in c["top"].items()))
        extra = "" if not c.get("unparseable") else " | %d corrupt" % c["unparseable"]
        lines.append("  - %s: %s | missing %d%s" % (c["name"], desc, c["missing"], extra))
    if info.get("kind") == "series":
        lines.append("Duplicate timestamps: %d | Sorted in file: %s" % (
            info.get("duplicates", 0), "yes" if info.get("sorted", True) else "no"))
    else:
        lines.append("Duplicate rows: %d" % info.get("duplicates", 0))
    problems = info.get("problems") or []
    if problems:
        lines.append("Problems found (%d) - report every one:" % len(problems))
        lines.extend("  - %s" % p for p in problems)
    else:
        lines.append("Problems found: 0 (every check above ran and found nothing)")
    return "\n".join(lines)
