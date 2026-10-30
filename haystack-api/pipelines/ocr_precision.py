"""
OCR numeric-value precision for chart and screen keyframes (axis values,
figures, prices): value check, cross-frame consistency and confidence flags
on every OCR reading.

Why this exists
---------------
Client rule (the client, 2026-10-27): "a wrong number presented as certain is worse
than a missing one". Until this module, ocr_track.labels_to_text() flattened
every reading of a keyframe to one string and threw away confidence and bbox,
so a misread such as the axis label 28,220.00 read as "20,220.00" (measured on
the 2026-10-27 restest set, 1080p, confidence 0.93) reached the index -- and
from there an answer -- looking exactly as certain as a correct price.

This module never CORRECTS a value (decided 2026-10-27: flag only). It assigns
each reading a status:

    ok            -- nothing wrong found (also every non-price reading)
    confirmed     -- the same price was read in >= 2 consecutive chart keyframes
    uncertain     -- OCR confidence below UNCERTAIN_BELOW, or a malformed
                     number ("28.100.00": dots used as thousand separators)
    out_of_range  -- the price lies outside the frame's price range (derived
                     from the price axis, or from the median/IQR of the frame's
                     prices when no axis is found), or sits on the axis column
                     far from the price its y position implies
    inconsistent  -- the same level (same position, same digit count, one
                     digit different) was read differently in other frames and
                     this reading is the minority

Precedence when several apply: out_of_range > inconsistent > uncertain >
confirmed > ok. A repeated misread is never "confirmed" past a range flag.

MARKER CONTRACT (team answer-precision / rag_filter depend on it EXACTLY)
-------------------------------------------------------------------------
In the flat OCR text that is embedded and stored as the Qdrant payload field
`ocr_text`, every reading whose status is NOT "ok" and NOT "confirmed" is
rendered as its original text followed by one space and the literal marker
`[uncertain]` -- e.g. ``20,220.00 [uncertain]``. One marker string for all
three flag statuses: the downstream rule is binary ("do not present this
number as certain"), and the detailed status stays in `ocr_labels`. The marker
follows the WHOLE reading, so a multi-number reading ("O28,194.25 H28,1...")
is marked once after its last number. The marker text is MARKER below;
changing it breaks the filter rule silently -- do not.

Structured data goes to the keyframe's payload as `ocr_labels` (additive;
`ocr_text` and every other field unchanged): a list of
{"text", "value", "values", "conf", "bbox", "status"} in reading order.
`value` is the first price in the reading (None when none), `values` all of
them.

Points ingested before this module existed carry no markers and no
`ocr_labels`: their absence means "not assessed", not "verified".
"""

from __future__ import annotations

import logging
import re
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("haystack-api.ocr_precision")

# The exact marker downstream code matches on. See MARKER CONTRACT above.
MARKER = "[uncertain]"

FLAG_STATUSES = ("uncertain", "out_of_range", "inconsistent")
_PRECEDENCE = {"ok": 0, "confirmed": 1, "uncertain": 2, "inconsistent": 3, "out_of_range": 4}

# Calibrated 2026-10-28 on the 30-frame restest set (60 images, 720p+1080p,
# 2,004 hand-scored price readings): the lowest-confidence CORRECT price was
# 0.819 (1080p) / 0.847 (720p); 0.75 leaves a margin under both so a correct
# label is not flagged for confidence, while the 0.45-0.75 band that
# ocr_track's min_score still keeps is flagged. Confidence alone CANNOT catch
# the known misreads (20,220.00 scored 0.93 at 1080p, 78,013.75 0.99 at 720p)
# -- the range and consistency checks do that.
UNCERTAIN_BELOW = 0.75

# Range margin as a fraction of the axis span visible on screen. Labels and
# header OHLC can sit slightly outside the plotted area; a one-digit misread
# of a 5-digit value is off by >= 1,000, many spans away.
AXIS_MARGIN = 0.25
# Axis-column readings must sit near the price their y implies. Price tags
# (last price, line labels) overlap the axis grid by up to a label height.
AXIS_RESIDUAL_STEPS = 2.0
# Fallback (no axis): Tukey fences at 3*IQR, IQR floored at 1% of the median
# so a frame with near-identical prices does not flag a normal one.
IQR_K = 3.0

# A price: comma-thousands (28,013.75 / 1,234) or plain decimal (28013.75 /
# 1.0850). The malformed form (28.100.00) is caught separately.
_NUM_RE = re.compile(r"(?<![\d])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)+|\d+\.\d+)")


def _parse_number(tok: str):
    """Return (value, malformed) for one numeric token, (None, False) if none."""
    parts = re.split(r"[.,]", tok)
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
        return float(tok.replace(",", "")), False
    if re.fullmatch(r"\d+\.\d+", tok):
        return float(tok), False
    if len(parts) >= 3:
        # "28.100.00" / "28.013,75": several separators in a non-standard
        # layout. Read all but the last as thousands; mark malformed so it is
        # flagged -- a misplaced separator is exactly a digit-level misread.
        try:
            return float("".join(parts[:-1]) + "." + parts[-1]), True
        except ValueError:
            return None, False
    if len(parts) == 2 and "," in tok:
        # "28476,75" -- the OCR read the decimal point as a comma (restest
        # 1080p 13:03, an execution label). A comma followed by exactly two
        # digits cannot be a thousands separator, so the value is unambiguous:
        # not flagged (flagging it was a false alarm on a key label). Any
        # other digit count is ambiguous -> malformed.
        try:
            return float(parts[0] + "." + parts[1]), len(parts[1]) != 2
        except ValueError:
            return None, False
    return None, False


def parse_prices(text: str):
    """Return ([values], malformed_any) for every price-like number in text.

    Times (01:30), percentages (0.5%) and bare integers (77) are not prices.
    """
    vals: List[float] = []
    bad = False
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(1)
        end = m.end(1)
        if end < len(text) and text[end] == "%":
            continue
        v, malformed = _parse_number(tok)
        if v is None:
            continue
        vals.append(v)
        bad = bad or malformed
    return vals, bad


def mark_occluded(labels: List[Dict[str, Any]], bgr: Any) -> List[Dict[str, Any]]:
    """Tag labels whose text box is crossed by a solid dark bar (a filled bar shape).

    P5 104-frame test 2026-10-28: "1 @ 28129.00" had its "9" half under a black bar and
    was read 28128.00 in every frame - consistent, on the tick grid, in range, so no text
    rule could see it. On a light chart background a run of >= 6 columns that are >= 60%
    near-black inside the box is a filled bar, not text (strokes stay under ~30%). Measured on
    3,447 priced readings: catches that misread, 3 correct key labels flagged (safe side).
    Dark-theme boxes (median < 120) are skipped. Never raises.
    """
    try:
        import numpy as np
        gray = bgr if getattr(bgr, "ndim", 3) == 2 else bgr.mean(axis=2)
        H, W = gray.shape[:2]
        for lab in labels:
            x0, y0, x1, y1 = [int(round(v)) for v in lab.get("bbox", [0, 0, 0, 0])]
            box = gray[max(y0 - 2, 0):min(y1 + 3, H), max(x0 - 2, 0):min(x1 + 3, W)]
            if box.size == 0 or float(np.median(box)) < 120:
                continue
            solid = (box < 60).mean(axis=0) >= 0.6
            run = best = 0
            for c in solid:
                run = run + 1 if c else 0
                best = max(best, run)
            if best >= 6:
                lab["occluded"] = True
    except Exception:  # degrade: no tag rather than a broken ingestion
        pass
    return labels


def _make_reading(label: Dict[str, Any]) -> Dict[str, Any]:
    vals, malformed = parse_prices(label.get("text", ""))
    return {
        "text": label.get("text", ""),
        "value": vals[0] if vals else None,
        "values": vals,
        "conf": round(float(label.get("score", label.get("conf", 0.0))), 4),
        "bbox": list(label.get("bbox", [0, 0, 0, 0])),
        "status": "ok",
        "_malformed": malformed,
        "_axis": False,
        "_occluded": bool(label.get("occluded")),
        # P5 2026-10-28: "0.75 (21,805" (Fib label cut at its box edge) was certain.
        # An unclosed bracket around a price means the text was cut mid-number.
        "_truncated": bool(vals) and label.get("text", "").count("(") > label.get("text", "").count(")"),
        # "(28,160,75)": the number regex stops at "28,160" and leaves ",75" - a decimal
        # point read as a comma, so the parsed value is wrong (28,160.00).
        "_comma_decimal": bool(re.search(r"\d,\d{3},\d{2}(?!\d)", label.get("text", ""))),
        # Decimal point only: "27,797" has a thousands comma ("7,79") that must not count.
        # "28476,75" (decimal read as a comma, value right) has decimals too; "27,797" does not.
        "_whole": bool(vals) and not re.search(r"\d\.\d{2}(?!\d)|(?<![\d,])\d{4,6},\d{2}(?!\d)",
                                                label.get("text", "")),
        # "O27,797 75": decimal point read as a space -> 27,797 parsed and "75" left over.
        "_space_decimal": bool(re.search(r"\d,\d{3} \d{2}(?!\d)", label.get("text", ""))),
    }


def _set(r: Dict[str, Any], status: str) -> None:
    if _PRECEDENCE[status] > _PRECEDENCE[r["status"]]:
        r["status"] = status


def _theil_sen(points):
    """Robust (slope, intercept) of value = a*y + b; None if < 3 points."""
    if len(points) < 3:
        return None
    slopes = [(v2 - v1) / (y2 - y1)
              for i, (y1, v1) in enumerate(points)
              for (y2, v2) in points[i + 1:] if abs(y2 - y1) > 1e-6]
    if not slopes:
        return None
    a = median(slopes)
    b = median(v - a * y for y, v in points)
    return a, b


def find_axis(readings, width: int, height: int):
    """Detect the price axis: single-price readings in a narrow column at the
    right (or left) edge whose values fall linearly as y grows.

    Returns dict(a, b, step, col=(x0, x1), members=set(ids)) or None.
    """
    edge = 0.12 * width
    best = None
    for side in ("right", "left"):
        cands = []
        for i, r in enumerate(readings):
            if len(r["values"]) != 1:
                continue
            x0, y0, x1, y1 = r["bbox"]
            if side == "right" and x1 < width - edge:
                continue
            if side == "left" and x0 > edge:
                continue
            cands.append((i, (y0 + y1) / 2.0, r["values"][0]))
        if len(cands) < 3:
            continue
        fit = _theil_sen([(y, v) for _, y, v in cands])
        if fit is None or fit[0] >= 0:
            continue                    # prices must fall as y grows
        a, b = fit
        # Grid step in price: median gap between neighbouring well-fitting
        # labels. Tolerance for "on the line" is that many steps.
        res = [abs(v - (a * y + b)) for _, y, v in cands]
        ys = sorted(y for _, y, _ in cands)
        dy = median([y2 - y1 for y1, y2 in zip(ys, ys[1:]) if y2 - y1 > 1] or [1.0])
        step = abs(a) * dy
        members = {i for (i, _, _), e in zip(cands, res) if e <= AXIS_RESIDUAL_STEPS * step}
        if len(members) < 3:
            continue
        if best is None or len(members) > len(best["members"]):
            best = dict(a=a, b=b, step=step, members=members,
                        cands={i for i, _, _ in cands}, side=side)
    return best


def assess_frame(labels: Sequence[Dict[str, Any]], width: int, height: int,
                 uncertain_below: float = UNCERTAIN_BELOW) -> List[Dict[str, Any]]:
    """Per-frame checks (value range + confidence) on one keyframe's labels.

    `labels` is ocr_track.extract_labels() output (text/score/bbox).
    Returns readings in reading order (top-to-bottom, left-to-right).
    """
    readings = [_make_reading(l) for l in labels]
    readings.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    priced = [r for r in readings if r["values"]]
    if not priced:
        return readings

    axis = find_axis(readings, width, height) if width and height else None
    if axis:
        a, b = axis["a"], axis["b"]
        top, bottom = b, a * height + b          # value at y=0 and y=H
        lo, hi = min(top, bottom), max(top, bottom)
        margin = AXIS_MARGIN * (hi - lo)
        lo, hi = lo - margin, hi + margin
        ref = (lo + hi) / 2.0
        for i in axis["members"]:
            # On the fitted axis line = value confirmed by its own position.
            # "28.020.00" (dot for comma) is a frequent axis misread whose
            # value is right; flagging it was 1-in-3 of all false alarms.
            readings[i]["_malformed"] = False
            readings[i]["_axis"] = True
        for i in axis["cands"] - axis["members"]:
            r = readings[i]
            yc = (r["bbox"][1] + r["bbox"][3]) / 2.0
            if abs(r["values"][0] - (a * yc + b)) > AXIS_RESIDUAL_STEPS * axis["step"] * 4:
                # On the axis column but nowhere near where its y says --
                # the 20,220.00-at-28,220-height case.
                _set(r, "out_of_range")
    else:
        allv = sorted(v for r in priced for v in r["values"])
        med = median(allv)
        q1 = allv[len(allv) // 4]
        q3 = allv[(3 * len(allv)) // 4]
        iqr = max(q3 - q1, 0.01 * abs(med))
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        ref = med

    # Tick grid (P5 2026-10-28): these prices move in fixed 0.25 tick steps. When nearly every
    # price on the frame sits on that grid, an off-grid price ("27,619.59" from two stacked
    # Fib labels read as one) is a misread, however confident and repeated it is.
    def _on_grid(v):
        return abs(round(v * 4) - v * 4) < 1e-6
    _prices = [v for r in priced for v in r["values"] if not (ref and abs(v) < 0.01 * abs(ref))]
    tick_chart = len(_prices) >= 8 and sum(_on_grid(v) for v in _prices) >= 0.9 * len(_prices)
    _with_dec = [r for r in priced if not r.get("_whole")]
    decimal_chart = len(priced) >= 4 and len(_with_dec) >= 0.8 * len(priced)
    for r in priced:
        for v in r["values"]:
            # Ratios, fib levels (0.618), percentages-without-% are not prices
            # on a chart whose prices are 100x larger: skip, never flag.
            if ref and abs(v) < 0.01 * abs(ref):
                continue
            if v < lo or v > hi:
                _set(r, "out_of_range")
        if (r["_malformed"] or r.get("_truncated") or r.get("_comma_decimal")
                or r.get("_space_decimal") or r.get("_occluded")):
            _set(r, "uncertain")
        # A whole number on a chart whose prices are written with 2 decimals is a
        # price cut short ("28,240" for 28,240.75) - only when most prices carry them.
        if r.get("_whole") and decimal_chart and any(
                not (ref and abs(v) < 0.01 * abs(ref)) for v in r["values"]):
            _set(r, "uncertain")
        if tick_chart and any(not _on_grid(v) for v in r["values"]
                              if not (ref and abs(v) < 0.01 * abs(ref))):
            _set(r, "uncertain")
        if r["conf"] < uncertain_below:
            _set(r, "uncertain")
    return readings


def _is_drawn(text: str) -> bool:
    """A label drawn on the chart (execution "@", Fib "(price)") - static between frames."""
    return "@" in text or "(" in text


def _label_prefix(text: str) -> str:
    """Text before the first price-sized number, normalised ("1 @ 28129.00" -> "1@")."""
    m = re.search(r"\d{2},?\d{3}", text)
    return re.sub(r"\s+", "", text[:m.start()] if m else text).lower()


def _one_digit_apart(a: float, b: float) -> bool:
    sa, sb = "%.2f" % a, "%.2f" % b
    return len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) == 1


def assess_sequence(frames: List[List[Dict[str, Any]]], height: Optional[int] = None,
                    width: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """Cross-frame consistency over consecutive chart keyframes of one video.

    Mutates and returns `frames` (each a list from assess_frame()).
    - A price read (exactly) in >= 2 frames is "confirmed" (unless flagged).
    - Readings of one level -- near the same position (x centre within 4% of
      width, y within 2 label heights) and one digit apart in value -- that
      disagree: the majority value is confirmed, the minority "inconsistent".
      A 1-vs-1 tie flags both: without a majority there is no evidence which
      read is right, and the flag-over-guess rule prefers a flag to a guess.
    """
    seen: Dict[float, int] = {}
    for fr in frames:
        for v in {v for r in fr for v in r["values"]}:
            seen[v] = seen.get(v, 0) + 1

    # Axis grid labels are excluded: they are validated in-frame by the axis
    # line fit, and when the chart scrolls or rescales a grid label at the
    # same screen height legitimately changes by one digit (29,200 -> 29,400)
    # -- treating that as a misread flagged every axis label (calibration run
    # 1: 437 false alarms out of 1,038 correct prices at 1080p).
    flat = [(fi, r) for fi, fr in enumerate(frames) for r in fr if r["value"] is not None
            and len(r["values"]) == 1 and not r.get("_axis")]
    for fi, r in flat:
        x0, y0, x1, y1 = r["bbox"]
        h = max(y1 - y0, 1)
        same_level = {}
        for fj, s in flat:
            if fj == fi:
                continue
            sx0, sy0, sx1, sy1 = s["bbox"]
            if width and abs((sx0 + sx1) - (x0 + x1)) / 2.0 > 0.04 * width:
                continue
            if not width and abs((sx0 + sx1) - (x0 + x1)) / 2.0 > 3 * h:
                continue
            if abs((sy0 + sy1) - (y0 + y1)) / 2.0 > 2 * h:
                continue
            # A rival must be one digit apart AND >= 1% away: a misread digit
            # in the leading places (28,013.75 vs 78,013.75 / 20,013.75). A
            # last-price tag that ticks 28,013.75 -> 28,018.75 between
            # keyframes is a real move, not a disagreement.
            # Drawn labels - executions ("1 @ 28129.00") and Fib levels ("0.5 (28,070.50)") -
            # never change between frames, so ANY one-digit disagreement is a misread
            # (P5 104 frames: "1 @ 28128.00" for 28129.00, last digit under a bar).
            drawn = _is_drawn(r["text"]) and _is_drawn(s["text"])
            if s["value"] == r["value"] or (
                    _one_digit_apart(s["value"], r["value"])
                    and (drawn or abs(s["value"] - r["value"]) >= 0.01 * max(abs(r["value"]), 1e-9))):
                same_level.setdefault(s["value"], set()).add(fj)
        rivals = {v: fs for v, fs in same_level.items() if v != r["value"]}
        if rivals:
            mine = len(same_level.get(r["value"], set())) + 1
            best_rival = max(len(fs) for fs in rivals.values())
            if mine <= best_rival:
                _set(r, "inconsistent")
    for fr in frames:
        for r in fr:
            if r["values"] and r["status"] == "ok" and all(seen.get(v, 0) >= 2 for v in r["values"]):
                r["status"] = "confirmed"
    return frames


def render_text(readings: Sequence[Dict[str, Any]]) -> str:
    """Flat OCR text for embedding/search, with MARKER after flagged readings.

    Same reading order and separator as ocr_track.labels_to_text(), so an
    unflagged frame produces byte-identical text to the old path.
    """
    out = []
    for r in readings:
        out.append(r["text"] + (" " + MARKER if r["status"] in FLAG_STATUSES else ""))
    return " ".join(out)


def payload_labels(readings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The `ocr_labels` payload list: public fields only, JSON-safe."""
    return [{k: r[k] for k in ("text", "value", "values", "conf", "bbox", "status")}
            for r in readings]


def read_frames(bgr_frames: Sequence[Any], min_score: Optional[float] = None,
                uncertain_below: float = UNCERTAIN_BELOW) -> List[List[Dict[str, Any]]]:
    """OCR + assess consecutive frames (numpy BGR). Offline entry point for
    test harnesses: no Qdrant, no network. One frame -> pass [frame].
    Returns one reading list per frame (see payload_labels for the fields).
    """
    from pipelines.ocr_track import DEFAULT_MIN_SCORE, extract_labels
    ms = DEFAULT_MIN_SCORE if min_score is None else min_score
    frames, w, h = [], None, None
    for f in bgr_frames:
        h, w = f.shape[:2]
        frames.append(assess_frame(mark_occluded(extract_labels(f, min_score=ms), f), w, h,
                                   uncertain_below))
    assess_sequence(frames, height=h, width=w)
    return [payload_labels(fr) for fr in frames]


def read_frame(bgr_frame: Any, **kw) -> List[Dict[str, Any]]:
    """Single-frame form of read_frames() (no cross-frame confirmation)."""
    return read_frames([bgr_frame], **kw)[0]


# ---------------------------------------------------------------------------
# Offline self-test: python pipelines/ocr_precision.py (PYTHONPATH=.)
# ---------------------------------------------------------------------------

def _axis_labels(top=28460.0, step=20.0, n=26, x=1658, y0=76, dy=34.6):
    return [{"text": "{:,.2f}".format(top - k * step), "score": 0.999,
             "bbox": [x, int(y0 + k * dy), x + 68, int(y0 + k * dy) + 21]} for k in range(n)]


def _self_test() -> None:
    n = 0

    def ok(name):
        nonlocal n
        n += 1
        print("ok", name)

    assert parse_prices("28,013.75") == ([28013.75], False)
    assert parse_prices("28013.75") == ([28013.75], False)
    assert parse_prices("O28,194.25 H28,196.75")[0] == [28194.25, 28196.75]
    assert parse_prices("01:30 PM 77 0.5%") == ([], False)
    assert parse_prices("27.600.00") == ([27600.0], True)
    ok("parse-prices")

    # The real 1080p misread (restest frame 15:29): 20,220.00 printed on the
    # axis column at the height of 28,220. Must be out_of_range + marked.
    labels = _axis_labels() + [
        {"text": "20,220.00", "score": 0.9345, "bbox": [1660, 500, 1725, 517]},
        {"text": "28,227.00", "score": 0.9891, "bbox": [1659, 485, 1726, 506]},
        {"text": "1 (28,007.75)", "score": 0.99, "bbox": [400, 860, 500, 880]},
        {"text": "0.618", "score": 0.99, "bbox": [300, 400, 340, 420]},
    ]
    rs = assess_frame(labels, 1728, 1080)
    st = {r["text"]: r["status"] for r in rs}
    assert st["20,220.00"] == "out_of_range", st
    assert st["28,227.00"] == "ok" and st["1 (28,007.75)"] == "ok" and st["0.618"] == "ok", st
    assert all(s == "ok" for t, s in st.items() if t.endswith(".00") and t != "20,220.00")
    ok("axis-value-check-flags-20220")

    txt = render_text(rs)
    assert "20,220.00 [uncertain]" in txt and txt.count(MARKER) == 1, txt
    assert MARKER == "[uncertain]"
    ok("marker-contract")

    # Unflagged frame renders exactly like the old labels_to_text().
    import importlib
    try:
        from pipelines.ocr_track import labels_to_text
        clean = _axis_labels()
        assert render_text(assess_frame(clean, 1728, 1080)) == labels_to_text(clean)
        ok("unflagged-text-identical-to-old-path")
    except ImportError:          # cv2 missing on this machine
        print("skip unflagged-text-identical-to-old-path (cv2 not installed)")

    # Low confidence -> uncertain; non-price low-confidence text untouched.
    rs = assess_frame(_axis_labels() + [
        {"text": "28,100.50", "score": 0.6, "bbox": [500, 700, 560, 715]},
        {"text": "P95", "score": 0.5, "bbox": [500, 300, 530, 315]}], 1728, 1080)
    st = {r["text"]: r["status"] for r in rs}
    assert st["28,100.50"] == "uncertain" and st["P95"] == "ok", st
    ok("low-confidence-price-uncertain")

    # No axis -> median/IQR fallback.
    rs = assess_frame([
        {"text": t, "score": 0.99, "bbox": [100 + 90 * k, 300, 170 + 90 * k, 315]}
        for k, t in enumerate(["28,013.75", "28,050.25", "27,990.00", "28,120.00", "78,013.75"])],
        1728, 1080)
    st = {r["text"]: r["status"] for r in rs}
    assert st["78,013.75"] == "out_of_range" and st["28,013.75"] == "ok", st
    ok("no-axis-iqr-fallback")

    # Malformed separators are flagged, never silently read as 28.1.
    rs = assess_frame(_axis_labels() + [
        {"text": "28.100.00", "score": 0.95, "bbox": [500, 900, 560, 915]}], 1728, 1080)
    assert [r["status"] for r in rs if r["text"] == "28.100.00"] == ["uncertain"]
    ok("malformed-number-uncertain")

    # P5 test 2026-10-28 (Shotgun Saturday Feb 01, 4 frames): the Fib label
    # "0.75 (21,805.25)" was cut at its box edge and read "0.75 (21,805" -> 21,805.00,
    # CERTAIN, and "confirmed" because every frame cut it the same way. A number in an
    # unclosed bracket, or a whole number on a chart whose prices carry decimals, is
    # incomplete text: uncertain, never certain.
    labs2 = _axis_labels() + [
        {"text": "0.75 (28,105", "score": 0.97, "bbox": [600, 300, 660, 312]},
        {"text": "0.5 (28,300.25)", "score": 0.97, "bbox": [600, 330, 680, 342]},
        {"text": "28,240", "score": 0.97, "bbox": [700, 400, 740, 412]},
    ]
    rs2 = {r["text"]: r["status"] for r in assess_frame(labs2, 1728, 1080)}
    assert rs2["0.75 (28,105"] == "uncertain", rs2
    assert rs2["0.5 (28,300.25)"] in ("ok", "confirmed"), rs2
    assert rs2["28,240"] == "uncertain", rs2
    seq2 = assess_sequence([assess_frame([dict(l) for l in labs2], 1728, 1080) for _ in range(3)])
    assert all(r["status"] != "confirmed" for fr in seq2 for r in fr
               if r["text"] in ("0.75 (28,105", "28,240")), seq2
    ok("truncated-number-uncertain")

    # P5 test 2026-10-28, video "Predicting Session Low & High With Executions": stacked
    # Fib labels ("0.5 (27,619.50)" drawn over "0.625 (27,617.75)") were read as one blended
    # label "0.025 (27,619.59)" at 0.9+ confidence, identically in every frame -> confirmed.
    # these prices move in fixed 0.25 tick steps: when nearly every price in the frame sits on that
    # grid, an off-grid price is a misread. And "(28,160,75)" (decimal read as a comma) was
    # parsed as 28,160.00 - a number followed by ",dd" is malformed.
    labs3 = _axis_labels() + [
        {"text": ".025 (28,019.59)", "score": 0.97, "bbox": [600, 300, 700, 312]},
        {"text": "-0,125 (28,160,75)", "score": 0.97, "bbox": [600, 330, 700, 342]},
        {"text": "0.5 (28,070.50)", "score": 0.97, "bbox": [600, 360, 700, 372]},
    ]
    rs3 = {r["text"]: r["status"] for r in assess_frame(labs3, 1852, 1080)}
    assert rs3[".025 (28,019.59)"] == "uncertain", rs3
    assert rs3["-0,125 (28,160,75)"] == "uncertain", rs3
    assert rs3["0.5 (28,070.50)"] in ("ok", "confirmed"), rs3
    ok("off-tick-and-comma-decimal-uncertain")

    # P5 104-frame run 2026-10-28: (a) execution label "1 @ 28129.00" read "1 @ 28128.00" in
    # one frame (last digit half under a chart bar) while 3 other frames read 28129.00 - the 1%
    # rule meant for moving price tags let it through. A drawn label ("@" execution, "(...)"
    # Fib) never changes, so any one-digit disagreement is a misread. (b) header "O27,797 75":
    # the decimal point came out as a space -> 27,797 parsed, and the thousands comma fooled
    # the whole-number check.
    def lab(t, y=500):
        return {"text": t, "score": 0.97, "bbox": [800, y, 900, y + 12]}
    frames4 = [assess_frame(_axis_labels() + [lab("1 @ 28129.00")], 1728, 1080) for _ in range(3)]
    frames4.append(assess_frame(_axis_labels() + [lab("1 @ 28128.00")], 1728, 1080))
    seq4 = assess_sequence(frames4, height=1080, width=1728)
    st4 = [r["status"] for fr in seq4 for r in fr if "@" in r["text"]]
    assert st4[-1] == "inconsistent" and st4[0] == "confirmed", st4
    tags = [assess_frame(_axis_labels() + [lab("28,013.75", 900)], 1728, 1080),
            assess_frame(_axis_labels() + [lab("28,018.75", 900)], 1728, 1080)]
    assert all(r["status"] != "inconsistent" for fr in assess_sequence(tags, 1080, 1728)
               for r in fr if r["text"].startswith("28,01")), "a moving price tag must not be flagged"
    rs5 = {r["text"]: r["status"] for r in assess_frame(_axis_labels() + [lab("O28,197 75", 60)], 1728, 1080)}
    assert rs5["O28,197 75"] == "uncertain", rs5
    rs6 = {r["text"]: r["status"] for r in assess_frame(_axis_labels() + [lab("28476,75", 200)], 1728, 1080)}
    assert rs6["28476,75"] in ("ok", "confirmed"), rs6     # comma-read decimal, value right
    ok("drawn-label-one-digit-and-space-decimal-uncertain")

    import numpy as np
    img = np.full((1080, 1728, 3), 200, dtype=np.uint8)
    img[650:672, 255:268] = 10                       # a filled bar across the label's last digit
    labs7 = mark_occluded([{"text": "1 @ 28128.00", "score": 0.97, "bbox": [210, 654, 269, 668]},
                           {"text": "1 @ 28085.00", "score": 0.97, "bbox": [210, 754, 269, 768]}], img)
    rs7 = {r["text"]: r["status"] for r in assess_frame(_axis_labels() + labs7, 1728, 1080)}
    assert rs7["1 @ 28128.00"] == "uncertain" and rs7["1 @ 28085.00"] == "ok", rs7
    ok("bar-over-label-uncertain")

    # Cross-frame: one level read 28,013.75 in 3 frames and 78,013.75 in 1.
    def tag(t, conf=0.99):
        return {"text": t, "score": conf, "bbox": [10, 60, 66, 77]}
    seq = [assess_frame([tag(t)], 1728, 1080)
           for t in ["28,013.75", "28,013.75", "78,013.75", "28,013.75"]]
    assess_sequence(seq, height=1080, width=1728)
    st = [fr[0]["status"] for fr in seq]
    assert st == ["confirmed", "confirmed", "inconsistent", "confirmed"], st
    ok("cross-frame-majority-confirmed-minority-inconsistent")

    # Seen once only -> stays ok; a flag beats confirmation.
    seq = [assess_frame([tag("28,013.75")], 1728, 1080), assess_frame([tag("28,500.00")], 1728, 1080)]
    assess_sequence(seq, height=1080, width=1728)
    assert [fr[0]["status"] for fr in seq] == ["ok", "ok"]
    seq = [assess_frame([tag("28,013.75", 0.5)], 1728, 1080), assess_frame([tag("28,013.75")], 1728, 1080)]
    assess_sequence(seq, height=1080, width=1728)
    assert [fr[0]["status"] for fr in seq] == ["uncertain", "confirmed"]
    ok("confirmation-needs-two-frames-and-never-overrides-flag")

    pl = payload_labels(seq[1])
    assert pl == [{"text": "28,013.75", "value": 28013.75, "values": [28013.75], "conf": 0.99,
                   "bbox": [10, 60, 66, 77], "status": "confirmed"}], pl
    ok("payload-labels-shape")

    print("%d checks passed" % n)


if __name__ == "__main__":
    _self_test()
