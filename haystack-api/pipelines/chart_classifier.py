"""Chart/non-chart frame classifier — flat-background + edge_ratio heuristic."""

import logging

import cv2
import numpy as np

logger = logging.getLogger("haystack-api.chart_classifier")

# A chart is mostly one flat background colour -- near-black on a dark UI
# theme, grey-white on a light theme -- with bars/lines on top. This
# used to test "dark_ratio > 0.40" instead, which only ever matched the dark
# theme: a light-theme chart scores dark 0.01-0.12 and every one was skipped
# by OCR and Gemini (2026-10-19, P141).
BACKGROUND_RATIO_THRESHOLD = 0.40
BACKGROUND_BAND = 10          # grey levels either side of the modal value
EDGE_RATIO_THRESHOLD = 0.015


def is_chart_frame(frame_bgr: np.ndarray) -> bool:
    """True if the frame looks like a screen chart (flat bg + visible edges)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256)
    mode = int(np.argmax(hist))
    band = hist[max(0, mode - BACKGROUND_BAND):mode + BACKGROUND_BAND + 1].sum()
    background_ratio = float(band) / gray.size
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    return (background_ratio > BACKGROUND_RATIO_THRESHOLD
            and edge_ratio > EDGE_RATIO_THRESHOLD)


def classify_frames(frames: list) -> list:
    """Batch convenience — returns list of bools parallel to input."""
    return [is_chart_frame(f) for f in frames]


def _self_test():
    """Synthetic frame tests — no real images needed."""
    h, w = 360, 640

    # 1. Solid dark frame — high dark_ratio but no edges → not chart
    dark = np.zeros((h, w, 3), dtype=np.uint8)
    assert not is_chart_frame(dark), "solid dark should not be chart (no edges)"
    print("ok 1 — solid dark rejected (no edges)")

    # 2. Solid white frame — low dark_ratio → not chart
    white = np.full((h, w, 3), 255, dtype=np.uint8)
    assert not is_chart_frame(white), "solid white should not be chart"
    print("ok 2 — solid white rejected")

    # 3. Dark frame with grid lines — both thresholds met → chart
    chart = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 20):
        cv2.line(chart, (0, y), (w, y), (0, 200, 0), 1)
    for x in range(0, w, 30):
        cv2.line(chart, (x, 0), (x, h), (0, 200, 0), 1)
    # Add some bar-like rectangles
    for x in range(50, w - 50, 60):
        cv2.rectangle(chart, (x, 100), (x + 10, 250), (0, 255, 0), -1)
    assert is_chart_frame(chart), "dark frame with grid+bars should be chart"
    print("ok 3 — dark + grid + bars accepted")

    # 4. Light-theme chart -- the light UI theme. The old
    # check asserted the opposite ("light + edges rejected") and so certified the
    # bug: 1 chart in 106 keyframes on a real lesson, 0 in 419 on another
    # (2026-10-19), which skipped OCR and Gemini on every one of those charts.
    light_chart = np.full((h, w, 3), (228, 222, 219), dtype=np.uint8)
    for y in range(0, h, 40):
        cv2.line(light_chart, (0, y), (w, y), (205, 200, 198), 1)
    for i, x in enumerate(range(20, w - 20, 14)):
        top = 120 + int(60 * np.sin(i / 4))
        colour = (0, 0, 0) if i % 3 else (110, 170, 90)
        cv2.line(light_chart, (x + 3, top - 15), (x + 3, top + 60), colour, 1)
        cv2.rectangle(light_chart, (x, top), (x + 6, top + 40), colour, -1)
    cv2.putText(light_chart, "Upper Band", (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 140), 1)
    assert is_chart_frame(light_chart), "light-theme chart must be accepted"
    print("ok 4 — light-theme chart accepted")

    # 5. Camera-like frame -- no flat background anywhere -> not a chart,
    # whatever its brightness. A full-range gradient plus noise has edges but no
    # dominant colour (blurred noise alone won't do: it averages to flat grey).
    rng = np.random.default_rng(0)
    ramp = np.tile(np.linspace(0, 255, w), (h, 1))[..., None].repeat(3, axis=2)
    photo = np.clip(ramp + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    assert not is_chart_frame(photo), "frame with no flat background should not be chart"
    print("ok 5 — no flat background rejected")

    # 6. classify_frames batch
    results = classify_frames([dark, white, chart, light_chart, photo])
    assert results == [False, False, True, True, False], f"batch: {results}"
    print("ok 6 — classify_frames batch")

    print("\n6 checks passed")


if __name__ == "__main__":
    _self_test()
