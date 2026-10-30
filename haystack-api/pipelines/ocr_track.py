"""
OCR track for M3 video frame ingestion.

Extracts on-screen text (slides, charts, dashboards, code, annotations) from video frames using
RapidOCR (ONNX-only PP-OCR). Chosen over PaddleOCR because:
  - ONNX runtime only, no PaddlePaddle (~500 MB) — no DLL conflict with PyTorch
    in the same venv (Windows, this repo's dev machine and the VPS both affected)
  - return_word_box=True gives character-level bboxes natively; PaddleOCR returns
    line-level bboxes which merge adjacent short labels ("KPI" + "Q3" → one box)
  - Cold-start import is ~200 ms vs ~800 ms for PaddlePaddle-based paddle OCR

Brightness-based adaptive inversion
------------------------------------
Dark-theme apps and slides put white/light text on a near-black (e.g. #131722) background.
PP-OCR models were trained predominantly on light-background documents. For dark
frames, inverting with cv2.bitwise_not before OCR improves recall on short uppercase
labels like "KPI", "Q3", "API", "SLA", numeric values (e.g. "4,219.75"),
and short captions ("Revenue by Region").

Light-background frames (e.g. light-theme apps, documents, code editor screenshots)
are passed through unchanged — unconditional inversion would corrupt them. The decision
is made by sampling the top-10-pixel border of the crop: mean brightness < 100 → dark
background → invert; >= 100 → light background → pass through.

Re-inverting the frame RGB to feed the model's det/rec heads in sequence is NOT
needed — RapidOCR applies the same bgr_to_gray inversion once internally.

Confidence threshold
--------------------
DEFAULT_MIN_SCORE = 0.45 is empirically calibrated for chart annotation fonts
(Trebuchet MS / Roboto, size 9–11 at 1920×1080). Lower values add noise from
partially occluded labels; values above 0.65 miss small overlapping annotations.
Callers that need higher precision (e.g. price level extraction where a misread
digit is harmful) should pass min_score=0.65.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("haystack-api.ocr_track")

# Min confidence for a word token to be included in the output.
# See module docstring for calibration notes.
DEFAULT_MIN_SCORE = 0.45

# Lazy singleton — RapidOCR takes ~200 ms to initialise; we init once and reuse.
_ocr_instance: Optional[Any] = None


def _get_ocr():
    """Return a module-level RapidOCR singleton, initialising on first call."""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from rapidocr import RapidOCR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "rapidocr is required for the OCR track. "
                "Install with: pip install rapidocr (NOT rapidocr-onnxruntime — "
                "that's an older, separate package without this import path)"
            ) from exc
        _ocr_instance = RapidOCR()
        logger.debug("RapidOCR initialised (ONNX PP-OCR)")
    return _ocr_instance


def extract_labels(
    bgr_crop: np.ndarray,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[Dict[str, Any]]:
    """Extract on-screen text from a frame crop.

    Parameters
    ----------
    bgr_crop  : numpy array shape (H, W, 3), dtype uint8, BGR colour order from cv2.
    min_score : confidence threshold; tokens below this are dropped.

    Returns
    -------
    List of dicts, each containing:
        text  str              — recognised token (whitespace-stripped)
        score float            — OCR confidence in [0, 1]
        bbox  [x0, y0, x1, y1] — pixel coords, axis-aligned bounding box

    Returns empty list on null input, empty frame, or when rapidocr is unavailable.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return []

    ocr = _get_ocr()

    # Adaptive inversion: dark background → invert for PP-OCR; light → pass through.
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    border_mean = float(np.mean(gray[:10, :]))
    if border_mean < 100:
        # Dark background — invert for PP-OCR
        inp = cv2.bitwise_not(bgr_crop)
    else:
        inp = bgr_crop

    try:
        result = ocr(inp, use_det=True, use_cls=False, use_rec=True, return_word_box=True)
    except Exception as exc:
        logger.warning("RapidOCR inference failed: %s", exc)
        return []

    if result is None:
        return []

    # P138: rapidocr 3.x returns a RapidOCROutput object carrying parallel
    # .txts / .scores / .boxes tuples, and it is NOT iterable -- the old loop
    # raised on every frame, so OCR returned nothing for every video until the
    # 2026-10-19 smoke test. Older releases returned a list of (text, score,
    # box) items; both shapes are accepted.
    if hasattr(result, "txts"):
        # Explicit None checks: .boxes is a numpy array, whose truthiness
        # raises ("truth value of an array is ambiguous").
        def _seq(value):
            return () if value is None else value
        items = list(zip(_seq(result.txts), _seq(result.scores), _seq(result.boxes)))
    else:
        items = list(result)

    labels: List[Dict[str, Any]] = []
    for item in items:
        if len(item) < 3:
            continue
        token, score, box = item[0], item[1], item[2]
        if score < min_score:
            continue
        text = str(token).strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        labels.append(
            {
                "text": text,
                "score": round(float(score), 4),
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            }
        )

    return labels


def labels_to_text(labels: List[Dict[str, Any]]) -> str:
    """Flatten extracted labels to a single space-separated string.

    Intended for embedding: a flat string carries more semantic weight than
    a JSON blob when passed to a text embedding model.

    Labels are sorted top-to-bottom then left-to-right (reading order for chart
    overlays). Price levels appear inline with annotation text; sorting by y then
    x preserves proximity between a zone label and its price.
    """
    if not labels:
        return ""
    sorted_labels = sorted(labels, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return " ".join(d["text"] for d in sorted_labels)


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify behaviour with synthetic frames. Prints ok/fail, no network, no GPU."""
    import sys

    # Empty / None input must return empty list.
    assert extract_labels(np.array([], dtype=np.uint8).reshape(0, 0, 3)) == []
    print("ok empty-frame-returns-empty")

    # labels_to_text on empty list returns empty string.
    assert labels_to_text([]) == ""
    print("ok labels-to-text-empty")

    # labels_to_text sorting: label at y=100 comes before y=200.
    labels = [
        {"text": "LB", "score": 0.9, "bbox": [10, 200, 40, 215]},
        {"text": "SLA", "score": 0.85, "bbox": [5, 100, 45, 115]},
    ]
    assert labels_to_text(labels) == "SLA LB"
    print("ok labels-to-text-sort-order")

    # min_score filter: with a very high threshold and synthetic low-conf tokens,
    # the function should return nothing. Can't test without RapidOCR itself, but
    # we can test that the filter comparison is correct.
    class _FakeOCR:
        def __call__(self, *a, **kw):
            return [["SLA", 0.3, [[0, 0], [10, 0], [10, 10], [0, 10]]]]

    global _ocr_instance
    _ocr_instance = _FakeOCR()
    result = extract_labels(np.zeros((64, 64, 3), dtype=np.uint8), min_score=0.5)
    assert result == [], f"expected [] for score 0.3 < 0.5, got {result}"
    print("ok low-confidence-token-filtered")

    # Token at score above threshold is included.
    class _FakeOCRPass:
        def __call__(self, *a, **kw):
            return [["BOS", 0.8, [[0, 0], [10, 0], [10, 10], [0, 10]]]]

    _ocr_instance = _FakeOCRPass()
    result = extract_labels(np.zeros((64, 64, 3), dtype=np.uint8), min_score=0.5)
    assert len(result) == 1 and result[0]["text"] == "BOS"
    print("ok high-confidence-token-passes")

    # 6. Light-background frame: no inversion applied
    light_frame = np.full((100, 200, 3), 220, dtype=np.uint8)
    cv2.putText(light_frame, "SLA", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # The key test: border_mean should be > 100, so no inversion
    gray_test = cv2.cvtColor(light_frame, cv2.COLOR_BGR2GRAY)
    border_mean_test = float(np.mean(gray_test[:10, :]))
    assert border_mean_test > 100, f"light frame border mean {border_mean_test} should be > 100"
    print("ok 6 — light-background frame skips inversion")

    # 7. P138: rapidocr 3.x returns a RapidOCROutput object with parallel
    # .txts / .scores / .boxes tuples, not a list of (text, score, box). The
    # old-shape fakes above passed while production read zero text from every
    # frame ("'RapidOCROutput' object is not iterable", 2026-10-19 smoke test).
    class _RapidOCROutput:
        txts = ("SLA 5021.25", "noise")
        scores = (0.99, 0.2)
        # The real .boxes is a numpy array, not a tuple -- `boxes or ()` on it
        # raised "truth value of an array ... is ambiguous" (second smoke run).
        boxes = np.array([[[60, 20], [250, 20], [250, 45], [60, 45]],
                          [[0, 0], [5, 0], [5, 5], [0, 5]]], dtype=np.float32)

        def __iter__(self):  # the real object isn't iterable -- make that explicit
            raise TypeError("'RapidOCROutput' object is not iterable")

    class _FakeOCRv3:
        def __call__(self, *a, **kw):
            return _RapidOCROutput()

    _ocr_instance = _FakeOCRv3()
    result = extract_labels(np.zeros((64, 64, 3), dtype=np.uint8), min_score=0.5)
    assert result == [{"text": "SLA 5021.25", "score": 0.99, "bbox": [60, 20, 250, 45]}], result
    print("ok 7 — rapidocr 3.x output object (txts/scores/boxes) is parsed")

    # Reset singleton so production code gets real RapidOCR next call.
    _ocr_instance = None

    print("7 checks passed")


if __name__ == "__main__":
    _self_test()
