"""
Keyframe detector for screen recordings (slides, charts, dashboards, code editors).

Accumulates structural dissimilarity from the *last emitted keyframe* (not from
the previous frame). Each raw frame contributes a delta; when the running total
exceeds ssim_threshold the frame is promoted to a keyframe, the reference is
reset, and the accumulator restarts from zero.

This is different from CaptnSeraph's approach (frame N vs N−1, threshold 0.25)
which emits on any single-frame spike — too sensitive for screen recordings where
annotations appear gradually and on-screen motion causes continuous low-level
change. The accumulated model captures the total structural shift since the last
interesting moment.

Gaussian blur is applied before SSIM to suppress encoder noise on the dark
chart background, which would otherwise accumulate noise deltas continuously and
over-trigger at low thresholds.

Tuning guide
------------
ssim_threshold : 0.15  — ~15% total structural change since last keyframe.
                         Raise to 0.20 to skip minor annotation tweaks; lower to
                         0.10 if small annotations (boxes, highlights) are being missed.
blur_ksize     : 5     — must be odd. Larger blurs suppress more noise but smear
                         annotation edges; 5 is the widest useful value.
blur_sigma     : 2.0   — Gaussian sigma. Correlated with ksize; don't raise both.

data_range=255 in ssim() is MANDATORY for uint8 frames. skimage computes an
incorrect score if this is omitted because it defaults to 1.0 (float range).
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class AnnotationChangeDetector:
    """Emit a keyframe whenever accumulated SSIM change exceeds a threshold.

    Usage
    -----
    detector = AnnotationChangeDetector()
    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if detector.process_frame(frame, frame_ms):
            process_keyframe(frame, frame_ms)

    cap.release()
    timestamps = detector.get_keyframe_timestamps()
    """

    def __init__(
        self,
        ssim_threshold: float = 0.15,
        blur_ksize: int = 5,
        blur_sigma: float = 2.0,
    ) -> None:
        if blur_ksize % 2 == 0:
            raise ValueError(f"blur_ksize must be odd; got {blur_ksize}")
        self.ssim_threshold = ssim_threshold
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self._reference_frame: Optional[np.ndarray] = None
        self._accum: float = 0.0
        self._keyframe_timestamps: List[int] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray, frame_ms: int) -> bool:
        """Process one video frame.

        Parameters
        ----------
        frame_bgr : numpy array, shape (H, W, 3), dtype uint8 — BGR from cv2.
        frame_ms  : timestamp in milliseconds (cv2.CAP_PROP_POS_MSEC as int).

        Returns True if this frame is a keyframe (first frame always is).
        """
        processed = self._preprocess(frame_bgr)

        if self._reference_frame is None:
            self._reference_frame = processed
            self._accum = 0.0
            self._keyframe_timestamps.append(frame_ms)
            return True

        score = ssim(processed, self._reference_frame, data_range=255)
        self._accum += 1.0 - score

        if self._accum >= self.ssim_threshold:
            self._reference_frame = processed
            self._accum = 0.0
            self._keyframe_timestamps.append(frame_ms)
            return True

        return False

    def get_keyframe_timestamps(self) -> List[int]:
        """Return a copy of all emitted keyframe timestamps (milliseconds)."""
        return list(self._keyframe_timestamps)

    def reset(self) -> None:
        """Reset internal state; call before processing a new video."""
        self._reference_frame = None
        self._accum = 0.0
        self._keyframe_timestamps = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Canonical resize for resolution-invariant SSIM
        resized = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(
            resized, (self.blur_ksize, self.blur_ksize), self.blur_sigma
        )
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


class NearDuplicateFilter:
    """Cost lever (VIDEO_KEYFRAME_DEDUPE_SSIM, video_ingestion._phase_visual):
    is a candidate keyframe visually the same as the last keyframe KEPT?

    AdaptiveCaptureController adds burst and safety frames on top of real
    changes. On a static chart those are near-identical to the frame before,
    yet each one bought a Gemini description and an image embedding (971
    bursts + 420 safety frames of 2,031 keyframes on z6AxPV2NQ9U). Uses the
    detector's own preprocess (640x360, blur, grey) so the comparison is
    resolution-invariant and ignores encoder noise the same way.

    Compares against the last kept frame, never the last candidate: chained
    candidate-to-candidate comparisons would let a slow drawing drift through
    as a run of "duplicates" and never be captured.
    """

    def __init__(self, threshold: float, blur_ksize: int = 5, blur_sigma: float = 2.0) -> None:
        # SSIM similarity in (0, 1]: 1.0 drops only pixel-identical frames;
        # anything <= 0 would call every frame a duplicate.
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
        self.threshold = threshold
        self._pre = AnnotationChangeDetector(blur_ksize=blur_ksize, blur_sigma=blur_sigma)
        self._kept: Optional[np.ndarray] = None

    def is_near_duplicate(self, frame_bgr: np.ndarray) -> bool:
        if self._kept is None:
            return False
        score = ssim(self._pre._preprocess(frame_bgr), self._kept, data_range=255)
        return score >= self.threshold

    def keep(self, frame_bgr: np.ndarray) -> None:
        self._kept = self._pre._preprocess(frame_bgr)


# ---------------------------------------------------------------------------
# Offline self-test (no network, no GPU, no test framework)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Verify core behaviour with synthetic frames. Prints ok/fail, no deps."""
    import sys

    detector = AnnotationChangeDetector(ssim_threshold=0.15)

    # Identical frame sequence — should emit only the first keyframe.
    base_frame = np.ones((64, 64, 3), dtype=np.uint8) * 30
    assert detector.process_frame(base_frame, 0), "first frame must be keyframe"
    for ms in range(33, 330, 33):
        result = detector.process_frame(base_frame, ms)
        assert not result, f"identical frame at {ms}ms should not be keyframe"
    assert detector.get_keyframe_timestamps() == [0], "only one keyframe expected"
    print("ok identical-frames-no-extra-keyframe")

    # Large structural change — should trigger a second keyframe.
    detector.reset()
    detector.process_frame(base_frame, 0)
    bright_frame = np.ones((64, 64, 3), dtype=np.uint8) * 200
    triggered = detector.process_frame(bright_frame, 1000)
    assert triggered, "bright frame should trigger keyframe"
    assert 1000 in detector.get_keyframe_timestamps()
    print("ok large-change-triggers-keyframe")

    # reset() clears state.
    detector.reset()
    assert detector.get_keyframe_timestamps() == []
    assert detector._reference_frame is None
    assert detector._accum == 0.0
    print("ok reset-clears-state")

    # Even-numbered ksize raises ValueError.
    try:
        AnnotationChangeDetector(blur_ksize=4)
        print("FAIL even-ksize-should-raise")
        sys.exit(1)
    except ValueError:
        pass
    print("ok even-ksize-raises")

    # Same content at different resolutions produces the same keyframe decision.
    # Both resolutions are resized to 640x360 internally, so SSIM should match.
    detector.reset()
    lo = np.ones((360, 640, 3), dtype=np.uint8) * 30
    hi = np.ones((720, 1280, 3), dtype=np.uint8) * 30
    detector.process_frame(lo, 0)
    # Feed the high-res version of the same solid colour — should NOT trigger
    # a second keyframe because, after resize, it is identical to the first.
    triggered_hi = detector.process_frame(hi, 1000)
    assert not triggered_hi, "same content at different resolution should not trigger keyframe"
    print("ok resolution-invariant-preprocessing")

    # NearDuplicateFilter (cost lever VIDEO_KEYFRAME_DEDUPE_SSIM): a candidate
    # keyframe that looks the same as the last KEPT keyframe is a near-
    # duplicate. On z6AxPV2NQ9U 971 of 2,031 keyframes were post-change
    # bursts and 420 were safety frames, most over an unchanged chart, and
    # every one cost a Gemini description plus an image embedding.
    dup = NearDuplicateFilter(threshold=0.98)
    chart = np.ones((360, 640, 3), dtype=np.uint8) * 30
    cv2.rectangle(chart, (100, 100), (300, 250), (200, 200, 200), -1)
    assert not dup.is_near_duplicate(chart), "nothing kept yet -> never a duplicate"
    dup.keep(chart)
    assert dup.is_near_duplicate(chart.copy()), "same frame must be a near-duplicate"
    # Same chart at another resolution: same 640x360 canonical preprocess.
    assert dup.is_near_duplicate(cv2.resize(chart, (1280, 720))), "resolution must not matter"
    redrawn = chart.copy()
    cv2.rectangle(redrawn, (350, 50), (600, 330), (0, 0, 255), -1)   # new zone drawn
    assert not dup.is_near_duplicate(redrawn), "a new annotation is not a duplicate"
    # Compares against the last KEPT frame, not the last candidate: a slow
    # drift of individually-tiny changes must not be dropped forever.
    dup.keep(redrawn)
    assert dup.is_near_duplicate(redrawn.copy())
    assert not dup.is_near_duplicate(chart)
    print("ok near-duplicate-filter")

    for bad in (0.0, -0.1, 1.5):
        try:
            NearDuplicateFilter(threshold=bad)
            print("FAIL near-duplicate-threshold-%s-should-raise" % bad)
            sys.exit(1)
        except ValueError:
            pass
    print("ok near-duplicate-threshold-validated")

    print("7 checks passed")


if __name__ == "__main__":
    _self_test()
