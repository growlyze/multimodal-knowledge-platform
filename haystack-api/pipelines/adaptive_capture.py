"""Adaptive periodic safety-frame + burst capture (client spec item).

Research basis (2026-10-08 pass, 17-backlog.md C17): MGSampler (ICCV 2021)
and an adaptive-keyframe-sampling write-up for screen recordings specifically
— both score change, then allocate more captures to high-change stretches
and fewer to static ones. This is the budget-allocation layer on top of the
SSIM signal AnnotationChangeDetector already computes, not a new model.

The gap: many lecture videos are long static screen
recordings with only narration, where the existing SSIM-only detector
captures nothing for minutes at a time — but a specific busy stretch (active
chart movement) may need a capture every 1-4 seconds. AdaptiveCaptureController
covers three cases SSIM-only misses:

1. Periodic safety frame — fires every `safety_interval_s` when nothing else
   has triggered a capture, so a long static narration segment isn't a total
   blank. Shrinks to `active_safety_interval_s` during a busy stretch.
2. Burst capture — a handful of extra consecutive sampled frames right after
   a detected change, to catch an annotation actually being drawn rather
   than just its first and last frame.
3. Audio-cue capture — a frame near a transcript timestamp where the
   presenter says something like "look here" (pipelines/audio_cue_capture.py
   finds these timestamps; this class just acts on them).
"""

import logging
import math
from typing import List, Optional

logger = logging.getLogger("haystack-api.adaptive_capture")


class AdaptiveCaptureController:
    """Call `decide()` once per sampled frame, in video order."""

    def __init__(
        self,
        safety_interval_s: float = 45.0,
        active_safety_interval_s: float = 2.0,
        active_window_s: float = 10.0,
        active_change_threshold: int = 3,
        burst_frame_count: int = 2,
        cue_timestamps_s: Optional[List[float]] = None,
        cue_tolerance_s: float = 1.0,
    ):
        self._safety_interval_ms = safety_interval_s * 1000
        self._active_safety_interval_ms = active_safety_interval_s * 1000
        self._active_window_ms = active_window_s * 1000
        self._active_change_threshold = active_change_threshold
        self._burst_frame_count = burst_frame_count
        self._cue_timestamps_ms = sorted(t * 1000 for t in (cue_timestamps_s or []))
        self._cue_tolerance_ms = cue_tolerance_s * 1000

        self._last_keyframe_ms = -math.inf
        self._recent_change_times_ms: List[float] = []
        self._pending_burst = 0
        self._consumed_cues_ms = set()

    def decide(self, frame_ms: float, changed: bool) -> Optional[str]:
        """Returns a capture reason ("change"/"burst"/"cue"/"safety") if this
        frame should become a keyframe, else None. `changed` is whatever
        AnnotationChangeDetector already decided for this frame — this class
        adds captures on top, it never removes a real detected change.
        """
        reason = None

        if changed:
            reason = "change"
            self._recent_change_times_ms.append(frame_ms)
            self._recent_change_times_ms = [
                t for t in self._recent_change_times_ms
                if frame_ms - t <= self._active_window_ms
            ]
            self._pending_burst = self._burst_frame_count
        elif self._pending_burst > 0:
            reason = "burst"
            self._pending_burst -= 1
        else:
            cue_ms = self._match_cue(frame_ms)
            if cue_ms is not None:
                reason = "cue"
            else:
                is_active = len(self._recent_change_times_ms) >= self._active_change_threshold
                interval_ms = self._active_safety_interval_ms if is_active else self._safety_interval_ms
                if frame_ms - self._last_keyframe_ms >= interval_ms:
                    reason = "safety"

        if reason is not None:
            self._last_keyframe_ms = frame_ms
        return reason

    def _match_cue(self, frame_ms: float) -> Optional[float]:
        # Nearest in-tolerance cue, not the first scanned — two cues can both
        # fall within tolerance of one frame, and timestamp order is not
        # distance order. Found in code review 2026-10-09.
        candidates = [
            cue_ms for cue_ms in self._cue_timestamps_ms
            if cue_ms not in self._consumed_cues_ms
            and abs(frame_ms - cue_ms) <= self._cue_tolerance_ms
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda cue_ms: abs(frame_ms - cue_ms))
        self._consumed_cues_ms.add(nearest)
        return nearest


def _self_test():
    # 1. A detected change is always captured, reason "change"
    c = AdaptiveCaptureController()
    assert c.decide(1000, changed=True) == "change"
    print("ok 1 — a detected change is captured with reason 'change'")

    # 2. Burst: the next burst_frame_count sampled frames after a change are
    # also captured, even with no further detected change.
    c2 = AdaptiveCaptureController(burst_frame_count=2, safety_interval_s=9999)
    assert c2.decide(0, changed=True) == "change"
    assert c2.decide(1000, changed=False) == "burst"
    assert c2.decide(2000, changed=False) == "burst"
    assert c2.decide(3000, changed=False) is None  # burst exhausted
    print("ok 2 — burst captures the frames right after a change, then stops")

    # 3. With no prior keyframe at all, the very first sampled frame fires a
    # safety capture immediately — there's nothing to wait "since", so a
    # video that starts with total silence still gets a baseline frame.
    c3 = AdaptiveCaptureController(safety_interval_s=5.0)
    assert c3.decide(0, changed=False) == "safety"
    print("ok 3 — first-ever frame fires a baseline safety capture")

    # 4. After that baseline, the next safety frame waits for the full
    # interval — not before, and fires right after.
    c4 = AdaptiveCaptureController(safety_interval_s=5.0)
    assert c4.decide(0, changed=False) == "safety"  # baseline (check 3's case)
    assert c4.decide(4000, changed=False) is None  # 4s since last keyframe < 5s interval
    assert c4.decide(5500, changed=False) == "safety"  # 5.5s >= 5s interval
    print("ok 4 — periodic safety frame fires once the interval elapses")

    # 5. During an "active" stretch (several changes within active_window_s),
    # the safety-frame interval shrinks to active_safety_interval_s.
    c5 = AdaptiveCaptureController(
        safety_interval_s=45.0, active_safety_interval_s=2.0,
        active_window_s=10.0, active_change_threshold=3)
    c5.decide(0, changed=True)
    c5.decide(1000, changed=True)
    c5.decide(2000, changed=True)  # 3 changes within 10s -> now "active"
    # burst window is still pending right after the 3rd change; drain it
    # (burst_frame_count default 2) before testing the safety cadence
    c5.decide(3000, changed=False)  # burst
    c5.decide(4000, changed=False)  # burst
    assert c5.decide(5000, changed=False) is None  # 1s since last keyframe (4000), < 2s active interval
    assert c5.decide(6500, changed=False) == "safety"  # 2.5s since 4000 >= 2s active interval
    print("ok 5 — active stretch shrinks the safety-frame interval")

    # 6. Audio-cue capture fires once near its timestamp, within tolerance,
    # and does not fire again for the same cue. Primed with a real change at
    # t=0 first so the "no prior keyframe" baseline-safety case (check 3)
    # doesn't interfere with what this check is actually testing.
    c6 = AdaptiveCaptureController(cue_timestamps_s=[10.0], cue_tolerance_s=1.0,
                                   safety_interval_s=9999)
    assert c6.decide(0, changed=True) == "change"
    assert c6.decide(1000, changed=False) == "burst"    # default burst_frame_count=2
    assert c6.decide(2000, changed=False) == "burst"
    assert c6.decide(8000, changed=False) is None       # 2s from cue, outside 1s tolerance
    assert c6.decide(10500, changed=False) == "cue"     # 0.5s away, within tolerance
    assert c6.decide(10600, changed=False) is None      # same cue already consumed
    print("ok 6 — audio-cue capture fires once per cue, within tolerance")

    # 7. A real detected change always wins over a coincidentally-nearby cue
    # or safety trigger — never silently dropped.
    c7 = AdaptiveCaptureController(cue_timestamps_s=[10.0], safety_interval_s=1.0)
    assert c7.decide(10000, changed=True) == "change"
    print("ok 7 — a detected change always takes priority")

    # 8. When two cues both fall within tolerance of the same frame,
    # _match_cue must pick the NEAREST one, not whichever comes first in
    # timestamp order. Cues at 10.0s and 10.9s, tolerance 1.0s, frame at
    # 10.6s: 10.0s is 0.6s away, 10.9s is 0.3s away — 10.9s is nearer.
    # Found in code review: the old loop returned the first in-tolerance
    # cue it scanned, which is always the earliest one, not the closest.
    c8 = AdaptiveCaptureController(cue_timestamps_s=[10.0, 10.9], cue_tolerance_s=1.0,
                                    safety_interval_s=9999)
    assert c8.decide(10600, changed=False) == "cue"
    assert c8._consumed_cues_ms == {10900.0}, c8._consumed_cues_ms
    print("ok 8 — _match_cue picks the nearest in-tolerance cue, not the first")

    print("\n8 checks passed")


if __name__ == "__main__":
    _self_test()
