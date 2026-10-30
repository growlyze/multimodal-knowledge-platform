"""ASR track — Groq-hosted Whisper transcription with domain-term normalization.

D65: moved from local WhisperX to Groq's hosted whisper-large-v3 (full
model, not Turbo — client's explicit choice for maximum quality). Groq runs
the identical model weights, so this is a speed/infra change, not an
accuracy one (Groq reports WER 10.3%, matching other hosted providers).
Word-level timestamps are requested alongside segment-level ones so that
data isn't thrown away even though TranscriptSegment is still segment-level
for every existing consumer (checkpoints, quality checks, Qdrant payloads).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List

import config

logger = logging.getLogger("haystack-api.asr_track")


@dataclass
class TranscriptSegment:
    text: str
    start_time: float       # seconds
    end_time: float         # seconds
    confidence: float       # Groq's avg_logprob — a log probability, NOT a 0-1 score
    words: List[dict] = field(default_factory=list)  # [{"word","start","end"}, ...] within this segment


# Domain terminology priming — fed to Whisper's initial_prompt so the decoder
# biases toward these spellings over phonetic alternatives. Empty by default:
# each deployment lists its own terms in DOMAIN_JARGON (comma-separated).
DOMAIN_JARGON = [t.strip() for t in os.getenv("DOMAIN_JARGON", "").split(",")
                 if t.strip()]

INITIAL_PROMPT = ", ".join(DOMAIN_JARGON)

# Post-transcription regex corrections for common ASR mishearings of the
# configured domain terms. Each tuple: (pattern, replacement). Applied with
# re.IGNORECASE. The list is derived mechanically from DOMAIN_JARGON (see
# _derive_normalizations) plus MANUAL_FIXES — hand-written pairs for phonetic
# mishearings a deployment actually encounters, e.g. (r"\bess ell aye\b", "SLA").
# Empty by default: with no DOMAIN_JARGON there is nothing to normalize.
MANUAL_FIXES: list = []


def _derive_normalizations(terms: list) -> list:
    """(pattern, replacement) pairs that undo mechanical manglings of `terms`.

    An acronym may come back letter-by-letter ("s l a") or lowercased; a
    multi-word term may come back with its spaces dropped ("batchprocessing").
    """
    fixes = []
    for term in terms:
        canonical = term.strip()
        if not canonical:
            continue
        words = canonical.split()
        if len(words) == 1:
            spaced = " ?".join(re.escape(c) for c in canonical)
            fixes.append((r"\b%s\b" % spaced, canonical))
        else:
            spaced = " +".join(re.escape(w) for w in words)
            collapsed = " ?".join(re.escape(w) for w in words)
            fixes.append((r"\b%s\b" % spaced, canonical))
            fixes.append((r"\b%s\b" % collapsed, canonical))
    return fixes


NORMALIZATIONS = _derive_normalizations(DOMAIN_JARGON) + MANUAL_FIXES

_NORM_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in NORMALIZATIONS]


def configure_domain_jargon(terms: list, manual_fixes: list = None) -> None:
    """Set the deployment's domain terms; rebuilds the prompt and the fixes.

    The code-level counterpart of the DOMAIN_JARGON env var, for tests and for
    deployments that keep their terminology in code.
    """
    global DOMAIN_JARGON, INITIAL_PROMPT, NORMALIZATIONS, _NORM_COMPILED
    DOMAIN_JARGON = [t.strip() for t in terms if t.strip()]
    INITIAL_PROMPT = ", ".join(DOMAIN_JARGON)
    NORMALIZATIONS = _derive_normalizations(DOMAIN_JARGON) \
        + list(MANUAL_FIXES if manual_fixes is None else manual_fixes)
    _NORM_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in NORMALIZATIONS]


def normalize_jargon(text: str) -> str:
    """Apply regex normalizations to fix common ASR misheard terms."""
    for pattern, replacement in _NORM_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def _field(obj, key, default=None):
    """Groq's verbose_json extras may come back as dicts or attribute-style
    objects depending on SDK version — support both rather than guessing."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# P136: Groq rejects any upload over 25 MB ("Max Attachment File Size: 25
# MB" -- dev tier's 100 MB applies only to the url= parameter). extract_audio
# writes uncompressed 16 kHz mono WAV, ~1.9 MB/minute (~115 MB/hour), so any
# video past ~13 minutes failed transcription outright -- invisible until now
# because the only ingested video was a 70-second reel. Over the cap, the
# audio is cut into overlapping FLAC chunks (lossless, ~5-10 MB per 10 min),
# each transcribed separately and re-timed onto the full timeline -- the
# chunking Groq's own docs recommend for long audio.
GROQ_MAX_UPLOAD_BYTES = 24 * 1024 * 1024   # 1 MB under the cap, for headroom
ASR_CHUNK_SECONDS = 600
# Each cut extends this far past its window on both sides, so a sentence
# spoken across a cut is heard whole by at least one chunk; _keep_window then
# keeps it only from the chunk whose window holds its midpoint.
ASR_CHUNK_OVERLAP_SECONDS = 5


def transcribe_video(audio_path: str, model: str = None,
                     on_chunk=None) -> List[TranscriptSegment]:
    """Transcribe audio via Groq's hosted Whisper API (D65), chunking it
    first if it's over Groq's upload cap (P136).

    `on_chunk(done, total)`, when given, runs before each chunk's Groq request
    (done = chunks finished so far). The ingestion orchestrator uses it for the
    ASR progress bar and as a cancel checkpoint: whatever it raises stops the
    loop before the next paid request. A file under the cap is one request,
    so it is never called there -- the orchestrator checks before the phase.
    """
    if os.path.getsize(audio_path) <= GROQ_MAX_UPLOAD_BYTES:
        return _transcribe_file(audio_path, model)

    import tempfile

    total_s = _audio_duration(audio_path)
    if total_s <= 0:
        raise ValueError("audio is over Groq's upload cap and its duration couldn't "
                         "be read, so it can't be chunked: %s" % audio_path)
    plan = _chunk_plan(total_s)
    logger.info("audio is %.1f MB (over Groq's %d MB cap): transcribing %.0fs in %d chunks",
                os.path.getsize(audio_path) / 1e6, GROQ_MAX_UPLOAD_BYTES // (1024 * 1024),
                total_s, len(plan))
    segments: List[TranscriptSegment] = []
    with tempfile.TemporaryDirectory() as out_dir:
        for index, (start, dur, keep_lo, keep_hi) in enumerate(plan):
            if on_chunk is not None:
                on_chunk(index, len(plan))
            chunk = _cut_chunk(audio_path, start, dur, out_dir, index)
            part = _transcribe_file(chunk, model, offset=start)
            segments.extend(_keep_window(part, keep_lo, keep_hi))
    return segments


def _chunk_plan(total_s: float, chunk_s: float = None, overlap_s: float = None) -> list:
    """[(cut_start, cut_duration, keep_from, keep_until), ...] covering total_s.

    Windows [keep_from, keep_until) tile the timeline with no overlap; each
    cut extends overlap_s past its window on both sides. The last window is
    open-ended so nothing at the very end can be dropped by rounding.
    """
    chunk_s = ASR_CHUNK_SECONDS if chunk_s is None else chunk_s
    overlap_s = ASR_CHUNK_OVERLAP_SECONDS if overlap_s is None else overlap_s
    plan = []
    index = 0
    while index * chunk_s < total_s:
        lo = float(index * chunk_s)
        hi = float(min((index + 1) * chunk_s, total_s))
        start = max(0.0, lo - overlap_s)
        end = min(float(total_s), hi + overlap_s)
        plan.append((start, end - start, lo, hi if hi < total_s else float("inf")))
        index += 1
    return plan


def _keep_window(segments: List[TranscriptSegment], lo: float, hi: float) -> list:
    """Segments whose midpoint falls in [lo, hi) -- the dedupe rule for overlaps."""
    return [s for s in segments if lo <= (s.start_time + s.end_time) / 2 < hi]


def _audio_duration(audio_path: str) -> float:
    from pipelines.video_acquisition import probe_duration_seconds
    return probe_duration_seconds(audio_path)


def _cut_chunk(audio_path: str, start: float, duration: float, out_dir: str, index: int) -> str:
    """One FLAC chunk of [start, start+duration). FLAC because it's lossless
    and Groq recommends it for shrinking audio without hurting accuracy."""
    import subprocess
    out = os.path.join(out_dir, "chunk%03d.flac" % index)
    subprocess.run(
        ["ffmpeg", "-ss", "%.3f" % start, "-t", "%.3f" % duration, "-i", audio_path,
         "-ac", "1", "-ar", "16000", "-c:a", "flac", out, "-y"],
        capture_output=True, check=True,
    )
    return out


def _transcribe_file(audio_path: str, model: str = None,
                     offset: float = 0.0) -> List[TranscriptSegment]:
    """One Groq call on one file (must be under the upload cap).

    Full large-v3 (not Turbo) — identical model weights to local WhisperX,
    so this is a speed/infra swap, not an accuracy one. Requests both word-
    and segment-level timestamps: segments populate TranscriptSegment for
    every existing consumer, words are kept on each segment's `.words` so
    finer-grained timestamp data isn't discarded. `offset` shifts every
    timestamp, so a chunk's results land at its place in the full video.
    """
    from groq import Groq

    if model is None:
        model = getattr(config, "GROQ_WHISPER_MODEL", "whisper-large-v3")

    client = Groq()
    logger.info("transcribing %s via Groq (%s)", audio_path, model)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            language="en",
            prompt=INITIAL_PROMPT,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    raw_segments = _field(result, "segments", []) or []
    raw_words = _field(result, "words", []) or []

    # Groq bills each request by its audio duration (10 s minimum), which
    # verbose_json reports. Metered per request so the chunk overlap is
    # counted -- the cost reported to the client is measured, not duration x rate.
    billed = _field(result, "duration", None)
    if billed is None:
        billed = max((_field(s, "end", 0.0) for s in raw_segments), default=0.0)
        logger.warning("Groq response had no duration; metering %.1fs from the last segment end", billed)
    from pipelines import usage_meter
    usage_meter.record_groq(float(billed))

    # Bucket words by which segment's window they fall in — Groq returns
    # them as two flat lists, not pre-grouped. Adjacent segments can leave a
    # small silence gap between one's end and the next's start; a word
    # landing in that gap belongs to neither segment's own [start, end), so
    # the window is widened to the midpoint of the gap instead, assigning
    # the word to whichever segment it's nearer to rather than dropping it.
    # Found in code review 2026-10-09.
    def _words_in(seg_start: float, seg_end: float) -> list:
        return [
            {"word": _field(w, "word", ""), "start": _field(w, "start", 0.0) + offset,
             "end": _field(w, "end", 0.0) + offset}
            for w in raw_words
            if seg_start <= _field(w, "start", 0.0) < seg_end
        ]

    segments = []
    norm_count = 0
    for idx, seg in enumerate(raw_segments):
        original = _field(seg, "text", "")
        normalized = normalize_jargon(original)
        if normalized != original:
            norm_count += 1
        start = _field(seg, "start", 0.0)
        end = _field(seg, "end", 0.0)
        window_start = start if idx == 0 else (
            _field(raw_segments[idx - 1], "end", 0.0) + start) / 2
        window_end = end if idx == len(raw_segments) - 1 else (
            end + _field(raw_segments[idx + 1], "start", 0.0)) / 2
        segments.append(TranscriptSegment(
            text=normalized,
            start_time=start + offset,
            end_time=end + offset,
            confidence=_field(seg, "avg_logprob", 0.0),
            words=_words_in(window_start, window_end),
        ))

    logger.info("transcribed %d segments, %d normalizations applied",
                len(segments), norm_count)
    return segments


def _self_test():
    """Offline tests — no Whisper or audio needed."""

    configure_domain_jargon(["SLA", "KPI", "batch processing"],
                            manual_fixes=[(r"\bess ell aye\b", "SLA")])

    # 1. normalize_jargon: spaced-out acronym -> canonical form
    assert normalize_jargon("the s l a was renewed") == "the SLA was renewed"
    print("ok 1 — 's l a' -> 'SLA'")

    # 2. normalize_jargon: lowercase acronym -> canonical form
    assert normalize_jargon("our kpi moved again") == "our KPI moved again"
    print("ok 2 — 'kpi' -> 'KPI'")

    # 3. normalize_jargon: phonetic mishearing fixed by MANUAL_FIXES
    result = normalize_jargon("the ess ell aye was renewed")
    assert "SLA" in result, f"got: {result}"
    print("ok 3 — 'ess ell aye' -> 'SLA'")

    # 4. normalize_jargon: case insensitive
    assert "SLA" in normalize_jargon("the S L A renewal date")
    print("ok 4 — acronym case insensitive")

    # 5. normalize_jargon: multi-word term with its spaces dropped is repaired
    assert normalize_jargon("a batchprocessing run") == "a batch processing run"
    print("ok 5 — 'batchprocessing' -> 'batch processing'")

    # 6. no false positives on normal text outside the configured terms
    normal = "the manager said hello to the team"
    result = normalize_jargon(normal)
    assert result == normal
    print("ok 6 — normal text untouched")

    # 7. All NORMALIZATIONS compile as valid regex
    for i, (pattern, _) in enumerate(NORMALIZATIONS):
        re.compile(pattern, re.IGNORECASE)
    print(f"ok 7 — all {len(NORMALIZATIONS)} normalizations are valid regex")

    # 8. TranscriptSegment dataclass, including the words field default
    seg = TranscriptSegment(text="test", start_time=1.0, end_time=2.0, confidence=0.9)
    assert seg.text == "test"
    assert seg.start_time == 1.0
    assert seg.words == [], "words should default to an empty list, not shared across instances"
    print("ok 8 — TranscriptSegment dataclass")

    # 9. INITIAL_PROMPT carries the configured terms
    for term in ["SLA", "KPI", "batch processing"]:
        assert term in INITIAL_PROMPT, f"{term} missing from INITIAL_PROMPT"
    print("ok 9 — INITIAL_PROMPT contains all configured terms")

    # 10. normalize_jargon: multiple substitutions in one string
    result = normalize_jargon("the s l a and the kpi renew together")
    assert "SLA" in result and "KPI" in result
    print("ok 10 — multiple substitutions in one pass")

    # 11. transcribe_video parses Groq's verbose_json shape (segments + words)
    # into TranscriptSegment, bucketing each word into its owning segment —
    # using a fake Groq client so no network call or API key is needed.
    import sys
    import types

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        def __init__(self, text, start, end, avg_logprob):
            self.text, self.start, self.end = text, start, end
            self.avg_logprob = avg_logprob

    class FakeTranscription:
        def __init__(self):
            self.text = "the s l a was renewed then the batch ran"
            self.segments = [
                FakeSegment("the s l a was renewed", 0.0, 2.0, -0.25),
                FakeSegment("then the batch ran", 2.0, 4.0, -0.30),
            ]
            self.words = [
                FakeWord("the", 0.0, 0.3), FakeWord("s", 0.3, 0.5),
                FakeWord("l", 0.5, 0.7), FakeWord("a", 0.7, 1.0),
                FakeWord("was", 1.0, 1.3), FakeWord("renewed", 1.3, 2.0),
                FakeWord("then", 2.0, 2.3), FakeWord("the", 2.3, 2.6),
                FakeWord("batch", 2.6, 2.9), FakeWord("ran", 2.9, 3.2),
            ]

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured_create_kwargs.append(kwargs)
            return FakeTranscription()

    class FakeAudio:
        def __init__(self):
            self.transcriptions = FakeTranscriptions()

    class FakeGroqClient:
        def __init__(self, **kwargs):
            self.audio = FakeAudio()

    captured_create_kwargs = []
    fake_groq = types.ModuleType("groq")
    fake_groq.Groq = FakeGroqClient
    sys.modules["groq"] = fake_groq

    import tempfile as _tempfile
    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
        f.write(b"fake audio bytes")
    try:
        segments = transcribe_video(audio_path)
    finally:
        del sys.modules["groq"]
        os.remove(audio_path)

    assert len(segments) == 2, segments
    assert segments[0].text == "the SLA was renewed", segments[0].text  # jargon-normalized
    assert segments[0].start_time == 0.0 and segments[0].end_time == 2.0
    assert segments[0].confidence == -0.25, segments[0].confidence
    assert len(segments[0].words) == 6, segments[0].words  # "the s l a was renewed"
    assert len(segments[1].words) == 4, segments[1].words  # "then the batch ran"
    assert captured_create_kwargs[0]["response_format"] == "verbose_json"
    assert set(captured_create_kwargs[0]["timestamp_granularities"]) == {"word", "segment"}
    print("ok 11 — transcribe_video parses Groq's verbose_json into TranscriptSegment + words")

    # 12. A word landing in the small gap between two segments (neither
    # segment's own [start, end) window covers it) must still be assigned —
    # to whichever segment it's nearer to — not silently dropped. Segments
    # end at 2.0 / start at 2.2 (a 0.2s gap); "um" at 2.05 is nearer the
    # first segment, "then" at 2.15 is nearer the second.
    # Found in code review 2026-10-09.
    class FakeTranscriptionGap(FakeTranscription):
        def __init__(self):
            self.text = "ready um then go"
            self.segments = [
                FakeSegment("ready", 0.0, 2.0, -0.2),
                FakeSegment("then go", 2.2, 4.0, -0.2),
            ]
            self.words = [
                FakeWord("ready", 0.0, 2.0),
                FakeWord("um", 2.05, 2.1),
                FakeWord("then", 2.15, 2.19),
                FakeWord("go", 2.2, 2.5),
            ]

    class FakeTranscriptionsGap:
        def create(self, **kwargs):
            return FakeTranscriptionGap()

    class FakeAudioGap:
        def __init__(self):
            self.transcriptions = FakeTranscriptionsGap()

    class FakeGroqClientGap:
        def __init__(self, **kwargs):
            self.audio = FakeAudioGap()

    fake_groq_gap = types.ModuleType("groq")
    fake_groq_gap.Groq = FakeGroqClientGap
    sys.modules["groq"] = fake_groq_gap

    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
        f.write(b"fake audio bytes")
    try:
        gap_segments = transcribe_video(audio_path)
    finally:
        del sys.modules["groq"]
        os.remove(audio_path)

    gap_words_0 = [w["word"] for w in gap_segments[0].words]
    gap_words_1 = [w["word"] for w in gap_segments[1].words]
    assert "um" in gap_words_0, gap_segments[0].words
    assert "then" in gap_words_1, gap_segments[1].words
    print("ok 12 — a word in the gap between segments is assigned to the nearer one, not dropped")

    # 13. P136: the chunk plan covers the whole file with overlapping cuts,
    # and each chunk "owns" a non-overlapping window -- a segment straddling
    # a boundary is kept exactly once (by whichever window holds its midpoint).
    plan = _chunk_plan(1250.0, chunk_s=600, overlap_s=5)
    assert plan == [
        (0.0, 605.0, 0.0, 600.0),
        (595.0, 610.0, 600.0, 1200.0),
        (1195.0, 55.0, 1200.0, float("inf")),
    ], plan
    straddler = TranscriptSegment("across the cut", 598.0, 604.0, -0.2)  # midpoint 601
    assert _keep_window([straddler], 0.0, 600.0) == []
    assert _keep_window([straddler], 600.0, 1200.0) == [straddler]
    print("ok 13 — chunk plan overlaps cuts but gives each segment to exactly one chunk")

    # 14. A file over Groq's 25 MB upload cap is split into FLAC chunks, each
    # transcribed separately, timestamps shifted back onto the full timeline,
    # and the boundary-straddling segment appears once -- not twice, not lost.
    # Found 2026-10-19: one uncompressed 16 kHz WAV is ~115 MB per hour, so
    # anything past ~13 minutes would have failed transcription outright.
    chunk_calls = []

    def _chunk_response(call_index):
        # chunk-relative times; chunk 0 starts at 0s, chunk 1 at 595s
        if call_index == 0:
            return [FakeSegment("opening words", 1.0, 3.0, -0.2),
                    FakeSegment("across the cut", 598.0, 604.0, -0.2)]   # abs 598-604
        if call_index == 1:
            return [FakeSegment("across the cut", 3.0, 9.0, -0.2),        # abs 598-604 again
                    FakeSegment("middle words", 100.0, 102.0, -0.2)]      # abs 695-697
        return [FakeSegment("closing words", 10.0, 12.0, -0.2)]           # abs 1205-1207

    class FakeChunkTranscription:
        def __init__(self, segs, duration):
            self.duration = duration   # verbose_json carries the audio duration
            self.text = " ".join(s.text for s in segs)
            self.segments = segs
            self.words = [FakeWord(s.text.split()[0], s.start, s.start + 0.5) for s in segs]

    class FakeChunkTranscriptions:
        def create(self, **kwargs):
            chunk_calls.append(kwargs["file"].name)
            i = len(chunk_calls) - 1
            return FakeChunkTranscription(_chunk_response(i), [605.0, 610.0, 55.0][i])

    class FakeChunkAudio:
        def __init__(self):
            self.transcriptions = FakeChunkTranscriptions()

    class FakeChunkGroq:
        def __init__(self, **kwargs):
            self.audio = FakeChunkAudio()

    fake_groq_chunk = types.ModuleType("groq")
    fake_groq_chunk.Groq = FakeChunkGroq
    sys.modules["groq"] = fake_groq_chunk

    global GROQ_MAX_UPLOAD_BYTES, _audio_duration, _cut_chunk
    saved = (GROQ_MAX_UPLOAD_BYTES, _audio_duration, _cut_chunk)
    cut_calls = []

    def _fake_cut(src, start, dur, out_dir, index):
        cut_calls.append((start, dur))
        path = os.path.join(out_dir, "chunk%03d.flac" % index)
        with open(path, "wb") as fh:
            fh.write(b"flac")
        return path

    GROQ_MAX_UPLOAD_BYTES = 4          # force the chunked path
    _audio_duration = lambda path: 1250.0
    _cut_chunk = _fake_cut
    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
        f.write(b"bigger than four bytes")
    from pipelines import usage_meter
    meter = usage_meter.start("asr-test")
    try:
        chunked = transcribe_video(audio_path)
    finally:
        usage_meter.stop()
        del sys.modules["groq"]
        GROQ_MAX_UPLOAD_BYTES, _audio_duration, _cut_chunk = saved
        os.remove(audio_path)

    assert cut_calls == [(0.0, 605.0), (595.0, 610.0), (1195.0, 55.0)], cut_calls
    assert all(p.endswith(".flac") for p in chunk_calls), chunk_calls
    texts = [s.text for s in chunked]
    assert texts == ["opening words", "across the cut", "middle words", "closing words"], texts
    starts = [s.start_time for s in chunked]
    assert starts == [1.0, 598.0, 695.0, 1205.0], starts
    assert chunked[2].words[0]["start"] == 695.0, chunked[2].words  # words shifted too
    print("ok 14 — an over-limit file is chunked, re-timed onto the full timeline, boundary kept once")

    # 15. Each Groq request's reported duration is metered -- the cost the client is
    # given is measured audio seconds, overlap included, not duration x rate.
    g = meter.summary()["groq"]
    assert g["requests"] == 3 and g["audio_seconds"] == 1270.0, g
    print("ok 15 — every Groq request's audio duration is metered (overlap included)")

    # 16. on_chunk(done, total) runs before every chunk's paid Groq request, so
    # the caller can report "chunks done / total" and stop a cancelled job
    # between chunks: an exception raised by it stops the loop before the
    # next request is sent (a 2-hour video is 12 requests; a cancel must not
    # wait for all of them).
    class _Stop(Exception):
        pass

    seen = []

    def _on_chunk(done, total):
        seen.append((done, total))
        if done == 1:
            raise _Stop()

    chunk_calls.clear()
    sys.modules["groq"] = fake_groq_chunk
    GROQ_MAX_UPLOAD_BYTES = 4
    _audio_duration = lambda path: 1250.0
    _cut_chunk = _fake_cut
    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
        f.write(b"bigger than four bytes")
    try:
        try:
            transcribe_video(audio_path, on_chunk=_on_chunk)
            raise AssertionError("on_chunk's exception must propagate")
        except _Stop:
            pass
    finally:
        del sys.modules["groq"]
        GROQ_MAX_UPLOAD_BYTES, _audio_duration, _cut_chunk = saved
        os.remove(audio_path)
    assert seen == [(0, 3), (1, 3)], seen
    assert len(chunk_calls) == 1, chunk_calls     # chunk 2 and 3 never sent
    print("ok 16 — on_chunk reports chunks done/total and can stop before the next paid request")

    print("\n15 checks passed")


if __name__ == "__main__":
    _self_test()
