"""Video acquisition via yt-dlp — metadata listing and 1080p download."""

import logging
import os
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config

logger = logging.getLogger("haystack-api.video_acquisition")


# P133: yt-dlp's `cookiefile` is read-AND-write -- YoutubeDL.close() always
# calls save_cookies(), dumping its in-memory jar back into that file (the
# docs: "file to read cookies from and dump cookie jar in"; there is no
# read-only mode). Pointed at the canonical uploaded cookies.txt, a handful
# of calls replaced a validated login session with YouTube's anonymous
# tracking cookies (GPS, SOCS, VISITOR_INFO1_LIVE, __Secure-YNID) and lost
# SID/SAPISID/LOGIN_INFO entirely -- observed live 2026-10-16. So yt-dlp
# only ever gets a disposable copy; the canonical file stays exactly what
# the admin uploaded and validated.
_COOKIE_LOCK = threading.Lock()
_COOKIE_COPY_DIR = os.path.join(tempfile.gettempdir(), "ytdlp-cookie-copies")
# Copies are swept after this long. Generous because yt-dlp writes its jar
# at close(), which for a long download can be hours after the copy was made.
_COOKIE_COPY_MAX_AGE_S = 6 * 3600


def _disposable_cookie_copy(cookies_file: str) -> str:
    """Copy cookies_file to a fresh private temp file and return its path.

    mkstemp creates it 0600 and copyfile keeps that mode -- these are live
    session credentials, not world-readable scratch. _base_ydl_opts can't
    delete its copy after use (the caller owns the YoutubeDL lifecycle), so
    stale copies are swept here on each call instead of accumulating.
    """
    os.makedirs(_COOKIE_COPY_DIR, mode=0o700, exist_ok=True)
    cutoff = time.time() - _COOKIE_COPY_MAX_AGE_S
    for name in os.listdir(_COOKIE_COPY_DIR):
        path = os.path.join(_COOKIE_COPY_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=_COOKIE_COPY_DIR)
    os.close(fd)
    shutil.copyfile(cookies_file, tmp_path)
    return tmp_path


def _base_ydl_opts(**overrides) -> dict:
    """Options every yt-dlp call uses — impersonation + cookies (P91 fix).

    `impersonate` disguises the TLS/HTTP fingerprint as a real browser
    (curl_cffi-backed) rather than changing the IP — this is the piece the
    client specifically asked for over a residential/mobile proxy (C21).
    `cookiefile`, if configured and under cookie_guard's budget, adds an
    authenticated session on top; 2026 community reports found the
    combination of both is what actually resolves YouTube's bot-detection
    challenge from a datacenter IP. Cookie usage is budgeted separately
    (daily cap, minimum delay between uses, a max file age forcing
    re-export) since a real account used this way risks being rate-limited
    or banned — see pipelines/cookie_guard.py.

    Both impersonation and cookies degrade to a no-op rather than raising:
    curl_cffi may not be installed yet, a cookies file may not exist, and a
    withheld-by-budget cookie file is not an error either — none of these
    may be the reason a download fails outright, only the reason it's more
    or less likely to succeed.
    """
    opts = _client_opts()
    cookies_file = getattr(config, "YTDLP_COOKIES_FILE", "")
    if cookies_file and os.path.isfile(cookies_file):
        from pipelines.cookie_guard import (check_cookie_budget, record_cookie_use,
                                            spacing_wait_seconds)
        # Held across the wait, the check and the record so two callers can't
        # both wait out the same gap and then both use the cookies.
        with _COOKIE_LOCK:
            # Unmet spacing alone is waited out, not treated as "withhold":
            # withholding sent the call anonymous, a guaranteed bot check on
            # YouTube content (fetch-url 21 s after a validate, 2026-10-19). Always
            # called from a worker thread (asyncio.to_thread), never the loop.
            wait = spacing_wait_seconds(cookies_file)
            if wait > 0:
                logger.info("waiting %.0fs for cookie spacing before this yt-dlp call", wait)
                time.sleep(wait)
            ok, reason = check_cookie_budget(cookies_file)
            if ok:
                opts["cookiefile"] = _disposable_cookie_copy(cookies_file)
                record_cookie_use(cookies_file)  # budget is tracked on the canonical file
            else:
                logger.warning("cookies withheld for this yt-dlp call: %s", reason)
    opts.update(overrides)
    return opts


def _client_opts() -> dict:
    """Impersonation + PO-token provider -- everything except cookies.

    Shared by _base_ydl_opts and probe_cookies so a cookie validation
    exercises the same client path as a real download; the probe used to
    build its own dict without the PO-token arg, testing a configuration
    nothing actually used.
    """
    opts = {"quiet": True, "no_warnings": True}
    target = getattr(config, "YTDLP_IMPERSONATE_TARGET", "")
    if target:
        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget
            opts["impersonate"] = ImpersonateTarget.from_str(target)
        except Exception as exc:
            logger.warning("impersonate target %r unavailable, continuing without it: %s",
                           target, exc)
    pot_url = getattr(config, "YTDLP_POT_PROVIDER_URL", "")
    if pot_url:
        # P135: bgutil-ytdlp-pot-provider 1.3.x reads this key and REJECTS
        # the old 'youtube': {'getpot_bgutil_baseurl': ...} outright ("extractor
        # arg is deprecated") -- so every PO-token request silently failed and
        # YouTube forced the logged-in web client onto SABR with no formats.
        opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [pot_url]}}
    return opts


@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration: int           # seconds
    upload_date: str        # YYYYMMDD from yt-dlp
    channel_id: str
    chapters: List[dict] = field(default_factory=list)
    thumbnail_url: str = ""


# Every tab that holds channel content. Only /videos was scanned until
# 2026-10-19, which missed a channel's live streams entirely (165 videos, 281.6 h):
# the library showed 370 h against the ~622 h the whole channel holds.
CHANNEL_TABS = ("videos", "streams", "shorts")


def list_channel_videos(channel_url: str) -> List[dict]:
    """Metadata-only listing of the whole channel. No download, minimal bandwidth.

    Returns plain dicts (not VideoMeta) so the video router can insert
    them into PostgreSQL and serialise them as JSON without conversion.
    Each CHANNEL_TABS tab is listed separately -- yt-dlp's extract_flat on a
    bare handle returns zero entries -- and de-duplicated by video_id.
    `channel_url` may be the handle or any tab URL of it.
    """
    import yt_dlp
    base = channel_url.rstrip("/")
    if base.rsplit("/", 1)[-1] in CHANNEL_TABS:
        base = base.rsplit("/", 1)[0]
    entries, seen = [], set()
    for i, tab in enumerate(CHANNEL_TABS):
        if i:
            # Wait out the cookie spacing: called sooner, _base_ydl_opts would
            # withhold the cookies and the listing would go out anonymous.
            time.sleep(config.YTDLP_COOKIES_MIN_DELAY_SECONDS)
        try:
            with yt_dlp.YoutubeDL(_base_ydl_opts(extract_flat=True)) as ydl:
                info = ydl.extract_info(base + "/" + tab, download=False)
        except Exception as exc:
            if i == 0:
                raise   # /videos is the core listing -- its failure is real
            logger.warning("channel tab %s skipped: %s", tab, exc)
            continue
        for entry in info.get("entries") or []:
            if entry and entry.get("id") not in seen:
                seen.add(entry.get("id"))
                entries.append(entry)
    channel_url = base
    results = []
    for entry in entries:
        meta = _to_meta(entry)
        results.append({
            "video_id": meta.video_id,
            "title": meta.title,
            "duration": meta.duration,
            "upload_date": meta.upload_date,
            "channel_id": meta.channel_id,
            "chapter_count": len(meta.chapters),
            "thumbnail_url": meta.thumbnail_url,
        })
    logger.info("listed %d videos from %s", len(results), channel_url)
    return results


def get_video_metadata(video_id: str) -> VideoMeta:
    """Full metadata for a single video (chapters, duration, etc.)."""
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = _base_ydl_opts()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return _to_meta(info)


def fetch_url_metadata(url: str) -> dict:
    """Fetch metadata for a single video pasted from any yt-dlp-supported
    platform (YouTube, Twitter/X, Instagram, TikTok, ...).

    Full (non-flat) extraction — a single paste needs title/thumbnail/duration
    immediately, unlike list_channel_videos() where flat extraction keeps a
    1800-video channel scan fast. Does not download anything.

    Returns the same dict shape as list_channel_videos() entries, plus
    "platform" (yt-dlp's extractor, lowercased) and "source_url" (the pasted
    URL) — source_url is what lets a later ingest download a non-YouTube
    video, since its video_id alone can't be turned back into a URL.
    """
    import yt_dlp
    ydl_opts = _base_ydl_opts()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    meta = _to_meta(info)
    platform = (info.get("extractor_key") or info.get("extractor") or "youtube").lower()
    return {
        "video_id": meta.video_id,
        "title": meta.title,
        "duration": meta.duration,
        "upload_date": meta.upload_date,
        "channel_id": meta.channel_id,
        "chapter_count": len(meta.chapters),
        "thumbnail_url": meta.thumbnail_url,
        "platform": platform,
        "source_url": url,
    }


def _resolve_download_url(video_id: str, platform: str = "youtube", source_url: str = "") -> str:
    """Pick the URL yt-dlp should download from.

    YouTube is always re-derived from video_id — it's stable and channel-
    discovered videos never had a source_url recorded. Every other platform
    has no synthesizable watch URL, so the source_url captured at fetch time
    is the only way back to the content; without it there is nothing safe to
    download, so this raises rather than guessing.
    """
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={video_id}"
    if not source_url:
        raise ValueError(
            f"no source_url recorded for non-YouTube video {video_id} "
            f"(platform={platform}) — cannot download"
        )
    return source_url


def download_video(
    video_id: str, output_dir: str,
    platform: str = "youtube", source_url: str = "",
) -> str:
    """Download video at up to 1080p. Returns path to downloaded file."""
    # Drive intake (V6): files the client's local fetcher uploaded to Google Drive.
    # A Drive view URL is not something yt-dlp can fetch (it needs the owner's
    # OAuth), so these go through rclone by file id instead. Checked before the
    # yt_dlp import so this path works even where yt-dlp is not installed.
    from pipelines import drive_intake
    if platform == drive_intake.PLATFORM:
        return drive_intake.download_for_ingest(video_id, output_dir)
    import yt_dlp
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = _base_ydl_opts(
        # Final unconditional "best" fallback: some platforms (e.g.
        # Instagram Reels) never offer anything <=720p -- a portrait Reel's
        # lowest video-only stream can already be 1280px tall, and its
        # combined-format options often report no height metadata at all,
        # so both height-restricted halves of this selector match nothing.
        # Found live 2026-10-12 (first real Instagram ingestion attempt).
        # P137: H.264 (avc1) first. yt-dlp ranks AV1 above H.264, YouTube
        # offers AV1 at 720p, and the image's OpenCV can't decode AV1 ("Failed
        # to get pixel format") -- _phase_visual then read zero frames and
        # "succeeded" with no keyframes, OCR or chart descriptions. The
        # unrestricted alternatives stay as fallbacks for sources with no
        # H.264 stream.
        # D-1080 (2026-10-27): cap raised 720 -> 1080. OCR reads the full frame and
        # small chart labels need the pixels: 30-frame test, 111 labels - 108 read /
        # 1 wrong at 1080p vs 102 / 9 wrong at 720p. Gemini scales every image to the
        # same 1,242-token budget, so description cost is unchanged.
        format=("bestvideo[height<=1080][vcodec^=avc1]+bestaudio"
                "/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"),
        outtmpl=f"{output_dir}/%(id)s.%(ext)s",
        merge_output_format="mp4",
        # Segmented sources (X serves ~355 HLS segments per clip) were fetched
        # one segment at a time: 75.6 s vs 12.3 s for the same 39 MB clip at 8
        # (benchmark 2026-10-19; 16 gave nothing more). The server's bandwidth
        # was never the limit -- per-segment round trips were.
        concurrent_fragment_downloads=8,
    )
    url = _resolve_download_url(video_id, platform=platform, source_url=source_url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    logger.info("downloaded %s to %s", video_id, path)
    return path


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """Extract audio track as WAV using ffmpeg. Returns audio path."""
    import subprocess
    if output_dir is None:
        output_dir = str(Path(video_path).parent)
    stem = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{stem}.wav")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, check=True,
    )
    logger.info("extracted audio to %s", audio_path)
    return audio_path


def probe_cookies(cookies_file: str, test_video_id: str = "dQw4w9WgXcQ") -> dict:
    """Confirm a cookies file actually authenticates against YouTube right
    now -- a metadata-only call (no download) against a stable, well-known
    public video, run with ONLY this file's cookies attached.

    Self-contained rather than routed through _base_ydl_opts(): validation
    must always present the cookies being tested, regardless of what
    cookie_guard's budget would currently decide for a normal call, and
    must record exactly one use either way -- going through the shared
    helper's own internal budget-check-then-maybe-record path risks either
    skipping the cookies entirely (if the budget says no), or double-
    counting the use (helper records once, caller records again).

    Returns {"valid": bool, "error": str | None}. Never raises -- a broken
    cookies file is a result to report, not a crash.
    """
    import yt_dlp
    from pipelines.cookie_guard import record_cookie_use

    # A throwaway copy, never the canonical file -- see P133 / _COOKIE_COPY_DIR.
    cookie_copy = _disposable_cookie_copy(cookies_file)
    opts = _client_opts()
    opts["cookiefile"] = cookie_copy

    record_cookie_use(cookies_file)
    url = "https://www.youtube.com/watch?v=%s" % test_video_id
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=False)
        return {"valid": True, "error": None}
    except Exception as exc:
        msg = str(exc)
        if "sign in" in msg.lower() or "not a bot" in msg.lower():
            return {"valid": False,
                    "error": "YouTube still challenges this session (bot check) -- "
                             "cookies may be stale, expired, or not enough on their own"}
        return {"valid": False, "error": msg[:300]}
    finally:
        try:
            os.remove(cookie_copy)
        except OSError:
            pass


def probe_duration_seconds(video_path: str) -> float:
    """Read the real duration from a downloaded file via ffprobe.

    Workaround for P104: some platforms' yt-dlp extractors never return
    `duration` anywhere in their metadata-only response -- confirmed for
    Instagram (neither the top-level info dict nor any per-format entry
    carries it), and not safely assumed fixed for other non-YouTube
    platforms either. It isn't an extraction bug to patch per-platform;
    the data simply doesn't exist until the file does. Once downloaded,
    ffprobe reads the real container duration directly, which works no
    matter which platform/extractor supplied (or failed to supply) it.

    Degrades to 0.0 on any failure (missing file, corrupt container,
    ffprobe unavailable) -- a quality-check input must never be the
    reason ingestion fails outright; callers already treat 0.0 as
    "duration unknown" (see ingestion_quality.check_visual_quality).
    """
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, check=True, text=True,
        )
        return float(out.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe duration probe failed for %s: %s", video_path, exc)
        return 0.0


def _to_meta(entry: dict) -> VideoMeta:
    return VideoMeta(
        video_id=entry.get("id", ""),
        title=entry.get("title", ""),
        duration=int(entry.get("duration") or 0),
        upload_date=entry.get("upload_date", ""),
        channel_id=entry.get("channel_id") or entry.get("uploader_id", ""),
        chapters=entry.get("chapters") or [],
        thumbnail_url=entry.get("thumbnail", ""),
    )


def _self_test():
    """Offline tests — no network, no yt-dlp needed."""
    # 1. _to_meta with full entry
    entry = {
        "id": "abc123", "title": "Test Video", "duration": 3600,
        "upload_date": "20261001", "channel_id": "UC123",
        "chapters": [{"title": "Intro", "start_time": 0}],
        "thumbnail": "https://img.youtube.com/vi/abc123/0.jpg",
    }
    meta = _to_meta(entry)
    assert meta.video_id == "abc123", f"video_id: {meta.video_id}"
    assert meta.duration == 3600
    assert len(meta.chapters) == 1
    print("ok 1 — _to_meta full entry")

    # 2. _to_meta with minimal entry (flat extraction)
    flat = {"id": "xyz789", "title": "Flat", "duration": None}
    meta2 = _to_meta(flat)
    assert meta2.video_id == "xyz789"
    assert meta2.duration == 0, "None duration should become 0"
    assert meta2.chapters == []
    assert meta2.channel_id == ""
    print("ok 2 — _to_meta minimal/flat entry")

    # 3. _to_meta with uploader_id fallback
    entry3 = {"id": "q", "title": "Q", "duration": 10, "uploader_id": "UC_fallback"}
    meta3 = _to_meta(entry3)
    assert meta3.channel_id == "UC_fallback"
    print("ok 3 — _to_meta uploader_id fallback")

    # 4. VideoMeta dataclass defaults
    vm = VideoMeta(video_id="t", title="T", duration=1, upload_date="20260101", channel_id="C")
    assert vm.chapters == []
    assert vm.thumbnail_url == ""
    print("ok 4 — VideoMeta defaults")

    # 5. _resolve_download_url: YouTube always reconstructs from video_id,
    # even if a (possibly stale) source_url is also on record.
    url5 = _resolve_download_url("abc123", platform="youtube", source_url="https://youtu.be/old")
    assert url5 == "https://www.youtube.com/watch?v=abc123", url5
    print("ok 5 — _resolve_download_url reconstructs YouTube URL from video_id")

    # 6. _resolve_download_url: non-YouTube platform uses the recorded source_url
    url6 = _resolve_download_url(
        "1234567890", platform="twitter",
        source_url="https://twitter.com/user/status/1234567890")
    assert url6 == "https://twitter.com/user/status/1234567890", url6
    print("ok 6 — _resolve_download_url uses source_url for non-YouTube platforms")

    # 7. _resolve_download_url: non-YouTube platform with no source_url can't
    # be downloaded — raise rather than silently building a wrong URL.
    try:
        _resolve_download_url("1234567890", platform="twitter", source_url="")
        raise AssertionError("expected ValueError for missing source_url")
    except ValueError:
        pass
    print("ok 7 — _resolve_download_url raises when non-YouTube source_url is missing")

    # 8. fetch_url_metadata parses a non-YouTube extractor response and tags
    # platform/source_url for later re-download.
    import sys
    import types

    fake_module = types.ModuleType("yt_dlp")

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            return {
                "id": "1234567890", "title": "A tweet video", "duration": 42,
                "extractor_key": "Twitter",
                "thumbnail": "https://pbs.twimg.com/thumb.jpg",
                "uploader_id": "some_user",
            }

    fake_module.YoutubeDL = FakeYDL
    sys.modules["yt_dlp"] = fake_module
    try:
        result = fetch_url_metadata("https://twitter.com/some_user/status/1234567890")
    finally:
        del sys.modules["yt_dlp"]

    assert result["video_id"] == "1234567890", result
    assert result["platform"] == "twitter", result
    assert result["source_url"] == "https://twitter.com/some_user/status/1234567890", result
    assert result["channel_id"] == "some_user", result
    assert result["duration"] == 42, result
    print("ok 8 — fetch_url_metadata parses non-YouTube extractor response")

    # 9. _base_ydl_opts always sets quiet/no_warnings and merges overrides
    base = _base_ydl_opts(extract_flat=True)
    assert base["quiet"] is True and base["no_warnings"] is True
    assert base["extract_flat"] is True
    print("ok 9 — _base_ydl_opts sets defaults and merges overrides")

    # 10. _base_ydl_opts degrades to no impersonation when the yt_dlp
    # networking submodule isn't importable (e.g. curl_cffi not installed) —
    # must never raise, just skip the key.
    saved_target = config.YTDLP_IMPERSONATE_TARGET
    config.YTDLP_IMPERSONATE_TARGET = "chrome"
    # Evict the real submodule cache entries too — `from a.b.c import X`
    # checks sys.modules["a.b.c"] directly, which survives swapping out the
    # top-level "yt_dlp" entry alone if it was ever imported for real earlier
    # in this process (e.g. by an earlier check in this same self-test).
    saved_submodules = {
        k: sys.modules.pop(k) for k in list(sys.modules)
        if k == "yt_dlp" or k.startswith("yt_dlp.")
    }
    sys.modules["yt_dlp"] = types.ModuleType("yt_dlp")  # no .networking submodule
    try:
        opts = _base_ydl_opts()
    finally:
        del sys.modules["yt_dlp"]
        sys.modules.update(saved_submodules)
        config.YTDLP_IMPERSONATE_TARGET = saved_target
    assert "impersonate" not in opts, opts
    print("ok 10 — _base_ydl_opts degrades safely when impersonation is unavailable")

    # 11. _base_ydl_opts sets impersonate when a target is configured and
    # the real yt_dlp (installed in this dev env) supports it.
    saved_target2 = config.YTDLP_IMPERSONATE_TARGET
    config.YTDLP_IMPERSONATE_TARGET = "chrome"
    try:
        opts2 = _base_ydl_opts()
    finally:
        config.YTDLP_IMPERSONATE_TARGET = saved_target2
    assert "impersonate" in opts2, "expected impersonate key when yt_dlp supports it locally"
    print("ok 11 — _base_ydl_opts sets impersonate when a target is configured")

    # 12. _base_ydl_opts ignores a configured cookies file that doesn't exist
    # on disk — never passes a dangling path to yt-dlp.
    saved_cookies = config.YTDLP_COOKIES_FILE
    config.YTDLP_COOKIES_FILE = "/nonexistent/cookies.txt"
    try:
        opts3 = _base_ydl_opts()
    finally:
        config.YTDLP_COOKIES_FILE = saved_cookies
    assert "cookiefile" not in opts3, opts3
    print("ok 12 — _base_ydl_opts ignores a cookies file that doesn't exist on disk")

    # 12b. A fresh, unused cookies file under cookie_guard's budget IS
    # attached -- the P91 cookies path is still live, just now gated.
    import tempfile as _tempfile_cookie
    with _tempfile_cookie.TemporaryDirectory() as _cookie_dir:
        cookies_path = os.path.join(_cookie_dir, "cookies.txt")
        with open(cookies_path, "w") as _f:
            _f.write("# netscape cookie file\n")
        saved_cookies2 = config.YTDLP_COOKIES_FILE
        config.YTDLP_COOKIES_FILE = cookies_path
        try:
            opts3b = _base_ydl_opts()
        finally:
            config.YTDLP_COOKIES_FILE = saved_cookies2
        attached = opts3b.get("cookiefile")
        assert attached, opts3b
        # P133: yt-dlp dumps its cookie jar back into whatever cookiefile it
        # was given when YoutubeDL closes, so it must get a throwaway copy,
        # never the canonical uploaded file.
        assert attached != cookies_path, ("canonical file handed to yt-dlp", attached)
        with open(attached) as _f:
            assert _f.read() == "# netscape cookie file\n"
        with open(attached, "w") as _f:  # what YoutubeDL.close() does
            _f.write("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tGPS\t1\n")
        with open(cookies_path) as _f:
            assert _f.read() == "# netscape cookie file\n", "canonical file was overwritten"
        print("ok 12b — _base_ydl_opts attaches a disposable copy; the canonical file survives yt-dlp's write-back")

        # 12c. Once cookie_guard's daily budget is exhausted, the SAME file
        # is withheld on the next call -- the guard actually gates reuse,
        # not just first-time absence.
        saved_max = config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY
        config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY = 1
        config.YTDLP_COOKIES_FILE = cookies_path
        try:
            opts3c = _base_ydl_opts()  # 2nd use this "day" -- over the cap of 1
        finally:
            config.YTDLP_COOKIES_FILE = saved_cookies2
            config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY = saved_max
        assert "cookiefile" not in opts3c, opts3c
        print("ok 12c — _base_ydl_opts withholds cookies once the daily budget is used up")

        # 12e. Only the 30 s spacing is unmet -> WAIT it out and attach cookies.
        # Withholding instead sent the call out anonymous, a guaranteed "Sign in
        # to confirm you're not a bot" on YouTube content (fetch-url 21 s after a
        # validate, 2026-10-19). The daily cap stays a hard limit (12c).
        from pipelines import cookie_guard as _cg12e
        st = _cg12e._load_state(cookies_path)
        st.update({"date": time.strftime("%Y-%m-%d"), "count": 1, "last_download_at": time.time() - 10})
        _cg12e._save_state(cookies_path, st)
        saved_sleep12e, slept12e = time.sleep, []

        def _fake_sleep12e(s):
            # A real sleep lets the clock run: age the last use by s.
            slept12e.append(s)
            st2 = _cg12e._load_state(cookies_path)
            st2["last_download_at"] -= s
            _cg12e._save_state(cookies_path, st2)
        time.sleep = _fake_sleep12e
        config.YTDLP_COOKIES_FILE = cookies_path
        try:
            opts3e = _base_ydl_opts()
        finally:
            time.sleep = saved_sleep12e
            config.YTDLP_COOKIES_FILE = saved_cookies2
        assert "cookiefile" in opts3e, opts3e
        assert len(slept12e) == 1 and 18 <= slept12e[0] <= config.YTDLP_COOKIES_MIN_DELAY_SECONDS, slept12e
        if os.path.exists(opts3e["cookiefile"]):
            os.remove(opts3e["cookiefile"])
        print("ok 12e — unmet spacing is waited out, cookies attached (not sent anonymous)")

        # 12d. _base_ydl_opts can't delete its copy (the caller owns the
        # YoutubeDL lifecycle), so old copies are swept on the next call --
        # one live credential copy per download must not pile up forever.
        stale = _disposable_cookie_copy(cookies_path)
        fresh = _disposable_cookie_copy(cookies_path)
        old = time.time() - _COOKIE_COPY_MAX_AGE_S - 60
        os.utime(stale, (old, old))
        newest = _disposable_cookie_copy(cookies_path)
        assert not os.path.exists(stale), "stale copy survived the sweep"
        assert os.path.exists(fresh), "a recent copy was swept too early"
        for _p in (fresh, newest, attached):
            if os.path.exists(_p):
                os.remove(_p)
        print("ok 12d — stale cookie copies are swept, recent ones are kept")

    # 13. _base_ydl_opts wires the PO-token provider base URL into
    # extractor_args when configured.
    saved_pot_url = config.YTDLP_POT_PROVIDER_URL
    config.YTDLP_POT_PROVIDER_URL = "http://bgutil-provider:4416"
    try:
        opts4 = _base_ydl_opts()
    finally:
        config.YTDLP_POT_PROVIDER_URL = saved_pot_url
    # P135: bgutil-ytdlp-pot-provider 1.3.x reads 'youtubepot-bgutilhttp:
    # base_url' and REJECTS the old 'youtube:getpot_bgutil_baseurl' (every
    # PO-token request failed with "extractor arg is deprecated").
    assert opts4["extractor_args"]["youtubepot-bgutilhttp"]["base_url"] == \
        ["http://bgutil-provider:4416"], opts4
    assert "getpot_bgutil_baseurl" not in str(opts4["extractor_args"]), opts4
    print("ok 13 — _base_ydl_opts wires the PO-token provider URL under the plugin's current arg name")

    # 14. _base_ydl_opts omits extractor_args entirely when the PO-token
    # provider URL is unset — no empty/broken config passed to yt-dlp.
    saved_pot_url2 = config.YTDLP_POT_PROVIDER_URL
    config.YTDLP_POT_PROVIDER_URL = ""
    try:
        opts5 = _base_ydl_opts()
    finally:
        config.YTDLP_POT_PROVIDER_URL = saved_pot_url2
    assert "extractor_args" not in opts5, opts5
    print("ok 14 — _base_ydl_opts omits extractor_args when the PO-token URL is unset")

    # 15. download_video's format string always has an unconditional final
    # fallback (plain "best"), so a platform whose resolution ladder never
    # drops to <=720p doesn't hard-fail with "Requested format is not
    # available". Found live 2026-10-12: an Instagram Reel's lowest
    # video-only stream was already 1280px (portrait height), above the
    # YouTube-oriented <=720 cap, and its combined-format options reported
    # no height metadata, so BOTH halves of the old selector
    # ("bestvideo[height<=720]+bestaudio/best[height<=720]") matched
    # nothing and the whole download failed outright.
    import sys as _sys
    import types as _types3

    captured_opts = {}

    class FakeYDL:
        def __init__(self, opts):
            captured_opts.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"id": "r1", "ext": "mp4"}

        def prepare_filename(self, info):
            return "/tmp/r1.mp4"

    fake_yt_dlp = _types3.ModuleType("yt_dlp")
    fake_yt_dlp.YoutubeDL = FakeYDL
    _sys.modules["yt_dlp"] = fake_yt_dlp
    try:
        import tempfile as _tempfile3
        with _tempfile3.TemporaryDirectory() as d:
            download_video("r1", d, platform="instagram",
                           source_url="https://www.instagram.com/reel/r1/")
    finally:
        del _sys.modules["yt_dlp"]

    assert captured_opts["format"].endswith("/best"), captured_opts["format"]
    # P137: the FIRST choice must be H.264 -- yt-dlp ranks AV1 above it, and
    # the image's OpenCV can't decode AV1, so _phase_visual read zero frames.
    first_choice = captured_opts["format"].split("/")[0]
    assert "vcodec^=avc1" in first_choice, captured_opts["format"]
    print("ok 15 — download_video's format selector always falls back to plain 'best'")
    # 15a. D-1080 (2026-10-27): the download cap is 1080p, not 720p. Frame test on 30
    # label-heavy frames: the unchanged OCR read 108/111 key labels with 1 wrong number
    # at 1080p vs 102/111 with 9 wrong at 720p (28013.75 read as 78013.75/20013.75);
    # Gemini bills both at the same 1,242 image tokens, so the cost is unchanged.
    assert "height<=1080" in first_choice and "720" not in captured_opts["format"],         captured_opts["format"]
    print("ok 15a — download cap is 1080p (H.264 first)")

    # 15b. Segmented (HLS/DASH) sources download 8 fragments at a time: X serves
    # a clip as ~355 small segments, fetched one by one by default -- 75.6 s vs
    # 12.3 s for the same 39 MB clip (benchmark 2026-10-19; 16 gave no more).
    assert captured_opts.get("concurrent_fragment_downloads") == 8, captured_opts
    print("ok 15b — downloads fetch 8 fragments in parallel")

    # 16. probe_duration_seconds parses ffprobe's stdout into a float.
    import subprocess as _subprocess

    class FakeCompleted:
        stdout = "63.483000\n"

    saved_run = _subprocess.run
    _subprocess.run = lambda *a, **kw: FakeCompleted()
    try:
        dur16 = probe_duration_seconds("/tmp/some_reel.mp4")
    finally:
        _subprocess.run = saved_run
    assert dur16 == 63.483, dur16
    print("ok 16 — probe_duration_seconds parses ffprobe's stdout")

    # 17. probe_duration_seconds degrades to 0.0 rather than raising --
    # a missing/corrupt file or absent ffprobe must never fail ingestion.
    def _raising_run(*a, **kw):
        raise FileNotFoundError("ffprobe not found")

    _subprocess.run = _raising_run
    try:
        dur17 = probe_duration_seconds("/tmp/missing.mp4")
    finally:
        _subprocess.run = saved_run
    assert dur17 == 0.0, dur17
    print("ok 17 — probe_duration_seconds degrades to 0.0 on failure")

    # 18. probe_cookies reports valid=True on a clean metadata fetch, and
    # records exactly one use -- validating cookies is itself a use of them.
    import sys as _sys_cookies
    import tempfile as _tempfile_probe
    import types as _types_cookies

    probe_calls = {}

    class FakeYDLOk:
        def __init__(self, opts):
            probe_calls["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            # The real YoutubeDL.close() calls save_cookies(), dumping its
            # jar into opts["cookiefile"] -- reproduce that (P133).
            with open(probe_calls["opts"]["cookiefile"], "w") as _cf:
                _cf.write("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tGPS\t1\n")
            return False

        def extract_info(self, url, download=False):
            probe_calls["url"] = url
            return {"id": "dQw4w9WgXcQ"}

    fake_yt_dlp_ok = _types_cookies.ModuleType("yt_dlp")
    fake_yt_dlp_ok.YoutubeDL = FakeYDLOk
    _sys_cookies.modules["yt_dlp"] = fake_yt_dlp_ok

    record_calls = []
    import pipelines.cookie_guard as _cookie_guard_mod
    saved_record = _cookie_guard_mod.record_cookie_use
    _cookie_guard_mod.record_cookie_use = lambda f: record_calls.append(f)
    try:
        with _tempfile_probe.TemporaryDirectory() as _d18:
            cfile = os.path.join(_d18, "cookies.txt")
            with open(cfile, "w") as _f:
                _f.write("# Netscape HTTP Cookie File\n")
            result18 = probe_cookies(cfile)
            with open(cfile) as _f:
                canonical_after = _f.read()
    finally:
        del _sys_cookies.modules["yt_dlp"]
        _cookie_guard_mod.record_cookie_use = saved_record
    assert result18 == {"valid": True, "error": None}, result18
    used_copy = probe_calls["opts"]["cookiefile"]
    assert used_copy != cfile, ("canonical file handed to yt-dlp", used_copy)
    assert canonical_after == "# Netscape HTTP Cookie File\n", "canonical file was overwritten"
    assert not os.path.exists(used_copy), "probe_cookies left its throwaway copy behind"
    assert record_calls == [cfile], record_calls  # usage is booked against the canonical file
    # Validation must exercise the same client path as a real download --
    # a probe without the PO-token arg tests a configuration nothing uses.
    if config.YTDLP_POT_PROVIDER_URL:
        assert probe_calls["opts"].get("extractor_args", {}).get(
            "youtubepot-bgutilhttp", {}).get("base_url") == [config.YTDLP_POT_PROVIDER_URL], probe_calls
    print("ok 18 — probe_cookies uses a throwaway copy, records one use, leaves the canonical file intact")

    # 19. probe_cookies translates YouTube's bot-detection message into a
    # clear, specific reason rather than a raw yt-dlp exception string.
    class FakeYDLBotCheck:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            raise Exception("ERROR: Sign in to confirm you're not a bot")

    fake_yt_dlp_bad = _types_cookies.ModuleType("yt_dlp")
    fake_yt_dlp_bad.YoutubeDL = FakeYDLBotCheck
    _sys_cookies.modules["yt_dlp"] = fake_yt_dlp_bad
    _cookie_guard_mod.record_cookie_use = lambda f: record_calls.append(f)
    try:
        with _tempfile_probe.TemporaryDirectory() as _d19:
            cfile2 = os.path.join(_d19, "cookies.txt")
            with open(cfile2, "w") as _f:
                _f.write("# Netscape HTTP Cookie File\n")
            result19 = probe_cookies(cfile2)
    finally:
        del _sys_cookies.modules["yt_dlp"]
        _cookie_guard_mod.record_cookie_use = saved_record
    assert result19["valid"] is False, result19
    assert "bot check" in result19["error"], result19
    print("ok 19 — probe_cookies reports a clear reason for a bot-detection failure")

    # 20. Whole channel, not just the /videos tab: the channel's live streams (165
    # videos, 281.6 h, 2026-10-19) were never catalogued -- the library said
    # 370 h against the ~622 h the client planned around. Every tab is listed,
    # de-duplicated, with the cookie spacing waited out between tabs.
    import types as _types20
    requested20 = []
    tab_entries = {
        "videos": [{"id": "v1", "title": "Lesson", "duration": 3600}],
        "streams": [{"id": "s1", "title": "Live", "duration": 7200},
                    {"id": "v1", "title": "Lesson", "duration": 3600}],   # same id on two tabs
        "shorts": [{"id": "h1", "title": "Short", "duration": None}],
    }

    class FakeYDL20:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            requested20.append(url)
            return {"entries": tab_entries[url.rstrip("/").rsplit("/", 1)[-1]]}

    fake20 = _types20.ModuleType("yt_dlp")
    fake20.YoutubeDL = FakeYDL20
    sys.modules["yt_dlp"] = fake20
    saved_sleep, slept = time.sleep, []
    time.sleep = slept.append
    try:
        listed = list_channel_videos("https://www.youtube.com/@InnerCircleTrader/videos")
    finally:
        del sys.modules["yt_dlp"]
        time.sleep = saved_sleep
    assert requested20 == ["https://www.youtube.com/@InnerCircleTrader/" + t
                           for t in ("videos", "streams", "shorts")], requested20
    assert [v["video_id"] for v in listed] == ["v1", "s1", "h1"], listed
    assert len(slept) == 2 and all(s >= config.YTDLP_COOKIES_MIN_DELAY_SECONDS for s in slept), slept
    print("ok 20 — channel discovery lists videos + streams + shorts, de-duplicated")

    # 20b. An optional tab that errors (no Shorts tab, say) must not lose the
    # rest of the listing; a failing /videos tab still raises.
    def _boom_shorts(self, url, download=False):
        if url.endswith("/shorts"):
            raise RuntimeError("This channel does not have a shorts tab")
        return {"entries": tab_entries[url.rsplit("/", 1)[-1]]}
    FakeYDL20.extract_info = _boom_shorts
    sys.modules["yt_dlp"] = fake20
    time.sleep = lambda s: None
    try:
        listed_b = list_channel_videos("https://www.youtube.com/@InnerCircleTrader")
    finally:
        del sys.modules["yt_dlp"]
        time.sleep = saved_sleep
    assert [v["video_id"] for v in listed_b] == ["v1", "s1"], listed_b
    print("ok 20b — a failing optional tab is skipped, the rest still listed")

    print("\n23 checks passed")


if __name__ == "__main__":
    _self_test()
