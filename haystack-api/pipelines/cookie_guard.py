"""Cookie-usage guardrail for yt-dlp's YouTube cookie authentication path.

Cookies (P91's still-open gap) let yt-dlp present an authenticated browser
session instead of a bare script, which is what actually defeats YouTube's
"sign in to confirm you're not a bot" challenge. But there is no official
published rate limit, and community/vendor guidance (researched 2026-10-14)
agrees a real account used this way risks being rate-limited, forced through
verification, or banned outright -- so cookie usage gets its own budget,
separate from the general download path, the same way D68 caps ingestion
spend.

State is a small JSON file next to the cookies file itself, not Postgres --
every call here runs synchronously inside ingest_video()'s worker thread
(_phase_download -> download_video -> _base_ydl_opts), and threading
eval_db's async pool through a sync call chain is exactly the cross-
event-loop crash P95 cost a day to fix. A local file degrades the same way
the rest of this codebase degrades a database hiccup: if it can't be read
or written, the check allows the request rather than blocking ingestion
over a guardrail's own plumbing.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import config

logger = logging.getLogger("haystack-api.cookie_guard")


def _state_path(cookies_file: str) -> str:
    return cookies_file + ".usage.json"


def _load_state(cookies_file: str) -> dict:
    try:
        with open(_state_path(cookies_file)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(cookies_file: str, state: dict) -> None:
    try:
        with open(_state_path(cookies_file), "w") as f:
            json.dump(state, f)
    except OSError as exc:
        logger.warning("could not persist cookie usage state: %s", exc)


def cookie_file_age_days(cookies_file: str) -> float:
    """Days since the cookies file was last modified (i.e. last re-exported).

    inf if the file doesn't exist -- callers treat that as "always expired."
    """
    try:
        return (time.time() - os.path.getmtime(cookies_file)) / 86400.0
    except OSError:
        return float("inf")


def _today_count(state: dict) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    return state.get("count", 0) if state.get("date") == today else 0


def check_cookie_budget(cookies_file: str) -> tuple:
    """Whether a yt-dlp call may present cookies right now.

    Returns (ok, reason) -- reason is "" when ok. Checks, in order: the file
    exists and isn't past YTDLP_COOKIES_MAX_AGE_DAYS, today's count is under
    YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY, and at least
    YTDLP_COOKIES_MIN_DELAY_SECONDS has passed since the last use.

    A failed check never blocks the yt-dlp call itself -- only whether
    cookies are attached to it. The existing impersonate/PO-token defenses
    (P91) apply regardless of this guard's verdict.
    """
    if not cookies_file or not os.path.isfile(cookies_file):
        return False, "no cookies file configured"

    age = cookie_file_age_days(cookies_file)
    if age > config.YTDLP_COOKIES_MAX_AGE_DAYS:
        return False, (
            "cookies.txt is %.1f days old, over the %d-day limit -- re-export it"
            % (age, config.YTDLP_COOKIES_MAX_AGE_DAYS)
        )

    state = _load_state(cookies_file)
    count_today = _today_count(state)
    if count_today >= config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY:
        return False, (
            "cookie daily budget reached (%d/%d) -- try again tomorrow"
            % (count_today, config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY)
        )

    warn_at = config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY * config.YTDLP_COOKIES_WARN_THRESHOLD_PCT
    if count_today >= warn_at:
        logger.warning("cookie usage at %d/%d today -- approaching the daily budget",
                       count_today, config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY)

    last_at = state.get("last_download_at")
    if last_at is not None:
        elapsed = time.time() - last_at
        if elapsed < config.YTDLP_COOKIES_MIN_DELAY_SECONDS:
            return False, (
                "only %.0fs since the last cookie-authenticated request, need "
                "%ds between them (%.0fs left)"
                % (elapsed, config.YTDLP_COOKIES_MIN_DELAY_SECONDS,
                   config.YTDLP_COOKIES_MIN_DELAY_SECONDS - elapsed)
            )

    return True, ""


def spacing_wait_seconds(cookies_file: str) -> float:
    """Seconds still to wait before the min-delay spacing allows a cookie use
    (0.0 when it already does, or when the file has never been used)."""
    if not cookies_file or not os.path.isfile(cookies_file):
        return 0.0
    last_at = _load_state(cookies_file).get("last_download_at")
    if last_at is None:
        return 0.0
    return max(0.0, config.YTDLP_COOKIES_MIN_DELAY_SECONDS - (time.time() - last_at))


def record_cookie_use(cookies_file: str) -> None:
    """Call immediately after a yt-dlp request actually presents cookies --
    counts toward today's budget and resets the minimum-delay clock.

    Merges into the existing state rather than replacing it outright --
    record_validation() stores a result in the same file, and a real
    download recorded afterward must not silently erase that.
    """
    state = _load_state(cookies_file)
    state.update({
        "date": datetime.now(timezone.utc).date().isoformat(),
        "count": _today_count(state) + 1,
        "last_download_at": time.time(),
    })
    _save_state(cookies_file, state)


def cookie_fingerprint(cookies_file: str) -> str:
    """A short, stable identifier for 'is this the same cookies file as
    last time' -- an 8-char content hash, never anything derived in a way
    that could leak the actual cookie values back out."""
    try:
        with open(cookies_file, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except OSError:
        return ""


def record_validation(cookies_file: str, valid: bool, error: str = None) -> None:
    """Persist the result of a probe_cookies() live check, so the admin UI
    can show it without re-probing YouTube on every page load."""
    state = _load_state(cookies_file)
    state["last_validation"] = {
        "valid": valid, "error": error,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(cookies_file, state)


def get_cookie_status(cookies_file: str) -> dict:
    """Everything the admin UI needs to show cookie health -- never the
    cookie content itself, which stays off every API response."""
    configured = bool(cookies_file) and os.path.isfile(cookies_file)
    if not configured:
        return {
            "configured": False, "age_days": None,
            "max_age_days": config.YTDLP_COOKIES_MAX_AGE_DAYS,
            "usage_today": 0, "daily_cap": config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY,
            "last_used_at": None, "fingerprint": None, "last_validation": None,
        }
    state = _load_state(cookies_file)
    last_at = state.get("last_download_at")
    return {
        "configured": True,
        "age_days": round(cookie_file_age_days(cookies_file), 1),
        "max_age_days": config.YTDLP_COOKIES_MAX_AGE_DAYS,
        "usage_today": _today_count(state),
        "daily_cap": config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY,
        "last_used_at": (
            datetime.fromtimestamp(last_at, tz=timezone.utc).isoformat() if last_at else None
        ),
        "fingerprint": cookie_fingerprint(cookies_file),
        "last_validation": state.get("last_validation"),
    }


def _looks_like_netscape_cookie_file(content: str) -> bool:
    """A light sanity check, not a full parser -- just enough to reject an
    obviously-wrong paste (a transcript, an HTML page, an empty upload)
    before it silently becomes a cookies file nothing will ever use.
    Netscape format: tab-separated, 7 fields per data line, comments start
    with '#'.
    """
    data_lines = [
        line for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not data_lines:
        return False
    return any(len(line.split("\t")) >= 6 for line in data_lines)


# Far-future expiry (2100-01-01) for a JSON-exported cookie that has no
# expiration -- usually a session cookie, which browser extensions often
# omit a timestamp for entirely. A row with no expiry reads as "already
# expired" to a Netscape-format parser, which would make the converted
# file useless the moment it's written.
_FAR_FUTURE_EXPIRY = 4102444800  # 2100-01-01T00:00:00Z

# Any one of these means the export carries an actual logged-in session.
# Modern Chrome exports often have only the __Secure- variants, not bare SID.
_SESSION_COOKIE_NAMES = frozenset({"SID", "__Secure-1PSID", "__Secure-3PSID"})


def _netscape_line(domain: str, path: str, secure, expiry, name: str, value: str) -> str:
    include_subdomains = domain.startswith(".")
    return "\t".join([
        domain,
        "TRUE" if include_subdomains else "FALSE",
        path or "/",
        "TRUE" if secure else "FALSE",
        str(int(float(expiry))),
        name,
        value,
    ])


def _json_cookies_to_netscape(data) -> str:
    """Convert a browser extension's JSON cookie export (Cookie-Editor,
    EditThisCookie, and similar all use the same {domain, path, secure,
    expirationDate, name, value} shape) into the Netscape format yt-dlp's
    `--cookies` actually requires -- yt-dlp has no JSON cookie support at
    all, so this conversion is the only way a JSON paste can ever work.

    Tolerant of the common shape variations: a bare array, a dict wrapping
    it under "cookies", and either "expirationDate" or "expiry" for the
    timestamp field. A row missing domain or name is skipped rather than
    raising -- one malformed entry in an otherwise-good export shouldn't
    sink the whole paste.
    """
    if isinstance(data, dict):
        data = data.get("cookies", [])
    if not isinstance(data, list):
        raise ValueError("JSON cookie export must be an array (or {\"cookies\": [...]})")

    lines = ["# Netscape HTTP Cookie File"]
    for entry in data:
        if not isinstance(entry, dict):
            continue
        domain = entry.get("domain", "")
        name = entry.get("name", "")
        if not domain or not name:
            continue
        expiry = entry.get("expirationDate") or entry.get("expiry") or _FAR_FUTURE_EXPIRY
        lines.append(_netscape_line(
            domain, entry.get("path", "/"), entry.get("secure", False),
            expiry, name, entry.get("value", ""),
        ))
    return "\n".join(lines) + "\n"


def save_cookie_file(content: str, cookies_file: str) -> None:
    """Write a freshly-exported cookies.txt, replacing whatever was there.

    Accepts either Netscape-format text directly, or a JSON cookie export
    (from Cookie-Editor, EditThisCookie, or similar) -- JSON is detected and
    converted to Netscape format automatically, since that's the only shape
    yt-dlp's cookiefile option understands.

    Raises ValueError on content that doesn't look like either -- never
    silently saves something that will just fail quietly inside yt-dlp
    later. Clears any prior usage state: a newly-saved cookie set is a
    fresh credential, not a continuation of the old one's daily count or
    delay clock.
    """
    stripped = content.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("looks like JSON but failed to parse: %s" % exc)
        content = _json_cookies_to_netscape(data)

    if not _looks_like_netscape_cookie_file(content):
        raise ValueError(
            "doesn't look like a Netscape-format cookies.txt file or a "
            "recognizable JSON cookie export (expected tab-separated fields "
            "or a JSON array of {domain, name, value, ...} objects)"
        )
    # P134: the SID-family cookies are what actually carry a YouTube login,
    # and all of them are HttpOnly -- so any export that reads cookies the
    # way page JavaScript does (document.cookie) silently leaves every one
    # of them out. A set with only SAPISID-type cookies makes yt-dlp send a
    # signed "logged in" header with no session behind it, and YouTube
    # answers with the cryptic "The page needs to be reloaded." Catch it
    # here, before anything is written or a daily validation slot is spent.
    names = {
        line.split("\t")[5] for line in content.splitlines()
        if line.strip() and not line.startswith("#") and len(line.split("\t")) >= 6
    }
    if not names & _SESSION_COOKIE_NAMES:
        raise ValueError(
            "this export has no YouTube session cookie (%s) -- those are "
            "HttpOnly, and the tool that produced it couldn't see HttpOnly "
            "cookies. Export again with an extension that can (Cookie-Editor, "
            "or Get cookies.txt LOCALLY) while logged in to youtube.com"
            % ", ".join(sorted(_SESSION_COOKIE_NAMES))
        )
    os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
    with open(cookies_file, "w") as f:
        f.write(content)
    try:
        os.remove(_state_path(cookies_file))
    except FileNotFoundError:
        pass


def remove_cookie_file(cookies_file: str) -> None:
    """Delete the cookies file and its usage state, if either exists."""
    for path in (cookies_file, _state_path(cookies_file)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _self_test():
    import tempfile

    saved = (config.YTDLP_COOKIES_MAX_AGE_DAYS, config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY,
             config.YTDLP_COOKIES_MIN_DELAY_SECONDS, config.YTDLP_COOKIES_WARN_THRESHOLD_PCT)
    config.YTDLP_COOKIES_MAX_AGE_DAYS = 14
    config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY = 3
    config.YTDLP_COOKIES_MIN_DELAY_SECONDS = 30
    config.YTDLP_COOKIES_WARN_THRESHOLD_PCT = 0.8

    try:
        with tempfile.TemporaryDirectory() as d:
            cookies_file = os.path.join(d, "cookies.txt")

            # 1. No file at all -- withheld, not an error.
            ok, reason = check_cookie_budget(os.path.join(d, "missing.txt"))
            assert ok is False and "no cookies file" in reason, (ok, reason)
            print("ok 1 — missing cookies file withholds cookies, no exception")

            # 2. A fresh file with no usage history is allowed.
            with open(cookies_file, "w") as f:
                f.write("# netscape cookie file\n")
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is True and reason == "", (ok, reason)
            print("ok 2 — a fresh, unused cookies file is allowed")

            # 3. Recording usage increments today's count and is visible
            # to the next check via the daily budget.
            record_cookie_use(cookies_file)
            state = _load_state(cookies_file)
            assert state["count"] == 1, state
            print("ok 3 — record_cookie_use persists a count for today")

            # 4. Too soon after the last use -- withheld with a delay reason.
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is False and "between them" in reason, (ok, reason)
            print("ok 4 — a request inside the minimum delay is withheld")

            # 5. Once enough time has passed (simulated, not slept), the
            # delay check clears and the daily count carries over.
            state["last_download_at"] = time.time() - 31
            _save_state(cookies_file, state)
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is True and reason == "", (ok, reason)
            print("ok 5 — the delay check clears once enough time has passed")

            # 6. Daily budget reached -- withheld regardless of delay.
            state["count"] = 3
            _save_state(cookies_file, state)
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is False and "daily budget reached" in reason, (ok, reason)
            print("ok 6 — the daily budget caps usage even with no delay issue")

            # 7. A new UTC day resets the count to zero, independent of the
            # stale count left over from a previous day.
            state = {"date": "2020-01-01", "count": 999, "last_download_at": 0}
            _save_state(cookies_file, state)
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is True and reason == "", (ok, reason)
            print("ok 7 — a new day resets the daily count instead of carrying it over")

            # 8. A stale (over-age) cookies file is withheld even with zero
            # usage today -- the age check is independent of the budget.
            old_time = time.time() - (config.YTDLP_COOKIES_MAX_AGE_DAYS + 1) * 86400
            os.utime(cookies_file, (old_time, old_time))
            os.remove(cookies_file + ".usage.json")
            ok, reason = check_cookie_budget(cookies_file)
            assert ok is False and "days old" in reason, (ok, reason)
            print("ok 8 — a cookies file past the max age is withheld for re-export")

            # 9. A missing/corrupt state file degrades to "no usage yet"
            # rather than raising -- a guardrail must not itself break
            # ingestion over its own bookkeeping.
            with open(cookies_file + ".usage.json", "w") as f:
                f.write("not json")
            assert _load_state(cookies_file) == {}
            print("ok 9 — a corrupt usage state file degrades to empty, not an exception")

            # 10. get_cookie_status reports "not configured" for a missing file.
            status = get_cookie_status(os.path.join(d, "missing.txt"))
            assert status == {
                "configured": False, "age_days": None,
                "max_age_days": 14, "usage_today": 0, "daily_cap": 3,
                "last_used_at": None, "fingerprint": None, "last_validation": None,
            }, status
            print("ok 10 — get_cookie_status reports 'not configured' for a missing file")

            # 11. get_cookie_status reports real numbers for a configured,
            # used file -- never the cookie content itself.
            os.remove(cookies_file + ".usage.json")
            os.utime(cookies_file, (time.time(), time.time()))
            record_cookie_use(cookies_file)
            status2 = get_cookie_status(cookies_file)
            assert status2["configured"] is True, status2
            assert status2["usage_today"] == 1, status2
            assert status2["daily_cap"] == 3, status2
            assert status2["last_used_at"] is not None, status2
            assert len(status2["fingerprint"]) == 8, status2
            assert status2["last_validation"] is None, status2
            print("ok 11 — get_cookie_status reports real usage numbers, never content")

            # 11b. The same file always hashes to the same fingerprint, and a
            # different file hashes differently -- this is the signal the
            # admin UI uses to answer "is this the cookies I uploaded, or
            # has someone else replaced them since."
            fp_a = cookie_fingerprint(cookies_file)
            fp_b = cookie_fingerprint(cookies_file)
            assert fp_a == fp_b and len(fp_a) == 8, (fp_a, fp_b)
            with open(cookies_file, "a") as f:
                f.write("extra-line\n")
            assert cookie_fingerprint(cookies_file) != fp_a
            print("ok 11b — cookie_fingerprint is stable for the same content, changes when it differs")

            # 11c. record_validation persists a result without clobbering
            # the usage count a sibling record_cookie_use() call already
            # wrote -- both share one state file.
            record_validation(cookies_file, valid=False, error="bot check")
            status3 = get_cookie_status(cookies_file)
            assert status3["usage_today"] == 1, status3  # untouched by record_validation
            assert status3["last_validation"]["valid"] is False, status3
            assert status3["last_validation"]["error"] == "bot check", status3
            record_cookie_use(cookies_file)
            status4 = get_cookie_status(cookies_file)
            assert status4["usage_today"] == 2, status4
            assert status4["last_validation"]["valid"] is False, (
                "record_cookie_use must not erase a prior validation result", status4)
            print("ok 11c — record_validation and record_cookie_use share state without clobbering each other")

            # 12. save_cookie_file rejects content that doesn't look like a
            # Netscape cookie file -- never silently saves garbage.
            try:
                save_cookie_file("this is not a cookie file", cookies_file)
                raise AssertionError("expected ValueError for non-cookie content")
            except ValueError as exc:
                assert "Netscape" in str(exc), exc
            print("ok 12 — save_cookie_file rejects non-cookie-shaped content")

            # 13. save_cookie_file accepts real Netscape-format content,
            # writes it, and clears any prior usage state -- a fresh upload
            # is a fresh credential, not a continuation of the old budget.
            netscape = (
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc123value\n"
            )
            new_path = os.path.join(d, "new_cookies.txt")
            save_cookie_file(netscape, new_path)
            assert os.path.isfile(new_path)
            with open(new_path) as f:
                assert f.read() == netscape
            assert not os.path.isfile(_state_path(new_path))
            print("ok 13 — save_cookie_file writes valid content and resets usage state")

            # 14. remove_cookie_file deletes both the cookies file and its
            # usage state, degrading safely if either is already gone.
            record_cookie_use(new_path)
            assert os.path.isfile(_state_path(new_path))
            remove_cookie_file(new_path)
            assert not os.path.isfile(new_path)
            assert not os.path.isfile(_state_path(new_path))
            remove_cookie_file(new_path)  # already gone -- must not raise
            print("ok 14 — remove_cookie_file deletes the file and its state, idempotently")

            # 15. save_cookie_file detects a JSON cookie export (Cookie-Editor/
            # EditThisCookie shape) and converts it to Netscape format, since
            # that's the only thing yt-dlp's cookiefile option understands.
            json_path = os.path.join(d, "json_cookies.txt")
            cookie_json = json.dumps([
                {"domain": ".youtube.com", "path": "/", "secure": True,
                 "expirationDate": 1999999999, "name": "SID", "value": "sid-value"},
                {"domain": ".youtube.com", "path": "/", "secure": True,
                 "name": "SESSION_ONLY", "value": "no-expiry-given"},  # session cookie, no expiry
                {"domain": "youtube.com", "path": "/", "secure": False,
                 "name": "", "value": "skip-me-no-name"},  # must be skipped, not crash
            ])
            save_cookie_file(cookie_json, json_path)
            with open(json_path) as f:
                written = f.read()
            assert written.startswith("# Netscape HTTP Cookie File"), written
            lines15 = [l for l in written.splitlines() if l and not l.startswith("#")]
            assert len(lines15) == 2, (len(lines15), written)  # the no-name entry was skipped
            assert ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tsid-value" in written, written
            # The session cookie (no expirationDate) got the far-future
            # fallback, not a crash or a "0" that would read as pre-expired.
            assert "SESSION_ONLY\tno-expiry-given" in written, written
            assert "\t0\tSESSION_ONLY" not in written, written
            print("ok 15 — save_cookie_file converts a JSON cookie export to Netscape format")

            # 16. The {"cookies": [...]} wrapper shape (some extensions nest
            # the array) is accepted the same way as a bare array.
            wrapped_path = os.path.join(d, "wrapped.txt")
            save_cookie_file(json.dumps({"cookies": [
                {"domain": ".youtube.com", "path": "/", "secure": True,
                 "expiry": 1999999999, "name": "HSID", "value": "hsid-value"},
                {"domain": ".youtube.com", "path": "/", "secure": True,
                 "expiry": 1999999999, "name": "__Secure-1PSID", "value": "psid-value"},
            ]}), wrapped_path)
            with open(wrapped_path) as f:
                assert "HSID\thsid-value" in f.read()
            print("ok 16 — save_cookie_file accepts the {'cookies': [...]} wrapper shape too")

            # 17. JSON that fails to parse gets a specific error, not the
            # generic "doesn't look like Netscape" message -- the two
            # failure modes have different fixes (fix the JSON vs. paste
            # the right thing at all).
            try:
                save_cookie_file("[{not valid json", os.path.join(d, "bad.txt"))
                raise AssertionError("expected ValueError for malformed JSON")
            except ValueError as exc:
                assert "failed to parse" in str(exc), exc
            print("ok 17 — malformed JSON gets a specific parse-error message")

            # 18. An export with no session (SID-family) cookie is rejected
            # before anything is written -- the exact 4-cookie set pasted
            # 2026-10-19 (only the non-HttpOnly ones), which saved, burned a
            # validation slot, and came back as YouTube's cryptic "The page
            # needs to be reloaded" (P134). The previous good file survives.
            keep_path = os.path.join(d, "keep.txt")
            save_cookie_file(netscape, keep_path)
            partial = json.dumps([
                {"domain": ".youtube.com", "name": n, "value": "v", "path": "/",
                 "secure": True, "httpOnly": False, "expirationDate": 1999999999}
                for n in ("PREF", "SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID")
            ])
            try:
                save_cookie_file(partial, keep_path)
                raise AssertionError("expected ValueError for an export with no session cookie")
            except ValueError as exc:
                assert "HttpOnly" in str(exc) and "__Secure-1PSID" in str(exc), exc
            with open(keep_path) as f:
                assert f.read() == netscape, "rejected export overwrote the previous file"
            print("ok 18 — an export missing the HttpOnly session cookies is rejected, nothing overwritten")
    finally:
        (config.YTDLP_COOKIES_MAX_AGE_DAYS, config.YTDLP_COOKIES_MAX_DOWNLOADS_PER_DAY,
         config.YTDLP_COOKIES_MIN_DELAY_SECONDS, config.YTDLP_COOKIES_WARN_THRESHOLD_PCT) = saved

    print("\n20 checks passed")


if __name__ == "__main__":
    _self_test()
