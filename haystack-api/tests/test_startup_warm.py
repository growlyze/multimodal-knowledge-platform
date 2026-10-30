"""
Planner warm start (P181 follow-up, approved 2026-10-25).

The first video question after a restart ran unplanned: building the Gemini and
Claude clients (imports + TLS) took 3.1-4.2 s, over PLANNER_BUDGET_S (3 s), so the
question was searched as typed -- live 2026-10-25 18:50 a definition question
never reached the named episode. main.py's lifespan now warms the planner in a
background task. What must hold:

  - the warm-up is scheduled at startup, and startup never waits for it;
  - a failing or hanging warm-up never fails startup, and shutdown cancels it;
  - warm_up() lists the indexed videos and calls each configured provider once,
    without touching the circuit breaker or the plan cache.

    PYTHONPATH=. python tests/test_startup_warm.py

Offline: no Qdrant, no Gemini, no Claude -- every boundary is faked.
"""
import asyncio
import os
import sys
import time

os.environ.setdefault("EVAL_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
os.environ.setdefault("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")

import main
from pipelines import query_planner

_count = 0


def ok(msg):
    global _count
    _count += 1
    print("ok  %s" % msg)


async def _run_lifespan(fake_warm, after_start=None):
    saved = query_planner.warm_up
    query_planner.warm_up = fake_warm
    try:
        started = time.monotonic()
        async with main.lifespan(main.app):
            startup_s = time.monotonic() - started
            if after_start:
                await after_start()
        return startup_s
    finally:
        query_planner.warm_up = saved


def test_warm_up_is_scheduled_without_blocking_startup():
    calls = []

    async def fake():
        calls.append("warm")
        return {"videos": 4}

    async def let_it_run():
        await asyncio.sleep(0.05)

    asyncio.run(_run_lifespan(fake, let_it_run))
    assert calls == ["warm"], calls
    ok("startup schedules the planner warm-up")


def test_hanging_warm_up_never_holds_startup_and_is_cancelled():
    state = {}

    async def hang():
        state["started"] = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def let_it_start():
        await asyncio.sleep(0.05)

    startup_s = asyncio.run(_run_lifespan(hang, let_it_start))
    assert state.get("started") and state.get("cancelled"), state
    # Startup also opens the (unreachable) database pool; the warm-up adds nothing.
    assert startup_s < main.PLANNER_WARM_TIMEOUT_S, startup_s
    ok("a hanging warm-up does not hold startup; shutdown cancels it")


def test_failing_warm_up_is_swallowed():
    async def boom():
        raise RuntimeError("gemini down")

    async def let_it_fail():
        await asyncio.sleep(0.05)

    asyncio.run(_run_lifespan(boom, let_it_fail))   # must not raise
    ok("a failing warm-up is logged, never raised")


def test_warm_up_calls_each_provider_once_and_leaves_no_state():
    import config
    calls = []
    saved = (query_planner.indexed_videos, query_planner._generate, query_planner._generate_fallback,
             config.PLANNER_ANTHROPIC_API_KEY, os.environ.get("GOOGLE_API_KEY"))

    async def videos():
        calls.append("videos")
        return [("E12", "Episode 12")]

    def gemini(prompt):
        calls.append("gemini")
        raise RuntimeError("402 RESOURCE_EXHAUSTED")

    def claude(prompt, timeout=None):
        calls.append(("claude", timeout))
        return {"queries": ["x"], "video_name": None, "intent": "passage"}

    query_planner.indexed_videos, query_planner._generate, query_planner._generate_fallback = videos, gemini, claude
    config.PLANNER_ANTHROPIC_API_KEY = "test-key"
    os.environ["GOOGLE_API_KEY"] = "test-key"
    query_planner._breaker.reset()
    query_planner._cache.clear()
    try:
        result = asyncio.run(query_planner.warm_up())
    finally:
        (query_planner.indexed_videos, query_planner._generate, query_planner._generate_fallback,
         config.PLANNER_ANTHROPIC_API_KEY, google) = saved
        if google is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = google
    assert calls[0] == "videos" and "gemini" in calls, calls
    assert [c for c in calls if isinstance(c, tuple)] == [("claude", query_planner.WARM_CLAUDE_TIMEOUT_S)], calls
    assert result["videos"] == 1 and result["claude"] == "ok" and result["gemini"].startswith("RuntimeError"), result
    # A billing error during warm-up must not open the circuit for real turns,
    # and the warm-up prompt is never served from the cache as a plan.
    assert query_planner._breaker.allow() and query_planner._breaker.failures == 0
    assert not query_planner._cache
    ok("warm_up: videos listed, each provider called once, breaker and cache untouched")


def test_warm_up_skips_unconfigured_providers():
    import config
    calls = []
    saved = (query_planner.indexed_videos, query_planner._generate, query_planner._generate_fallback,
             config.PLANNER_ANTHROPIC_API_KEY, os.environ.pop("GOOGLE_API_KEY", None),
             os.environ.pop("GEMINI_API_KEY", None))

    async def videos():
        return []

    query_planner.indexed_videos = videos
    query_planner._generate = lambda p: calls.append("gemini")
    query_planner._generate_fallback = lambda p, timeout=None: calls.append("claude")
    config.PLANNER_ANTHROPIC_API_KEY = ""
    try:
        result = asyncio.run(query_planner.warm_up())
    finally:
        (query_planner.indexed_videos, query_planner._generate, query_planner._generate_fallback,
         config.PLANNER_ANTHROPIC_API_KEY, google, gemini_key) = saved
        if google is not None:
            os.environ["GOOGLE_API_KEY"] = google
        if gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = gemini_key
    assert calls == [], calls
    assert result == {"videos": 0, "gemini": "not configured", "claude": "not configured"}, result
    ok("warm_up: an unconfigured provider is skipped, not called")


def main_():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\n%d warm-start checks passed" % _count)


if __name__ == "__main__":
    main_()
