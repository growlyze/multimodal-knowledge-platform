"""D68 safety layer — running cost ledger and spend ceiling for M3 ingestion.

Degrade-never-break, same convention as the rest of this codebase: when
eval_db is unavailable, check_budget() allows processing to continue rather
than silently stalling the whole pipeline on a database hiccup — a cost
overrun from a few missed checks is recoverable, a stuck queue is not.
"""

import json
import logging

import config

logger = logging.getLogger("haystack-api.cost_tracker")


def _estimate(quantity: float, rate_per_unit: float) -> float:
    """quantity * rate — the one thing every estimate_* function below does.
    Consolidated per code review 2026-10-09: three copy-pasted one-liners
    risked a typo (e.g. the wrong quantity) going uncaught outside their
    own self-test."""
    return quantity * rate_per_unit


def estimate_groq_cost(duration_s: float) -> float:
    """Groq ASR cost for one video, by audio duration."""
    return _estimate(duration_s / 3600.0, config.GROQ_COST_PER_HOUR_USD)


def estimate_voyage_cost(frame_count: int) -> float:
    """Voyage multimodal (image) embedding cost for one video, by keyframe count."""
    return _estimate(frame_count, config.VOYAGE_COST_PER_FRAME_USD)


def estimate_voyage_text_cost(item_count: int) -> float:
    """Voyage text embedding cost — OCR text, OD descriptions, and transcript
    segments all go through embed_texts_voyage(), a different (much cheaper)
    rate than the image-based multimodal calls estimate_voyage_cost() covers.
    Added in code review 2026-10-09: the cost ledger previously only counted
    keyframe/image embeddings, silently undercounting real Voyage spend and
    weakening the D68 spend-ceiling safety check.
    """
    return _estimate(item_count, config.VOYAGE_TEXT_COST_PER_ITEM_USD)


def estimate_vision_cost(frame_count: int) -> float:
    """Gemini vision-description cost for one video, by chart-frame count."""
    return _estimate(frame_count, config.VISION_COST_PER_FRAME_USD)


async def record_cost(
    eval_db, video_id: str,
    groq_cost: float = 0.0, voyage_cost: float = 0.0, vision_cost: float = 0.0,
) -> None:
    """Upsert this video's cost row. No-op if eval_db is unavailable."""
    if eval_db is None:
        return
    await eval_db.execute(
        "INSERT INTO video_ingest_costs (video_id, groq_cost_usd, voyage_cost_usd, vision_cost_usd) "
        "VALUES ($1, $2, $3, $4) "
        "ON CONFLICT (video_id) DO UPDATE SET "
        "groq_cost_usd = EXCLUDED.groq_cost_usd, voyage_cost_usd = EXCLUDED.voyage_cost_usd, "
        "vision_cost_usd = EXCLUDED.vision_cost_usd, recorded_at = NOW()",
        video_id, groq_cost, voyage_cost, vision_cost,
    )


async def record_usage(eval_db, video_id: str, usage: dict, phase_seconds: dict,
                       status: str) -> None:
    """Store one run's measured usage (usage_meter summary) and per-stage
    wall-clock, and write its MEASURED dollars into video_ingest_costs -- the
    ledger check_budget() reads -- replacing the flat per-item estimates.
    Called for failed runs too: the phases before a failure were paid for.
    No-op if eval_db is unavailable."""
    if eval_db is None:
        return
    await eval_db.execute(
        "INSERT INTO video_ingest_usage "
        "(video_id, usage, phase_seconds, prices_as_of, total_usd, status) "
        "VALUES ($1, $2, $3, $4, $5, $6) "
        "ON CONFLICT (video_id) DO UPDATE SET usage = EXCLUDED.usage, "
        "phase_seconds = EXCLUDED.phase_seconds, prices_as_of = EXCLUDED.prices_as_of, "
        "total_usd = EXCLUDED.total_usd, status = EXCLUDED.status, recorded_at = NOW()",
        video_id, json.dumps(usage), json.dumps(phase_seconds or {}),
        usage["prices_as_of"], usage["total_usd"], status,
    )
    await record_cost(
        eval_db, video_id,
        groq_cost=usage["groq"]["usd"],
        voyage_cost=usage["voyage"]["usd"],
        vision_cost=usage["gemini"]["usd"],
    )


def _as_dict(value):
    """JSONB arrives as str from asyncpg (no codec registered) or as a dict."""
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def _sum_usage(base, extra):
    """Two usage_meter summaries added field by field: numbers sum, nested
    dicts recurse, anything else (video_id, prices_as_of) keeps base's value
    unless base lacks it."""
    out = dict(base)
    for key, value in extra.items():
        mine = out.get(key)
        if isinstance(value, dict) and isinstance(mine, dict):
            out[key] = _sum_usage(mine, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and isinstance(mine, (int, float)) and not isinstance(mine, bool):
            out[key] = mine + value
        elif key not in out:
            out[key] = value
    return out


async def add_usage(eval_db, video_id: str, extra_usage: dict) -> None:
    """Add a later run's metered usage (e.g. the transcript re-index) to the
    video's existing video_ingest_usage row, keeping its phase times and
    status; with no row yet, store it as status "reindex". The table holds one
    row per video, so writing the extra usage with record_usage() would have
    replaced the real ingestion figures. No-op if eval_db is unavailable."""
    if eval_db is None:
        return
    row = await eval_db.fetchrow(
        "SELECT usage, phase_seconds, status FROM video_ingest_usage WHERE video_id = $1",
        video_id)
    if row is None:
        await record_usage(eval_db, video_id, extra_usage, {}, "reindex")
        return
    merged = _sum_usage(_as_dict(row["usage"]), extra_usage)
    await record_usage(eval_db, video_id, merged, _as_dict(row["phase_seconds"]), row["status"])


def summarize_costs(usage_rows: list, estimate_rows: list) -> dict:
    """Dashboard cost view, from video_ingest_usage + video_ingest_costs rows.

    Per-platform totals come only from MEASURED usage. Ledger rows for videos
    with no usage row predate metering (P146) and are listed separately as
    estimates, never summed into the measured totals. Free tier is reported as
    "not confirmed": no Groq/Gemini/Voyage response reports a charge or
    free-tier consumption -- only each platform's billing system does, and that
    isn't integrated yet (user decision 2026-10-19: billing reconciliation later).
    """
    from pipelines.usage_meter import unit_prices

    plat = {
        "Groq": {"platform": "Groq", "requests": 0, "audio_seconds": 0.0, "billed_seconds": 0.0, "usd": 0.0},
        "Gemini": {"platform": "Gemini", "calls": 0, "input_tokens": 0, "output_tokens": 0,
                   "thought_tokens": 0, "usd": 0.0},
        "Voyage": {"platform": "Voyage", "text_tokens": 0, "mm_text_tokens": 0, "mm_pixels": 0,
                   "mm_images": 0, "usd": 0.0},
    }
    videos = []
    metered_ids = set()
    for row in usage_rows:
        u = _as_dict(row["usage"])
        metered_ids.add(row["video_id"])
        g, gm, v = u["groq"], u["gemini"], u["voyage"]
        for k in ("requests", "audio_seconds", "billed_seconds", "usd"):
            plat["Groq"][k] += g[k]
        for k in ("calls", "input_tokens", "output_tokens", "thought_tokens", "usd"):
            plat["Gemini"][k] += gm[k]
        for k in ("text_tokens", "mm_text_tokens", "mm_pixels", "mm_images", "usd"):
            plat["Voyage"][k] += v[k]
        videos.append({
            "video_id": row["video_id"], "title": row.get("title") or "",
            "status": row["status"], "recorded_at": str(row.get("recorded_at") or ""),
            "phase_seconds": _as_dict(row.get("phase_seconds")),
            "prices_as_of": row.get("prices_as_of") or u.get("prices_as_of"),
            "groq_usd": g["usd"], "gemini_usd": gm["usd"], "voyage_usd": v["usd"],
            "total_usd": u["total_usd"], "usage": u,
        })
    for p in plat.values():
        p["free_tier"] = "not confirmed"
    videos.sort(key=lambda r: r["recorded_at"], reverse=True)

    estimates = []
    for row in estimate_rows:
        if row["video_id"] in metered_ids:
            continue
        usd = float(row["groq_cost_usd"]) + float(row["voyage_cost_usd"]) + float(row["vision_cost_usd"])
        estimates.append({"video_id": row["video_id"], "title": row.get("title") or "",
                          "recorded_at": str(row.get("recorded_at") or ""), "usd": usd,
                          "groq_usd": float(row["groq_cost_usd"]),
                          "voyage_usd": float(row["voyage_cost_usd"]),
                          "gemini_usd": float(row["vision_cost_usd"])})
    estimates.sort(key=lambda r: r["recorded_at"], reverse=True)

    platforms = [plat["Groq"], plat["Gemini"], plat["Voyage"]]
    return {
        "unit_prices": unit_prices(),
        "platforms": platforms,
        "measured_total_usd": sum(p["usd"] for p in platforms),
        "videos": videos,
        "estimated_pre_metering": estimates,
        # D82: billing integration rejected (2026-10-19) -- list price is final.
        "free_tier_note": ("Not tracked: no API response reports free-tier use or a charge, "
                           "and billing data is deliberately not integrated (D82). Figures are "
                           "ingestion-only usage at official list price."),
    }


async def total_spend(eval_db) -> float:
    """Sum of every recorded video's cost. 0.0 if eval_db is unavailable."""
    if eval_db is None:
        return 0.0
    row = await eval_db.fetchrow(
        "SELECT COALESCE(SUM(groq_cost_usd + voyage_cost_usd + vision_cost_usd), 0.0) AS total "
        "FROM video_ingest_costs"
    )
    return float(row["total"]) if row else 0.0


async def check_budget(eval_db) -> bool:
    """True if there's still room under MAX_INGESTION_SPEND_USD.

    Degrades to True (allow processing) when eval_db is unavailable — a
    missed check is recoverable, a queue stuck on a DB hiccup is not.
    """
    if eval_db is None:
        return True
    spent = await total_spend(eval_db)
    ok = spent < config.MAX_INGESTION_SPEND_USD
    if not ok:
        logger.warning(
            "ingestion spend ceiling reached: $%.2f spent, limit $%.2f",
            spent, config.MAX_INGESTION_SPEND_USD,
        )
    return ok


def _self_test():
    import asyncio

    # 1. estimate_groq_cost scales with duration
    assert abs(estimate_groq_cost(3600) - config.GROQ_COST_PER_HOUR_USD) < 1e-9
    assert estimate_groq_cost(1800) == estimate_groq_cost(3600) / 2
    print("ok 1 — estimate_groq_cost scales linearly with duration")

    # 2. estimate_voyage_cost / estimate_vision_cost scale with frame count
    assert estimate_voyage_cost(100) == 100 * config.VOYAGE_COST_PER_FRAME_USD
    assert estimate_vision_cost(50) == 50 * config.VISION_COST_PER_FRAME_USD
    print("ok 2 — estimate_voyage_cost / estimate_vision_cost scale with frame count")

    # 2b. estimate_voyage_text_cost uses the separate, cheaper text rate —
    # not the image-based VOYAGE_COST_PER_FRAME_USD rate.
    assert estimate_voyage_text_cost(200) == 200 * config.VOYAGE_TEXT_COST_PER_ITEM_USD
    assert config.VOYAGE_TEXT_COST_PER_ITEM_USD != config.VOYAGE_COST_PER_FRAME_USD
    print("ok 2b — estimate_voyage_text_cost uses the separate text-item rate")

    # 3. record_cost / total_spend / check_budget with a fake eval_db
    class FakeEvalDB:
        def __init__(self):
            self.rows = {}

        async def execute(self, sql, *args):
            video_id, groq, voyage, vision = args
            self.rows[video_id] = (groq, voyage, vision)
            return True

        async def fetchrow(self, sql, *args):
            total = sum(sum(v) for v in self.rows.values())
            return {"total": total}

    db = FakeEvalDB()

    async def run():
        assert await total_spend(db) == 0.0
        await record_cost(db, "v1", groq_cost=5.0, voyage_cost=1.0)
        assert await total_spend(db) == 6.0
        assert await check_budget(db) is True  # well under the $50 default
        return True

    assert asyncio.run(run())
    print("ok 3 — record_cost / total_spend / check_budget track correctly")

    # 4. check_budget returns False once the ceiling is reached
    saved_limit = config.MAX_INGESTION_SPEND_USD
    config.MAX_INGESTION_SPEND_USD = 5.0
    try:
        assert asyncio.run(check_budget(db)) is False  # 6.0 spent >= 5.0 limit
    finally:
        config.MAX_INGESTION_SPEND_USD = saved_limit
    print("ok 4 — check_budget returns False once the spend ceiling is reached")

    # 5. Every function degrades to a safe default when eval_db is None
    assert asyncio.run(total_spend(None)) == 0.0
    assert asyncio.run(check_budget(None)) is True
    asyncio.run(record_cost(None, "v1", groq_cost=100.0))  # must not raise
    print("ok 5 — degrades safely (allows processing) when eval_db is None")

    # 6. record_usage stores the measured usage and writes MEASURED dollars to
    # the ledger the spend ceiling reads -- for failed runs too, since the
    # paid phases before a failure still spent.
    import json as _json
    from pipelines.usage_meter import UsageMeter
    m = UsageMeter("v6")
    m.record_groq(600.0)
    m.record_gemini(type("U", (), {"prompt_token_count": 1300, "candidates_token_count": 90,
                                    "thoughts_token_count": 60})())
    m.record_voyage_text(5000)
    m.record_voyage_multimodal(0, 331_776, 1)
    usage = m.summary()

    class UsageDB:
        def __init__(self):
            self.calls = []

        async def execute(self, sql, *args):
            self.calls.append((sql, args))
            return True

    udb = UsageDB()
    asyncio.run(record_usage(udb, "v6", usage, {"download": 3.5, "asr": 41.0}, "failed"))
    usage_rows = [a for s, a in udb.calls if "video_ingest_usage" in s]
    cost_rows = [a for s, a in udb.calls if "video_ingest_costs" in s]
    assert len(usage_rows) == 1 and len(cost_rows) == 1, udb.calls
    vid, usage_json, phases_json, as_of, total, status = usage_rows[0]
    assert vid == "v6" and status == "failed" and as_of == usage["prices_as_of"]
    assert _json.loads(usage_json)["gemini"]["thought_tokens"] == 60
    assert _json.loads(phases_json) == {"download": 3.5, "asr": 41.0}
    assert abs(total - usage["total_usd"]) < 1e-12
    assert cost_rows[0] == ("v6", usage["groq"]["usd"], usage["voyage"]["usd"], usage["gemini"]["usd"])
    print("ok 6 — record_usage stores measured usage and puts measured dollars in the ledger")

    # 7. summarize_costs: per-platform totals come ONLY from measured usage;
    # pre-metering ledger rows are listed separately as estimates, never mixed
    # in; free tier is "not confirmed" until billing data exists (2026-10-19:
    # no API response reports free-tier consumption or a charge).
    m2 = UsageMeter("v7")
    m2.record_groq(5.0)            # billed as 10 s
    m2.record_voyage_text(1000)
    usage_rows = [
        {"video_id": "v6", "title": "Lesson", "status": "done", "usage": _json.dumps(usage),
         "phase_seconds": _json.dumps({"asr": 41.0}), "prices_as_of": usage["prices_as_of"],
         "recorded_at": "2026-10-19T15:00:00"},
        {"video_id": "v7", "title": "", "status": "failed", "usage": m2.summary(),
         "phase_seconds": {}, "prices_as_of": "2026-10-19", "recorded_at": "2026-10-19T15:05:00"},
    ]
    estimate_rows = [
        {"video_id": "old1", "title": "Pre-metering", "groq_cost_usd": 0.1,
         "voyage_cost_usd": 0.2, "vision_cost_usd": 0.3, "recorded_at": "2026-10-19T14:00:00"},
        {"video_id": "v6", "title": "Lesson", "groq_cost_usd": 9.0, "voyage_cost_usd": 9.0,
         "vision_cost_usd": 9.0, "recorded_at": "2026-10-19T15:00:00"},   # superseded by usage
    ]
    out = summarize_costs(usage_rows, estimate_rows)
    plat = {p["platform"]: p for p in out["platforms"]}
    assert set(plat) == {"Groq", "Gemini", "Voyage"}, plat
    assert plat["Groq"]["billed_seconds"] == 610.0, plat["Groq"]
    assert abs(plat["Groq"]["usd"] - (usage["groq"]["usd"] + m2.summary()["groq"]["usd"])) < 1e-12
    assert plat["Gemini"]["thought_tokens"] == 60 and plat["Voyage"]["text_tokens"] == 6000
    assert all(p["free_tier"] == "not confirmed" for p in out["platforms"])
    assert abs(out["measured_total_usd"] - sum(p["usd"] for p in out["platforms"])) < 1e-12
    assert [v["video_id"] for v in out["videos"]] == ["v7", "v6"]          # newest first
    assert out["videos"][1]["phase_seconds"] == {"asr": 41.0}
    assert out["videos"][0]["status"] == "failed"
    assert [e["video_id"] for e in out["estimated_pre_metering"]] == ["old1"], out
    assert abs(out["estimated_pre_metering"][0]["usd"] - 0.6) < 1e-12
    units = {(u["platform"], u["unit"]): u["usd"] for u in out["unit_prices"]}
    assert units[("Gemini", "1M output tokens (incl. thinking)")] == 3.75, units
    assert all(u["as_of"] and u["source"].startswith("https://") for u in out["unit_prices"])
    print("ok 7 — summarize_costs: measured per-platform totals, estimates kept apart, free tier unconfirmed")

    # 8. add_usage: a later run on an already-ingested video (the transcript
    # re-index, P152) ADDS its metered usage to that video's row instead of
    # replacing it -- the 2026-10-20 re-index only printed its spend, so the
    # panel's totals left it out.
    m3 = UsageMeter("v6")
    m3.record_gemini(type("U", (), {"prompt_token_count": 100, "candidates_token_count": 10,
                                     "thoughts_token_count": 5})())
    m3.record_voyage_text(2000)
    extra = m3.summary()

    class RowDB(UsageDB):
        async def fetchrow(self, sql, *args):
            self.calls.append((sql, args))
            return {"usage": _json.dumps(usage), "phase_seconds": _json.dumps({"asr": 41.0}),
                    "status": "done"}

    rdb = RowDB()
    asyncio.run(add_usage(rdb, "v6", extra))
    ins = [a for s, a in rdb.calls if "INSERT INTO video_ingest_usage" in s]
    assert len(ins) == 1, rdb.calls
    vid, usage_json, phases_json, as_of, total, status = ins[0]
    merged = _json.loads(usage_json)
    assert vid == "v6" and status == "done" and _json.loads(phases_json) == {"asr": 41.0}
    assert merged["gemini"]["input_tokens"] == 1300 + 100 and merged["gemini"]["calls"] == 2
    assert merged["voyage"]["text_tokens"] == 5000 + 2000
    assert merged["groq"]["audio_seconds"] == 600.0, "untouched platforms keep their figures"
    assert abs(total - (usage["total_usd"] + extra["total_usd"])) < 1e-12
    assert merged["video_id"] == "v6"

    class EmptyDB(UsageDB):
        async def fetchrow(self, sql, *args):
            return None

    edb = EmptyDB()
    asyncio.run(add_usage(edb, "v9", extra))
    ins = [a for s, a in edb.calls if "INSERT INTO video_ingest_usage" in s]
    assert ins and ins[0][0] == "v9" and ins[0][5] == "reindex", ins
    asyncio.run(add_usage(None, "v9", extra))      # no DB: no-op, never raises
    print("ok 8 — add_usage adds a later run's usage to the video's row, never replaces it")

    print("\n8 checks passed")


if __name__ == "__main__":
    _self_test()
