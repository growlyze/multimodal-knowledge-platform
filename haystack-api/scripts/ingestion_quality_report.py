"""
Pull M3 ingestion quality, performance, and failure data out of Phoenix, and
cost data out of Postgres, for debugging, QA, and quality tracking.

Why this exists: every ingestion phase already writes its quality verdict,
score, timing, and output into Phoenix (pipelines/video_ingestion.py's
span()/record_decision()/set_output() calls) -- but nothing ever reads it
back out. Phoenix answers "what happened in this one case" when someone opens
a trace by hand; this script is the read side that turns the same data into
something usable for QA/testing, not just forensic debugging.
project-knowledge-base/19-observability.md has the fuller "why".

Cost does NOT live in Phoenix -- routers/video.py writes it straight to
Postgres' video_ingest_costs (D68) -- so it's pulled from there and merged
into the same report rather than reinvented as a second cost tracker.

Usage:
    python scripts/ingestion_quality_report.py --video-id abc123
    python scripts/ingestion_quality_report.py --video-id abc123 --verbose
    python scripts/ingestion_quality_report.py --since 7d
    python scripts/ingestion_quality_report.py --since 30d --format json

Reads PHOENIX_ENDPOINT/PHOENIX_COLLECTOR_ENDPOINT + PHOENIX_API_KEY (already
set for OTLP export, see processing/telemetry/tracing.py) and DATABASE_URL
(eval_db) from the environment -- no new credentials to configure.

LIVE-VERIFIED 2026-10-25 against the 4 pilot videos (Phoenix column shapes match;
the Phoenix base URL default below was the one fix it needed). Original note:
no M3 ingestion had run since the Phoenix
session/project wiring (ingestion_session(), same session as this script)
was added, so the exact dataframe column names this script expects have not
been checked against a real trace. _self_test() below verifies the
aggregation/rendering logic offline against hand-built rows; the first real
run against the VPS is what confirms Phoenix's actual column shapes.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/app"))
sys.path.insert(0, str(ROOT))

from evaluation.eval_db import get_eval_db
from telemetry_compat import INGESTION_PROJECT

# phase span name -> (decision attribute, reason attribute). None/None means
# that phase doesn't record a quality verdict (download/index/cleanup don't;
# only asr/visual/embed have a quality check, see pipelines/ingestion_quality.py).
PHASE_QUALITY_ATTRS = {
    "ingest_video": ("ingest.decision", "ingest.reason"),
    "ingest.download": (None, None),
    "ingest.asr": ("asr.quality.decision", "asr.quality.reason"),
    "ingest.visual": ("visual.quality.decision", "visual.quality.reason"),
    "ingest.embed": ("embed.quality.decision", "embed.quality.reason"),
    "ingest.index": (None, None),
    "ingest.cleanup": (None, None),
}

SELECT_ATTRS = [
    "name", "start_time", "end_time", "status_code",
    "output.value", "session.id", "ingest.platform", "metadata",
    "ingest.decision", "ingest.reason",
    "asr.quality.decision", "asr.quality.reason", "asr.avg_logprob",
    "visual.quality.decision", "visual.quality.reason", "visual.keyframes_per_hour",
    "embed.quality.decision", "embed.quality.reason",
]


def _phoenix_client():
    """Import lazily -- a script that only needs --help shouldn't require
    arize-phoenix-client to be importable, and this keeps the degrade-never-
    break convention consistent with the rest of the codebase."""
    import os
    from phoenix.client import Client
    return Client(base_url=_phoenix_base_url(os.environ))


def _phoenix_base_url(env) -> str:
    """Phoenix's REST URL. Without one, phoenix.client defaults to localhost:6006,
    which inside the haystack-api container is nothing: the first live run (2026-10-25)
    failed with "Connection refused" until the URL was passed by hand.
    PHOENIX_COLLECTOR_ENDPOINT is honoured only when it is not the 4317 gRPC
    collector that processing/telemetry/tracing.py exports spans to -- the
    REST API is not served there."""
    if env.get("PHOENIX_BASE_URL"):
        return env["PHOENIX_BASE_URL"]
    collector = env.get("PHOENIX_COLLECTOR_ENDPOINT") or ""
    if collector and not collector.rstrip("/").endswith(":4317"):
        return collector
    return "http://phoenix:6006"


def _span_query(where: str):
    from phoenix.client.types.spans import SpanQuery
    return SpanQuery().where(where).select(*SELECT_ATTRS)


def fetch_video_rows(client, video_id: str) -> list[dict]:
    """Every span for one video's ingestion session (first attempt + any
    resume_interrupted retries, since they share session.id -- see
    ingestion_session() in processing/telemetry/tracing.py)."""
    df = client.spans.get_spans_dataframe(
        query=_span_query(f"session.id == '{video_id}'"),
        project_identifier=INGESTION_PROJECT,
    )
    return _rows_from_dataframe(df)


def fetch_rows_since(client, since: timedelta) -> list[dict]:
    """Every ingestion span in the last `since` -- the whole project, not
    just root spans, so per-phase decision counts can be rolled up too."""
    df = client.spans.get_spans_dataframe(
        query=_span_query("name is not None"),
        project_identifier=INGESTION_PROJECT,
        start_time=datetime.now(timezone.utc) - since,
    )
    return _rows_from_dataframe(df)


def _rows_from_dataframe(df) -> list[dict]:
    """pandas DataFrame -> plain dicts. Isolated so the report-building
    functions below can be tested without pandas or a real Phoenix response
    (see _self_test())."""
    return df.reset_index().to_dict("records")


async def fetch_costs(video_ids: list[str]) -> dict[str, dict]:
    """video_id -> {groq_cost_usd, voyage_cost_usd, vision_cost_usd, total}
    from Postgres (D68's video_ingest_costs), not Phoenix."""
    if not video_ids:
        return {}
    db = get_eval_db()
    placeholders = ", ".join(f"${i + 1}" for i in range(len(video_ids)))
    rows = await db.fetch(
        f"SELECT video_id, groq_cost_usd, voyage_cost_usd, vision_cost_usd "
        f"FROM video_ingest_costs WHERE video_id IN ({placeholders})",
        *video_ids,
    )
    return {
        r["video_id"]: {
            **r,
            "total_usd": r["groq_cost_usd"] + r["voyage_cost_usd"] + r["vision_cost_usd"],
        }
        for r in rows
    }


# ── report building (pure functions, no I/O -- see _self_test) ─────────────

def build_video_report(rows: list[dict], video_id: str, cost: dict = None,
                       verbose: bool = False) -> dict:
    """One video's phase-by-phase breakdown."""
    phases = []
    def _sort_key(r):
        start = r.get("start_time")
        return (start is None, start if start is not None else 0)

    for row in sorted(rows, key=_sort_key):
        name = row.get("name")
        decision_attr, reason_attr = PHASE_QUALITY_ATTRS.get(name, (None, None))
        phase = {
            "phase": name,
            "status": row.get("status_code"),
            "duration_s": _duration_s(row),
            "decision": row.get(decision_attr) if decision_attr else None,
            "reason": row.get(reason_attr) if reason_attr else None,
        }
        if verbose:
            phase["output"] = row.get("output.value")
        phases.append(phase)
    return {
        "video_id": video_id,
        "phases": phases,
        "cost": cost or None,
    }


def build_aggregate_report(rows: list[dict], since_label: str) -> dict:
    """Per-video summary + per-phase decision rollup across a time window."""
    by_video: dict = defaultdict(list)
    for row in rows:
        video_id = row.get("session.id")
        if video_id:
            by_video[video_id].append(row)

    videos = []
    decision_counts: dict = defaultdict(lambda: defaultdict(int))
    for video_id, video_rows in by_video.items():
        root = next((r for r in video_rows if r.get("name") == "ingest_video"), None)
        worst = _worst_decision(video_rows)
        videos.append({
            "video_id": video_id,
            "status": root.get("status_code") if root else None,
            "worst_decision": worst,
            "total_duration_s": sum(_duration_s(r) for r in video_rows),
        })
        for row in video_rows:
            decision_attr, _ = PHASE_QUALITY_ATTRS.get(row.get("name"), (None, None))
            if decision_attr and row.get(decision_attr):
                decision_counts[row["name"]][row[decision_attr]] += 1

    return {
        "since": since_label,
        "video_count": len(videos),
        "videos": videos,
        "decision_counts": {k: dict(v) for k, v in decision_counts.items()},
    }


_DECISION_SEVERITY = {"failed": 3, "aborted": 3, "degraded": 2, "declined": 1, "done": 0, "answered": 0}


def _worst_decision(rows: list[dict]) -> str | None:
    worst, worst_score = None, -1
    for row in rows:
        decision_attr, _ = PHASE_QUALITY_ATTRS.get(row.get("name"), (None, None))
        decision = row.get(decision_attr) if decision_attr else None
        if decision is None:
            continue
        score = _DECISION_SEVERITY.get(decision, 1)
        if score > worst_score:
            worst, worst_score = decision, score
    return worst


def _duration_s(row: dict) -> float:
    start, end = row.get("start_time"), row.get("end_time")
    if not start or not end:
        return 0.0
    if isinstance(start, str):
        return 0.0  # ISO parsing deliberately left to the caller's dataframe layer
    return (end - start).total_seconds()


# ── rendering ────────────────────────────────────────────────────────────

def render_text_video(report: dict) -> str:
    lines = [f"Video {report['video_id']}"]
    for phase in report["phases"]:
        verdict = f"{phase['decision']} - {phase['reason']}" if phase["decision"] else "-"
        lines.append(f"  {phase['phase']:<16} {phase['status'] or '?':<8} "
                     f"{phase['duration_s']:>6.1f}s  {verdict}")
        if "output" in phase:
            lines.append(f"      output: {phase['output']}")
    if report["cost"]:
        c = report["cost"]
        lines.append(f"  cost: groq=${c['groq_cost_usd']:.4f} voyage=${c['voyage_cost_usd']:.4f} "
                     f"vision=${c['vision_cost_usd']:.4f} total=${c['total_usd']:.4f}")
    return "\n".join(lines)


def render_text_aggregate(report: dict) -> str:
    lines = [f"Ingestion report, last {report['since']} ({report['video_count']} videos)", ""]
    for v in report["videos"]:
        lines.append(f"  {v['video_id']:<24} {v['status'] or '?':<8} "
                     f"worst={v['worst_decision'] or 'ok':<10} {v['total_duration_s']:>7.1f}s")
    lines.append("")
    lines.append("Per-phase decision counts:")
    for phase, counts in report["decision_counts"].items():
        lines.append(f"  {phase}: {dict(counts)}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_since(value: str) -> timedelta:
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "d":
        return timedelta(days=amount)
    if unit == "h":
        return timedelta(hours=amount)
    raise ValueError(f"--since must end in 'd' or 'h', got {value!r}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", help="Full phase-by-phase report for one video")
    parser.add_argument("--since", help="Aggregate report, e.g. 7d or 48h")
    parser.add_argument("--verbose", action="store_true",
                        help="Include each phase's full raw output text")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    if not args.video_id and not args.since:
        parser.error("pass --video-id or --since")

    client = _phoenix_client()

    if args.video_id:
        rows = fetch_video_rows(client, args.video_id)
        costs = await fetch_costs([args.video_id])
        report = build_video_report(rows, args.video_id, cost=costs.get(args.video_id),
                                    verbose=args.verbose)
        print(json.dumps(report, default=str, indent=2) if args.format == "json"
              else render_text_video(report))
    else:
        since = _parse_since(args.since)
        rows = fetch_rows_since(client, since)
        report = build_aggregate_report(rows, args.since)
        print(json.dumps(report, default=str, indent=2) if args.format == "json"
              else render_text_aggregate(report))


# ── offline self-test -- no network, no Phoenix, no database ──────────────

def _self_test():
    # 1. build_video_report orders phases by start_time and pulls the right
    # decision/reason attribute per phase name. Timestamps are real datetimes
    # -- what a pandas dataframe from Phoenix actually carries, not strings
    # or ints (a prior version of this test used ints and masked a falsy-zero
    # sort bug: start_time=0 is falsy, so `start_time or ""` silently turned
    # the earliest phase's sort key into a string, crashing the comparison
    # against every other row's int/datetime key).
    t0 = datetime(2026, 10, 12, 0, 0, 0)
    rows = [
        {"name": "ingest.asr", "status_code": "OK",
         "start_time": t0 + timedelta(seconds=10), "end_time": t0 + timedelta(seconds=15),
         "asr.quality.decision": "degraded", "asr.quality.reason": "avg_logprob -0.52"},
        {"name": "ingest_video", "status_code": "OK",
         "start_time": t0, "end_time": t0 + timedelta(seconds=100),
         "ingest.decision": "done", "ingest.reason": "6/6 phases completed"},
        {"name": "ingest.download", "status_code": "OK",
         "start_time": t0, "end_time": t0 + timedelta(seconds=8)},
    ]
    report = build_video_report(rows, "vid1")
    assert [p["phase"] for p in report["phases"]] == [
        "ingest_video", "ingest.download", "ingest.asr",
    ] or [p["phase"] for p in report["phases"]] == [
        "ingest.download", "ingest_video", "ingest.asr",
    ], report  # ingest_video and ingest.download share start_time=t0; either tie order is fine
    asr_phase = next(p for p in report["phases"] if p["phase"] == "ingest.asr")
    assert asr_phase["decision"] == "degraded" and "avg_logprob" in asr_phase["reason"], asr_phase
    assert asr_phase["duration_s"] == 5.0, asr_phase
    print("ok 1 — build_video_report orders phases and extracts the right decision/reason")

    # 2. verbose=True includes raw output text; default does not.
    rows_with_output = [{"name": "ingest.asr", "status_code": "OK",
                         "start_time": t0, "end_time": t0 + timedelta(seconds=1),
                         "output.value": "23 segments, avg_logprob=-0.31"}]
    quiet = build_video_report(rows_with_output, "vid2")
    loud = build_video_report(rows_with_output, "vid2", verbose=True)
    assert "output" not in quiet["phases"][0]
    assert loud["phases"][0]["output"] == "23 segments, avg_logprob=-0.31"
    print("ok 2 — --verbose includes raw phase output, the default summary omits it")

    # 3. build_aggregate_report groups by session.id (video_id) and finds the
    # worst decision per video -- "failed" outranks "degraded" outranks "done".
    agg_rows = [
        {"name": "ingest_video", "status_code": "OK", "session.id": "v1",
         "start_time": t0, "end_time": t0 + timedelta(seconds=10), "ingest.decision": "done"},
        {"name": "ingest.asr", "session.id": "v1",
         "start_time": t0, "end_time": t0 + timedelta(seconds=5),
         "asr.quality.decision": "degraded"},
        {"name": "ingest_video", "status_code": "ERROR", "session.id": "v2",
         "start_time": t0, "end_time": t0 + timedelta(seconds=3), "ingest.decision": "failed"},
    ]
    agg = build_aggregate_report(agg_rows, "7d")
    by_id = {v["video_id"]: v for v in agg["videos"]}
    assert by_id["v1"]["worst_decision"] == "degraded", by_id["v1"]
    assert by_id["v2"]["worst_decision"] == "failed", by_id["v2"]
    assert agg["decision_counts"]["ingest.asr"] == {"degraded": 1}, agg["decision_counts"]
    print("ok 3 — build_aggregate_report groups by video and surfaces the worst decision")

    # 4. render_text_video and render_text_aggregate don't crash and mention
    # the key facts -- a smoke check, not a snapshot test (format may evolve).
    text = render_text_video(build_video_report(rows, "vid1"))
    assert "vid1" in text and "degraded" in text, text
    text2 = render_text_aggregate(agg)
    assert "v1" in text2 and "v2" in text2 and "failed" in by_id["v2"]["worst_decision"], text2
    print("ok 4 — text rendering includes video id and worst decision")

    # 5. _parse_since accepts 'Nd'/'Nh' and rejects anything else.
    assert _parse_since("7d") == timedelta(days=7)
    assert _parse_since("48h") == timedelta(hours=48)
    try:
        _parse_since("7w")
        raise AssertionError("must reject an unsupported unit")
    except ValueError:
        pass
    print("ok 5 — _parse_since accepts d/h suffixes, rejects anything else")

    # 6. The Phoenix REST base URL. Inside the haystack-api container no PHOENIX_* URL is
    # set, so phoenix.client fell back to localhost:6006 and the first live
    # run (2026-10-25) died on "Connection refused". The default must be the
    # compose service (phoenix:6006 -- the HTTP port, not the 4317 gRPC
    # collector tracing.py exports to), and an explicit setting still wins.
    assert _phoenix_base_url({}) == "http://phoenix:6006"
    assert _phoenix_base_url({"PHOENIX_BASE_URL": "http://x:1"}) == "http://x:1"
    assert _phoenix_base_url({"PHOENIX_COLLECTOR_ENDPOINT": "http://phoenix:4317"}) == "http://phoenix:6006"
    assert _phoenix_base_url({"PHOENIX_COLLECTOR_ENDPOINT": "http://other:6006"}) == "http://other:6006"
    print("ok 6 — Phoenix base URL defaults to the compose service, never localhost or the gRPC port")

    print("\n6 checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        asyncio.run(_main())
