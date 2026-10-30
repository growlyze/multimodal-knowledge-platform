"""Provider health, on demand -- gemini_down.md items D and F.

    # One bounded, near-free call per provider (Gemini/Voyage/Anthropic: a few
    # tokens; Groq: model listing). Output: one JSON object per provider, keys
    # redacted. Exit 0 only if every probed provider answered "ok".
    docker compose exec -e PYTHONPATH=/app -w /app haystack-api python scripts/provider_health.py probe
    docker compose exec -e PYTHONPATH=/app -w /app haystack-api python scripts/provider_health.py probe gemini

    # Count Gemini failure signatures in container logs (stdin). Unlike the old
    # `grep -c "query planner failed"`, this still sees Gemini failing while the
    # Claude fallback covers it -- and never calls silence "healthy".
    docker compose logs --no-log-prefix --since 1h litellm haystack-api 2>&1 \
      | docker compose exec -T -e PYTHONPATH=/app -w /app haystack-api python scripts/provider_health.py logs

Every probe makes a real (tiny) API call: run it by hand or from a schedule of
hours, not minutes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines import provider_health  # noqa: E402


def main(argv) -> int:
    if not argv or argv[0] not in ("probe", "logs"):
        print(__doc__)
        return 2
    if argv[0] == "logs":
        report = provider_health.scan_logs(sys.stdin)
        print(json.dumps(report, indent=2))
        return 1 if report["verdict"].startswith("GEMINI FAILING") else 0
    names = argv[1:] or list(provider_health.PROBES)
    unknown = [n for n in names if n not in provider_health.PROBES]
    if unknown:
        print("unknown provider(s): %s (choose from %s)" % (unknown, list(provider_health.PROBES)))
        return 2
    results = [provider_health.PROBES[n]() for n in names]
    for r in results:
        print(json.dumps(r.to_dict()))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
