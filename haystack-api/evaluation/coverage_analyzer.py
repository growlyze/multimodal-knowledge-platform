"""
Knowledge gaps — the queries the knowledge base keeps failing to answer.

Every empty search collapses to a short pattern ("what is the judas swing?" ->
"judas swing"), and identical patterns accumulate a count. Three occurrences of
the same pattern is content nobody has ingested yet.

    analyzer = CoverageAnalyzer()
    await analyzer.record_empty_query("Can you explain the Judas Swing?")
    await analyzer.get_gaps(min_count=3)
    await analyzer.mark_resolved("Judas Swing")

ponytail: stopword strip + the first two content words, not embeddings. It
groups the phrasings people actually type ("what is X", "explain X", "X entry
model") onto one row, costs nothing, and needs no model to be up. Upgrade path
is semantic clustering over embedding_service — only `extract_pattern` changes.
"""

import logging
import os
import re

from evaluation.eval_db import EvalDB, cutoff, get_eval_db

logger = logging.getLogger("haystack-api.evaluation")

MIN_COUNT = int(os.getenv("EVAL_COVERAGE_MIN_COUNT", "3"))
PATTERN_WORDS = 2           # "service level", "load balancer" — two words is the topic
MAX_PATTERN_CHARS = 120

STOPWORDS = frozenset("""
a an the this that these those there here it its is are was were be been being am
do does did done doing can could should would will shall may might must
what whats which who whom whose when where why how
i me my we our you your he she they them their
and or but if then than so as of in on at to for from with without about into
over under between during before after again
explain define describe tell show give find get list mean means meaning
please help me some any all more most other another such no not only own same
work works working use used using does new best good
""".split())

_WORD = re.compile(r"[a-z0-9']+")


def extract_pattern(query: str) -> str:
    """A query -> the phrase it is really asking about, lowercased."""
    words = _WORD.findall((query or "").lower())
    content = [w for w in words if w not in STOPWORDS and len(w) > 1] or words
    return " ".join(content[:PATTERN_WORDS])[:MAX_PATTERN_CHARS]


class CoverageAnalyzer:
    def __init__(self, db: EvalDB = None):
        self.db = db or get_eval_db()

    async def record_empty_query(self, query: str) -> str:
        """Count one empty result against its pattern. Returns the pattern."""
        pattern = extract_pattern(query)
        if not pattern:
            return ""
        # ponytail: `resolved` is deliberately not reset here. Marking a gap
        # resolved is an admin decision; the count keeps climbing underneath it,
        # and get_gaps(include_resolved=True) is how you see that.
        await self.db.execute(
            """
            INSERT INTO eval_coverage_gaps (query_pattern, occurrence_count)
            VALUES ($1, 1)
            ON CONFLICT (query_pattern) DO UPDATE
            SET occurrence_count = eval_coverage_gaps.occurrence_count + 1,
                last_seen = NOW()
            """,
            pattern,
        )
        return pattern

    async def get_gaps(self, min_count: int = None, limit: int = 50,
                       include_resolved: bool = False) -> list:
        """Patterns that repeatedly return empty, worst first."""
        rows = await self.db.fetch(
            """
            SELECT query_pattern, occurrence_count, first_seen, last_seen, resolved
            FROM eval_coverage_gaps
            WHERE occurrence_count >= $1 %s
            ORDER BY occurrence_count DESC, last_seen DESC
            LIMIT $2
            """ % ("" if include_resolved else "AND resolved = FALSE"),
            int(MIN_COUNT if min_count is None else min_count), max(1, int(limit)),
        )
        return [{
            "pattern": row["query_pattern"].title(),
            "count": row["occurrence_count"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "resolved": bool(row["resolved"]),
        } for row in rows]

    async def mark_resolved(self, pattern: str, resolved: bool = True) -> bool:
        """Hide a gap once the content behind it has been ingested."""
        # get_gaps title-cases for display, so lowering is enough to find the row again.
        return await self.db.execute(
            "UPDATE eval_coverage_gaps SET resolved = $1 WHERE query_pattern = $2",
            bool(resolved), (pattern or "").strip().lower(),
        )

    async def backfill(self, hours: float = 24) -> int:
        """Rebuild gaps from eval_query_log — for empty queries logged before
        the analyzer was wired in, or after a manual reset."""
        rows = await self.db.fetch(
            "SELECT query FROM eval_query_log WHERE empty_result AND timestamp >= $1",
            cutoff(hours),
        )
        for row in rows:
            await self.record_empty_query(row["query"])
        logger.info("coverage backfill: %d empty queries over %sh", len(rows), hours)
        return len(rows)


if __name__ == "__main__":
    assert extract_pattern("What is the Judas Swing?") == "judas swing"
    assert extract_pattern("Can you explain judas swing entry models") == "judas swing"
    assert extract_pattern("JUDAS SWING") == "judas swing"
    assert extract_pattern("wyckoff") == "wyckoff"
    assert extract_pattern("what is it") == "what is"        # nothing but stopwords
    assert extract_pattern("") == ""
    print("coverage analyzer self-check ok")
