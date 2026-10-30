"""
title: Search Datasets
author: Growlyze
version: 1.0
description: Search stored datasets in the knowledge base. Use this to find
available historical data — and the table name to pass to load_dataset() — before
writing analysis code.
"""

import httpx
from pydantic import BaseModel, Field

SOURCE_TYPE = "structured_data"

UNAVAILABLE = (
    "Dataset search is unavailable right now. Ask the user to attach the data file "
    "to the chat instead, then load it with load_file()."
)


def _row(hit: dict) -> str:
    """One dataset, formatted for a model that is about to write pandas."""
    meta = hit.get("metadata") or {}
    table = meta.get("table_name")
    lines = ["• %s (%s rows)" % (hit.get("source", "unknown"),
                                 format(int(meta.get("row_count") or 0), ","))]
    if table:
        lines.append("  Table: %s" % table)
    if meta.get("sheet"):
        lines.append("  Sheet: %s" % meta["sheet"])
    columns = meta.get("columns") or []
    if columns:
        lines.append("  Columns: %s" % ", ".join(str(c) for c in columns))
    if meta.get("symbols"):
        lines.append("  Symbols: %s" % ", ".join(str(s) for s in meta["symbols"]))
    dates = meta.get("date_range") or {}
    if dates.get("start"):
        lines.append("  Date range: %s to %s" % (dates["start"], dates.get("end", "?")))
    # The whole point of the tool: the exact call that loads it.
    lines.append('  Load with: load_dataset("%s")' % (table or hit.get("source", "")))
    return "\n".join(lines)


def _format(query: str, hits: list) -> str:
    if not hits:
        return (
            "No datasets found matching '%s'. Ask the user to upload the data file "
            "to the chat first — it is stored automatically and searchable after "
            "about a minute." % query
        )
    return "Found %d dataset(s):\n\n%s\n" % (len(hits), "\n\n".join(_row(h) for h in hits))


class Tools:
    class Valves(BaseModel):
        haystack_api_url: str = Field(default="http://haystack-api:8000")
        max_results: int = Field(default=5)
        timeout: float = Field(default=5.0)

    def __init__(self):
        self.valves = self.Valves()

    async def search_datasets(self, query: str) -> str:
        """
        Search the stored datasets.

        :param query: What data to look for, e.g. "sales", "revenue 2022", "regional totals".
        :return: Matching datasets with their PostgreSQL table name, columns, row
            count and date range. Pass the table name to load_dataset() in code.
        """
        # The structured pipeline writes table_name / columns / row_count / date_range
        # straight into the Qdrant payload, so /search filtered to structured_data
        # already answers this. Semantic, so "regional totals" finds the right dataset.
        try:
            async with httpx.AsyncClient(timeout=self.valves.timeout) as client:
                response = await client.get(
                    "%s/search" % self.valves.haystack_api_url.rstrip("/"),
                    params={"q": query[:500], "top_k": self.valves.max_results,
                            "source_type": SOURCE_TYPE},
                )
                response.raise_for_status()
                hits = response.json().get("results", [])
        except Exception:
            return UNAVAILABLE
        return _format(query, hits)


def demo():
    """Offline self-check: python dataset_tool.py"""
    hit = {
        "source": "sales_2020_2023.csv",
        "source_type": SOURCE_TYPE,
        "metadata": {
            "table_name": "sales_2020_2023_20250831",
            "sheet": None,
            "row_count": 211043,
            "columns": ["region", "time", "units", "revenue", "cost", "tax", "total"],
            "symbols": ["EMEA"],
            "date_range": {"column": "time", "start": "2020-03-11T00:00:00",
                           "end": "2023-03-11T00:00:00"},
        },
    }
    out = _format("sales", [hit])
    assert "Found 1 dataset(s):" in out
    assert "• sales_2020_2023.csv (211,043 rows)" in out
    assert 'Load with: load_dataset("sales_2020_2023_20250831")' in out
    assert "Date range: 2020-03-11T00:00:00 to 2023-03-11T00:00:00" in out
    assert "Sheet:" not in out                       # None sheet is not printed

    # A half-populated payload must still render — never KeyError at the model.
    bare = _format("x", [{"source": "odd.json", "metadata": {}}])
    assert "odd.json (0 rows)" in bare and 'load_dataset("odd.json")' in bare

    empty = _format("nonexistent", [])
    assert "No datasets found matching 'nonexistent'" in empty

    assert Tools().valves.haystack_api_url == "http://haystack-api:8000"
    print("ok")


if __name__ == "__main__":
    demo()
