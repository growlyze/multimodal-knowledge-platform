"""Stored-data routing table: summary or the whole data? (2026-10-22, P185).

A stored file's rows live only in PostgreSQL; the search index holds ONE catalogue
summary chunk per table (date range, a few column stats - "an index entry, not the
data itself"). 1.29 sent the model to the sandbox (load_dataset, the whole table)
only when the question ALSO held a compute keyword: 13 of 15 everyday questions
("how many records are in the sales log?", "list the dates where volume was zero")
got no sandbox and were answered from that one summary chunk.

The rule this table pins:
  data     - asks about what is IN stored data (counts, lists, values, extremes,
             checks, summaries of a specific dataset, per-period figures, follow-ups
             to a data turn) -> code flag True, the whole table is loaded and computed.
  catalog  - asks only whether/which data exists ("do we have sales data from 2020?",
             "which datasets are stored?") -> the catalogue summary is the right
             answer; no sandbox.
  concept  - a knowledge question that merely uses data words ("what is a row?",
             "explain how a weighted average is calculated") -> no sandbox.
Pass bar: 100%. Attached files are not in this table: they reach the sandbox through
data_files on every turn (verified live, `files: 1` on follow-ups).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import _turn_flags

DATA = [
    # counts
    "How many records are in the sales log?",
    "How many rows are in the 2023 dataset?",
    "In the stored sales data, how many rows are there?",
    "How many rows does sales_test.csv have?",
    "Count the flagged rows in my records file",
    "how many days are in the sales data?",
    # lists / finds
    "List all the dates in the sales file where volume was zero",
    "Find the days with the biggest range in the sales data",
    "Show me every record that lost more than $500 in my records log",
    "Which days had a gap up of more than 1% in the 2022 dataset?",
    "Give me all the timestamps where value is below zero in the stored sales data",
    # values / extremes
    "What was the highest value in my sales log?",
    "What's the lowest value in the 2022 data?",
    "What was the biggest losing record in my records file?",
    "What's the average daily range of the sales data?",
    "What was the total in the sales log?",
    # checks
    "Do we have any gaps in the 2022 dataset?",
    "Are there duplicate timestamps in the sales dataset?",
    "Is the volume usable in the stored sales data?",
    "Is anything missing in my sales data?",
    "Check the uploaded sales file for bad values",
    # summaries / overviews of a specific dataset
    "What's in the SALES_2020_2023 file?",
    "Summarize my sales 2023 data",
    "Tell me about the sales data we uploaded last week",
    "Give me an overview of sales_test.csv",
    "What does the REPORTS_MINI_2026 file contain?",
    "Describe the sales log",
    "What's in my records file?",
    # per-period / comparisons
    "Which month had the highest total in my records file?",
    "Compare the 2022 sales data and the 2023 sales data: which year had the larger average range?",
    "What was the average per weekday in my records log?",
    "Break down the sales log by hour of entry",
    # compute words (already routed by 1.29 - must stay routed)
    "Compute the totals for the sales 2023 data",
    "Calculate the average daily range of the sales data",
    # Romanian (the client's team writes some turns in Romanian)
    "Câte tranzacții sunt în jurnalul de vânzări?",
]

CATALOG = [
    "Do we have sales data from 2020?",
    "Which datasets are stored?",
    "What data files do we have?",
    "Is there any sales data uploaded?",
    "Do you have historical data for revenue?",
    "List the datasets we have",
    "What files have I uploaded?",
]

CONCEPT = [
    "What is a moving average?",
    "Explain how a weighted average is calculated",
    "What is a row in a dataset?",
    "What does CSV mean?",
    "Why do servers have a maintenance window?",
    "What's the difference between a price gap and a time gap?",
    # Not here: "How should I read volume on a 5 minute chart?" - the older
    # CODE_KEYWORDS gate already sends it to the sandbox (pre-existing, not this class).
    "Tell me about the history of aviation",
    "hello",
    "thanks, that's great",
]


def code_flag(query, history=None):
    return _turn_flags(query, history, False)


@pytest.mark.parametrize("q", DATA)
def test_data_questions_reach_the_sandbox(q):
    assert code_flag(q), "answered from the catalogue summary, not the data: %r" % q


@pytest.mark.parametrize("q", CATALOG + CONCEPT)
def test_catalog_and_concept_questions_do_not(q):
    assert not code_flag(q), "sent to the sandbox for no reason: %r" % q


FOLLOW_UPS = [
    "and how many of them were flagged?",
    "what about the lowest one?",
    "now show it per month",
    "are there any gaps?",
]


@pytest.mark.parametrize("q", FOLLOW_UPS)
def test_follow_up_to_a_data_turn_stays_on_the_data(q):
    history = [
        {"role": "user", "content": "How many records are in the sales log?"},
        {"role": "assistant", "content": "The sales log has 29,707 records."},
    ]
    assert code_flag(q, history), q


def test_follow_up_without_a_data_turn_is_not_forced():
    history = [
        {"role": "user", "content": "What is a moving average?"},
        {"role": "assistant", "content": "A moving average smooths a series."},
    ]
    assert not code_flag("what about the second one?", history)


def test_stored_data_hint_tells_the_model_the_summary_is_not_the_data():
    hint = rag_filter.STORED_DATA_HINT
    assert "load_dataset" in hint
    assert "summary" in hint.lower() and "not the data" in hint.lower()
    assert "describe_file" in hint


async def _no_hits(*_a, **_kw):
    return [], None, ""


def _system_after_route(messages):
    import asyncio
    f = rag_filter.Filter()
    f._search = _no_hits
    body = {"messages": messages}
    asyncio.run(f._route(body, messages[-1]["content"], []))
    return "\n".join(m["content"] for m in body["messages"] if m.get("role") == "system")


@pytest.mark.parametrize("q", ["What's in my records file?",
                               "How many records are in the sales log?",
                               "List all the dates in the sales file where volume was zero"])
def test_data_question_gets_the_load_dataset_hint(q):
    system = _system_after_route([{"role": "user", "content": q}])
    assert rag_filter.STORED_DATA_HINT in system, q


def test_catalog_question_gets_no_sandbox_hint():
    system = _system_after_route([{"role": "user", "content": "Do we have sales data from 2020?"}])
    assert rag_filter.STORED_DATA_HINT not in system


def test_attached_file_hint_says_compute_over_every_row_not_eyeball():
    # User's chat 7604a099 (2026-10-21, 1.27): the model read all 533 rows, printed
    # head/tail/describe as the hint suggested, and called the file gap-free.
    hint = rag_filter.FILE_READY_HINT
    assert "inspect the resulting DataFrame" not in hint
    assert "never from .head(), .tail(), .sample()" in hint
    assert "say how many you left out" in hint
