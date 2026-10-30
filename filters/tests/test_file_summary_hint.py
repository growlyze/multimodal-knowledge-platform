"""F1-F5 - "What's in this file?" is answered from describe_file(), not from judgement.

2026-10-22, 68 Claude file-summary replies hand-adjudicated: Sonnet asserted "clean /
continuous / no gaps" 7 times when false or unchecked and missed real defects 12 times
(the client file's two long halts: 13/14 replies missed them). The 1.28 hint
asked for a gap check; the model still decided what to run. data_tools now has
describe_file(path) -> dict and format_file_summary(info) -> str, which compute every F3
check; the hint makes a summary turn print that and report all of it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter

PATH = "/mnt/uploads/abc_DATA_SET_01, 5.csv"


def _hint():
    return rag_filter._file_ready_hint([{"name": "data.csv", "path": PATH}], "What's in this file?")


def test_summary_turn_prints_format_file_summary_of_describe_file_on_the_real_path():
    h = _hint()
    assert 'print(format_file_summary(describe_file("%s")))' % PATH in h
    assert "from data_tools import describe_file, format_file_summary" in h
    low = h.lower()
    assert "what's in this file" in low and "summar" in low


def test_every_printed_fact_and_problem_is_reported():
    low = _hint().lower()
    assert "every fact" in low and "every problem" in low
    assert "do not drop" in low or "never drop" in low


def test_no_unverified_absolutes():
    low = _hint().lower()
    assert "clean" in low and "continuous" in low and "no gaps" in low
    assert "printed zero" in low or "printed 0" in low


def test_columns_and_time_zone_definitions_f5():
    low = _hint().lower()
    assert "raw_columns" in low
    assert "time zone" in low


def test_mis_parse_is_reported_not_described_f4():
    low = _hint().lower()
    assert "could not be read correctly" in low


def test_load_file_guidance_and_128_gap_text_kept_as_fallback():
    h = _hint()
    assert 'load_file("%s")' % PATH in h
    assert "Never call the series continuous or gap-free unless that printed count is 0" in h
    assert "describe_file is not available" in h
    # existing pinned phrases
    for s in (".tail()", "nrows=0", "load_issues"):
        assert s in h, s


def test_hint_formats_with_a_percent_free_path():
    # the hint is %-formatted; a literal % inside it would raise at runtime
    assert _hint().count(PATH) >= 3


def test_hint_asks_for_the_first_and_last_timestamp():
    # 2026-10-23: summaries stopped giving the file's first/last row facts (1.30.0).
    assert "first and last timestamp" in rag_filter.FILE_READY_HINT
