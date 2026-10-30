"""Cite-on-use, enforced in the outlet (P191, 2026-10-23).

SOURCES_RULE has asked since P11/P12 for "only the entries you actually drew on".
Prose alone did not hold it: on the live 1.30 run Sonnet's answer said
"the term isn't formally defined ... anywhere in the knowledge base" and then
listed 5 sources (3 videos, 2 doc pages) cited nowhere in the text - the same
contradiction the client flagged on 2026-10-22 ("no mechanical definition, yet
cites Ep 36 / Ep 12"). An entry now survives only if the answer refers to it, by
its [n]
or by its title; no surviving entry means no Sources section.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import _drop_uncited_sources, _renumber_citations

E36 = "[Training Video Episode 36 (1:39–2:38)](https://example.com/watch?v=pWnMpdN_g98&t=99)"
E12 = "[Training Video Episode 12 (46:13–47:10)](https://example.com/watch?v=8GkQfdAXZP0&t=2773)"
BOX = "[Lines and boxes — User Manual](https://example.com/docs/visuals/lines-and-boxes/)"


def _guard(text):
    return _renumber_citations(_drop_uncited_sources(text))


def test_an_answer_without_any_marker_keeps_its_block():
    # Replay of 105 real replies: unmarked answers often DO use their sources
    # ("the docs", "Episodes 36 and 12"); without [n] there is no safe way to tell.
    text = ("The term isn't formally defined in the knowledge base. I'm using the "
            "standard three-row construction.\n\n```python\n"
            "x = compute(rows[1])\n```\n\nSources:\n[1] %s\n[2] %s\n[3] %s" % (E36, E12, BOX))
    out = _guard(text)
    assert out == text


def test_only_the_cited_entry_survives_and_is_renumbered():
    text = "Price returns to the level [2].\n\nSources:\n[1] %s\n[2] %s\n[3] %s" % (E36, E12, BOX)
    out = _guard(text)
    assert out == "Price returns to the level [1].\n\nSources:\n[1] %s" % E12


def test_grouped_markers_count_as_citations():
    text = "The 5-minute chart shows delivery into the zone [1, 3].\n\nSources:\n[1] %s\n[2] %s\n[3] %s" % (E36, E12, BOX)
    out = _guard(text)
    assert "[1] [2]" in out, out
    assert E36 in out and BOX in out and E12 not in out


def test_a_source_named_in_the_text_is_kept_alongside_marked_ones():
    text = ("The level is respected [3]; Episode 12 shows the same.\n\nSources:\n"
            "[1] %s\n[2] %s\n[3] %s" % (E36, E12, BOX))
    out = _guard(text)
    assert E12 in out and BOX in out and E36 not in out


def test_index_brackets_in_code_do_not_count_as_citations():
    text = ("Use `rows[1]` for the previous row, as in Episode 36 [1].\n\n```\nx = values[2]\n```"
            "\n\nSources:\n[1] %s\n[2] %s" % (E36, E12))
    out = _guard(text)
    assert E36 in out and E12 not in out and "x = values[2]" in out


def test_text_without_sources_is_unchanged():
    text = "No sources here [1] in prose."
    assert _drop_uncited_sources(text) == text


def test_adjacent_markers_both_count():
    # replay 2026-10-23: "[1][2]" in a real reply lost [2] - a [n] right after "]"
    # was invisible to the citation pattern.
    text = "Use the offset flag with an index [1][2].\n\nSources:\n[1] %s\n[2] %s\n[3] %s" % (E36, E12, BOX)
    out = _guard(text)
    assert E36 in out and E12 in out and BOX not in out


def test_sources_render_one_per_line():
    # Live screenshot 2026-10-23: single newlines collapse in the chat, so
    # "[1] ... [2] ... [3] ..." ran together as one paragraph.
    from rag_filter import _format_sources_list
    text = "Level respected [1][2].\n\nSources:\n[1] %s\n[2] %s" % (E36, E12)
    out = _format_sources_list(_guard(text))
    assert out.endswith("Sources:\n\n- [1] %s\n- [2] %s" % (E36, E12)), out
    assert _format_sources_list(out) == out, "idempotent"
