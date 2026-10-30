"""Inline [n] citations are asked for in the rules, not only enforced after (2026-10-23).

_drop_uncited_sources keeps only the Sources entries the answer cites by [n] - but an
answer with no [n] at all keeps the whole block (there is no safe way to tell which
entries it used). So the model must be told to place the markers: one right after each
statement taken from a source, and every kept entry cited at least once.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import CITATION_RULES, CITATION_RULES_OPEN, CITE_INLINE_RULE, _context_block


def test_rule_text_asks_for_markers_and_cited_entries():
    assert "[n]" in CITE_INLINE_RULE
    assert "right after each statement" in CITE_INLINE_RULE
    assert "at least once" in CITE_INLINE_RULE


def test_both_citation_variants_carry_it():
    assert CITE_INLINE_RULE in CITATION_RULES
    assert CITE_INLINE_RULE in CITATION_RULES_OPEN


def test_the_context_block_carries_it_strict_and_open():
    hit = {"citation": "[1] [Ep 36](https://youtube.com/watch?v=a)", "content": "x"}
    for strict in (True, False):
        assert CITE_INLINE_RULE in _context_block([hit], "[1] Ep 36", strict=strict)
