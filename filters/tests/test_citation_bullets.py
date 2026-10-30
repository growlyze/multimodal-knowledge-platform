"""Sources lists the model writes as a markdown list (P214, 2026-10-25).

_SOURCE_LINE_RE only matched lines starting "[n] ", so a list the model bulleted
itself - copying the "- [1] ..." style its previous reply was DISPLAYED with - was
skipped by drop, renumber and format alike: the screen showed "Sources: - [2] ..."
alone (chat 9558b71b turn 2) or [1] [3] [6]. A list of unnumbered "- [Title](url)"
entries under prose citing [1] [2] (a66d02f1 turn 3) linked no marker to anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_filter import (_drop_uncited_sources, _format_sources_list,
                        _normalize_sources_block, _renumber_citations)


def chain(t):
    return _format_sources_list(_renumber_citations(_drop_uncited_sources(_normalize_sources_block(t))))


def test_bulleted_numbered_list_is_renumbered_from_one():
    t = ("A gap [3] and a block [6].\n\nSources:\n\n"
         "- [3] [Ep 36 (3:08-4:07)](https://y/a)\n- [6] [Ep 12](https://y/b)")
    out = chain(t)
    assert "A gap [1] and a block [2]." in out
    assert out.endswith("Sources:\n\n- [1] [Ep 36 (3:08-4:07)](https://y/a)\n- [2] [Ep 12](https://y/b)")


def test_single_bulleted_entry_starts_at_one():
    t = "About Episode 36.\n\nSources:\n\n- [2] [Ep 36 (3:08-4:07)](https://y/a)"
    assert chain(t).endswith("Sources:\n\n- [1] [Ep 36 (3:08-4:07)](https://y/a)")


def test_star_and_bullet_glyphs_too():
    t = "X [2].\n\nSources:\n* [2] [A](https://y/a)\n• [4] [B](https://y/b)"
    out = chain(t)
    assert "X [1]." in out and "- [1] [A](https://y/a)" in out and "[B]" not in out


def test_unnumbered_list_is_numbered_in_order_and_uncited_dropped():
    t = ("It is a gap [1] and a zone [2].\n\nSources:\n- [Ep 12 (1:07:55)](https://y/a)\n"
         "- [Ep 12 (5:27)](https://y/b)\n- [Lines and boxes](https://y/c)")
    out = chain(t)
    assert out.endswith("Sources:\n\n- [1] [Ep 12 (1:07:55)](https://y/a)\n- [2] [Ep 12 (5:27)](https://y/b)")


def test_unnumbered_list_without_markers_is_left_alone():
    t = "Plain answer.\n\nSources:\n- [Ep 12](https://y/a)\n- [Ep 36](https://y/b)"
    assert chain(t) == t


def test_already_clean_list_unchanged_and_idempotent():
    t = "A [1] B [2].\n\nSources:\n\n- [1] [A](https://y/a)\n- [2] [B](https://y/b)"
    assert chain(t) == t and chain(chain(t)) == t


def test_code_blocks_untouched():
    t = "```python\nprint(x[1])\n```\nSee [2].\n\nSources:\n- [2] [A](https://y/a)"
    out = chain(t)
    assert "print(x[1])" in out and "See [1]." in out and "- [1] [A](https://y/a)" in out
