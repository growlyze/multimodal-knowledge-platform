"""Property + fuzz tests for filter PURE LOGIC only.

rag_filter.py and dataset_tool.py themselves must stay framework-free single
files (they get pasted whole into Open WebUI Admin -> Functions/Tools), so
their own self-checks print "ok" with no pytest. This directory is never
pasted anywhere, so it uses pytest + Hypothesis normally. Only functions
that don't touch the network or Open
WebUI's request cycle are testable here — Filter.route() and dataset_tool's
_format()/_row() qualify; inlet()/outlet() do not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from hypothesis import given, settings, strategies as st

from dataset_tool import _format, _row
from rag_filter import Filter
import rag_filter

_VALID_KINDS = {"structured", "document", "media", "code", "archive", None}


# CODE_KEYWORDS behaviour is pinned through the labelled routing set in
# tests/test_routing_accuracy.py.


@given(st.text(max_size=100))
@settings(max_examples=200, deadline=None)
def test_route_never_raises_and_returns_a_valid_kind(filename):
    result = Filter().route(filename)
    assert result in _VALID_KINDS


def test_route_fixed_cases():
    """A few human-legible cases the property test above is hard to eyeball."""
    assert Filter().route("data.CSV") == "structured"
    assert Filter().route("noext") is None
    assert Filter().route("weird.name.with.dots.csv") == "structured"
    assert Filter().route("") is None


_metadata_value = st.one_of(
    st.text(max_size=30),
    st.integers(min_value=0, max_value=10**9),
    st.none(),
    st.lists(st.text(max_size=10), max_size=5),
)

_hit_strategy = st.fixed_dictionaries({
    "source": st.one_of(st.text(max_size=50), st.none()),
    "metadata": st.one_of(
        st.none(),
        st.dictionaries(
            st.sampled_from(["table_name", "sheet", "row_count", "columns",
                              "symbols", "date_range"]),
            _metadata_value,
            max_size=6,
        ),
    ),
})


@pytest.mark.xfail(strict=True, reason="P25: _row() has no defensive coercion for row_count/columns/symbols/date_range — see project-knowledge-base/14-problems-bugs.md; fixing dataset_tool.py is out of scope for this testing-only branch")
@given(st.lists(_hit_strategy, max_size=10))
@settings(max_examples=200, deadline=None)
def test_dataset_format_never_raises_on_malformed_hits(hits):
    """API drift (a field arriving as the wrong type) must degrade to a
    rendered string, not a 500 in the model's face — the project's
    'degrade, never break the chat' convention, applied to this tool too."""
    result = _format("query", hits)
    assert isinstance(result, str)
