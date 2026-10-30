"""M4 live web search: the model picks the right SOURCE per question (sprint/web).

Four sources, in priority order (SOURCE_CHOICE_RULE):
  C  conversation reuse  - the answer is already in an earlier turn ("what was the
                            total again?") - reuse it, do not search or recompute;
  K  knowledge base      - the knowledge base's own documents, the video library,
                            uploaded documents, stored datasets - never the web;
  W  web search          - current or outside facts that change (news, a central-
                            bank decision, current prices, a software release);
  G  general knowledge   - stable facts needing no lookup ("what does CSV stand for?").

classify_source() is only a HINT (a one-line nudge for obvious W and C turns); the
rule decides. It must never push a K or C question toward the web.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter

# Open WebUI 0.11.4 hands a filter the models-dict entry, whose DB row sits under
# "info" (utils/models.py: model['info'] = custom_model.model_dump()).
WEB_MODEL = {"id": "claude-sonnet-5", "info": {"meta": {
    "capabilities": {"terminal": True, "web_search": True},
    "builtinTools": {"web_search": True, "knowledge": False}}}}
NO_WEB_MODEL = {"id": "claude-sonnet-5", "info": {"meta": {
    "capabilities": {"terminal": True, "web_search": True},
    "builtinTools": {"web_search": False}}}}

HISTORY_COMPUTE = [
    {"role": "user", "content": "compute the totals in my file"},
    {"role": "assistant", "content": "Total: 12,450 units over 214 rows; mean 48.1%."},
]
HISTORY_FED = [
    {"role": "user", "content": "what did the central bank decide at the last meeting?"},
    {"role": "assistant", "content": "It held rates at 3.75-4.00% [1].\n\nSources:\n- [1] [Statement](https://example.com/x.htm)"},
]

# (question, history, expected class). >= 40, every class, the tricky ones decided
# and explained inline.
LABELLED = [
    # ── W: current / outside facts ──────────────────────────────────────────
    ("What is the latest news on the S&P 500 today?", None, "W"),
    ("What did the Fed decide this week?", None, "W"),
    ("What was the outcome of the most recent Federal Reserve meeting?", None, "W"),
    ("What is the current inflation print for the euro area?", None, "W"),
    ("Search the web for the CPI release date this month", None, "W"),
    ("What's the latest version of pandas?", None, "W"),
    ("Is the stock market open tomorrow?", None, "W"),
    ("What is the price of gold right now?", None, "W"),
    ("Any news on Nvidia earnings this quarter?", None, "W"),
    ("When is the central bank's next meeting?", None, "W"),
    ("What are the current opening hours this holiday week?", None, "W"),
    ("Google the current 10-year treasury yield", None, "W"),
    ("Look up online who won the election yesterday", None, "W"),
    ("What happened in the markets today?", None, "W"),
    # Explicit request beats the knowledge-base gate on purpose - the rule then
    # makes the model label the result as outside views, never as the knowledge
    # base's teaching.
    ("Search the web for what other people say about the new policy", None, "W"),
    # ── K: the knowledge base is the authority ──────────────────────────────
    ("What does the manual say about exports?", None, "K"),
    ("according to the documentation, how do I configure it?", None, "K"),
    ("What is the latest concept taught in Episode 12?", None, "K"),
    ("In Episode 12, what does he draw at 1:08:00?", None, "K"),
    ("Do any of the videos show a Bitcoin chart?", None, "K"),
    # "latest" alone must not send a knowledge-base question to the web.
    ("What is the latest version covered in the documentation?", None, "K"),
    ("How do I use pd.read_csv according to the docs?", None, "K"),
    ("Write a script that marks the events in the documentation", None, "K"),
    # Decided: the catalogue answers existence questions, so they are K.
    ("What datasets do you have stored?", None, "K"),
    ("Show me the current values in the stored data", None, "K"),
    ("What is in the handbook about backups today?", None, "K"),
    ("Explain the current guidance in the manual", None, "K"),
    # ── C: already answered earlier in this chat ────────────────────────────
    ("What was the total again?", HISTORY_COMPUTE, "C"),
    ("Remind me what the mean was", HISTORY_COMPUTE, "C"),
    ("How many rows did you say there were?", HISTORY_COMPUTE, "C"),
    ("What was that number you gave me earlier?", HISTORY_COMPUTE, "C"),
    ("Can you repeat the decision you mentioned above?", HISTORY_FED, "C"),
    ("What did you say the rate was?", HISTORY_FED, "C"),
    # ── G: stable general knowledge ─────────────────────────────────────────
    ("What does CSV stand for?", None, "G"),
    ("What is a moving average?", None, "G"),
    ("What is the capital of Romania?", None, "G"),
    ("Convert 5 miles to km", None, "G"),
    ("What is the difference between a stock and a bond?", None, "G"),
    ("What is standard deviation?", None, "G"),
    ("Who was Einstein?", None, "G"),
    # "again" with NO earlier turn is not C - nothing to reuse.
    ("What was the total again?", None, "G"),
    # ── traps ───────────────────────────────────────────────────────────────
    # a fresh run is not reuse
    ("calculate the totals again with 2 filters", HISTORY_COMPUTE, "K"),
    # "current"/"latest"/"news" next to course material or stored data stays K
    ("What is the current trend in the sales data?", None, "K"),
    ("What does the latest episode say about the setup?", None, "K"),
    # "current" with no moving subject is not W
    ("What is the current best way to learn chess?", None, "G"),
]


def test_the_labelled_set_is_big_enough_and_covers_every_class():
    assert len(LABELLED) >= 40
    assert {c for _q, _h, c in LABELLED} == {"W", "K", "C", "G"}


@pytest.mark.parametrize("q,history,expected", LABELLED)
def test_classify_source(q, history, expected):
    assert rag_filter.classify_source(q, history or []) == expected, q


# ── availability: read from what Open WebUI will actually offer this turn ─────

def test_web_available_needs_the_chat_toggle_and_the_model_builtin():
    on = {"features": {"web_search": True}}
    off = {"features": {"web_search": False}}
    assert rag_filter._web_search_available(on, WEB_MODEL)
    assert not rag_filter._web_search_available(off, WEB_MODEL)
    assert not rag_filter._web_search_available({}, WEB_MODEL)
    assert not rag_filter._web_search_available(on, NO_WEB_MODEL)
    caps_off = {"info": {"meta": {"capabilities": {"web_search": False}}}}
    assert not rag_filter._web_search_available(on, caps_off)
    # builtinTools absent defaults to enabled in Open WebUI (is_builtin_tool_enabled)
    assert rag_filter._web_search_available(on, {"info": {"meta": {}}})
    assert rag_filter._web_search_available(on, None)


# ── injection ─────────────────────────────────────────────────────────────────

def _inlet(query, *, web, model=WEB_MODEL, history=(), hits=(), sources=""):
    async def _search(*_a, **_k):
        return list(hits), None, sources
    f = Filter()
    f._search = _search
    msgs = [dict(m) for m in history] + [{"role": "user", "content": query}]
    body = {"metadata": {"chat_id": "web1"}, "messages": msgs,
            "features": {"web_search": web}, "terminal_id": "t1"}
    asyncio.run(f.inlet(body, __model__=model))
    return "\n".join(str(m.get("content")) for m in body["messages"]
                     if m.get("role") == "system")


def test_rule_is_on_every_turn_with_web_on():
    for q in ("What does the manual say about exports?", "What does CSV stand for?",
              "What did the Fed decide this week?"):
        s = _inlet(q, web=True)
        assert rag_filter.SOURCE_CHOICE_RULE in s, q
        assert rag_filter.SOURCE_CHOICE_RULE_NO_WEB not in s


def test_without_web_the_rule_never_mentions_the_search_tool():
    s = _inlet("What did the Fed decide this week?", web=False)
    assert rag_filter.SOURCE_CHOICE_RULE_NO_WEB in s
    assert rag_filter.SOURCE_CHOICE_RULE not in s
    assert "search_web" not in s
    assert rag_filter.WEB_SEARCH_NUDGE not in s


def test_rule_wording_contract():
    r = rag_filter.SOURCE_CHOICE_RULE
    # names the tool Open WebUI 0.11.4 actually exposes
    assert "search_web" in r
    # knowledge-base material is never answered from the web
    assert "never from the web" in r
    # an explicit web request on a knowledge-base topic is labelled as outside views
    assert "outside" in r
    # web pages are cited through the same Sources machinery the outlet repairs
    assert "[n] [Title](URL)" in r
    # the one sanctioned exception to SOURCES_RULE's "never add an entry"
    assert "only entries you may add" in r
    # reuse beats recompute
    assert "earlier in this conversation" in r
    # must not trip the outlet's own context-wording fixer / CONTEXT_WORD_RULE
    assert "context" not in r.lower()
    nw = rag_filter.SOURCE_CHOICE_RULE_NO_WEB
    assert "earlier in this conversation" in nw and "context" not in nw.lower()
    # short: it rides on every turn
    assert rag_filter.count_tokens(r) < 400


def test_obvious_w_question_gets_the_nudge_only_when_web_is_on():
    s = _inlet("What did the Fed decide this week?", web=True)
    assert rag_filter.WEB_SEARCH_NUDGE in s


@pytest.mark.parametrize("q", [
    "What does the manual say about exports?",
    "What is the latest version covered in the documentation?",
    "What datasets do you have stored?",
    "What does CSV stand for?",
])
def test_k_and_g_questions_never_get_the_web_nudge(q):
    assert rag_filter.WEB_SEARCH_NUDGE not in _inlet(q, web=True)


def test_c_question_gets_the_reuse_nudge_and_no_web_nudge():
    s = _inlet("What was the total again?", web=True, history=HISTORY_COMPUTE)
    assert rag_filter.CONVERSATION_REUSE_NUDGE in s
    assert rag_filter.WEB_SEARCH_NUDGE not in s


def test_kb_gap_keeps_no_kb_info_and_is_not_rescued_by_the_web():
    # Zero KB hits on a knowledge-base question: NO_KB_INFO's agreed label must
    # stay, and the turn must carry the "never from the web" clause, not a web
    # nudge.
    s = _inlet("Explain the guidance in the documentation", web=True)
    assert rag_filter.NO_KB_INFO in s
    assert rag_filter.WEB_SEARCH_NUDGE not in s
    assert "never from the web" in s


def test_existing_injections_unchanged_apart_from_the_rule():
    # whole-picture: on a KB hit turn every block that was there before is still there
    hit = {"content": "A summary of the three-row layout.", "source": "Ep 12",
           "citation": "[Ep 12](https://example.com/x)", "score": 0.9}
    s = _inlet("What does the manual say about exports?", web=True, hits=[hit],
               sources="[1] [Ep 12](https://example.com/x)")
    assert rag_filter.IDENTITY_GUARD in s
    assert "three-row layout" in s
    assert "[1] [Ep 12](https://example.com/x)" in s


def test_rule_sits_right_after_the_identity_guard():
    s = _inlet("What does CSV stand for?", web=True)
    assert s.index(rag_filter.IDENTITY_GUARD) < s.index(rag_filter.SOURCE_CHOICE_RULE)


def test_search_query_not_widened_by_the_rule():
    seen = []

    async def _search(q, **_k):
        seen.append(q)
        return [], None, ""
    f = Filter()
    f._search = _search
    body = {"metadata": {"chat_id": "w2"}, "features": {"web_search": True},
            "messages": [{"role": "user", "content": "What does CSV stand for?"}]}
    asyncio.run(f.inlet(body, __model__=WEB_MODEL))
    assert seen == ["What does CSV stand for?"]


# ── outlet: web pages go through the same Sources repair as KB entries ───────

WEB_REPLY = (
    "The bank held rates at 3.75-4.00% [3], per a web search. Its framework for "
    "this is in the lecture [1].\n\n"
    "Sources:\n"
    "[1] [Episode 12 - How To](https://example.com/abc?t=60)\n"
    "[2] [API Reference - format()](https://example.com/docs/format)\n"
    "[3] [Federal Reserve statement, October 2026](https://example.gov/press/20261028a.htm)\n"
    "[4] [Reference rate (Wikipedia)](https://en.wikipedia.org/wiki/Reference_rate)\n"
)


def test_web_sources_dropped_renumbered_and_formatted_like_kb_sources():
    out = rag_filter._format_sources_list(rag_filter._renumber_citations(
        rag_filter._drop_uncited_sources(rag_filter._normalize_sources_block(WEB_REPLY))))
    # [3] cited first -> becomes [1]; [1] -> [2]; uncited [2] and [4] dropped
    assert "3.75-4.00% [1]" in out and "lecture [2]" in out
    assert "- [1] [Federal Reserve statement, October 2026](https://example.gov/press/20261028a.htm)" in out
    assert "- [2] [Episode 12 - How To](https://example.com/abc?t=60)" in out
    assert "docs/format" not in out and "wikipedia" not in out


def test_web_url_with_parentheses_survives_untouched():
    text = ("Rates are set by the bank [1].\n\nSources:\n"
            "[1] [Reference rate (Wikipedia)](https://en.wikipedia.org/wiki/Reference_rate)\n")
    out = rag_filter._format_sources_list(rag_filter._renumber_citations(
        rag_filter._drop_uncited_sources(rag_filter._normalize_sources_block(text))))
    assert "- [1] [Reference rate (Wikipedia)](https://en.wikipedia.org/wiki/Reference_rate)" in out


def test_bulleted_web_only_sources_block_is_normalised():
    # a W-only turn has no KB block to copy; the model writes its own, often bulleted
    text = ("The agency raised margins [1] and the change took effect Monday [2].\n\n"
            "Sources:\n"
            "- [Clearing Advisory](https://example.com/notices/2026/10/advisory.html)\n"
            "- [Reuters: margin hike](https://www.reuters.com/markets/margins-2026-10-20/)\n")
    out = rag_filter._format_sources_list(rag_filter._renumber_citations(
        rag_filter._drop_uncited_sources(rag_filter._normalize_sources_block(text))))
    assert "- [1] [Clearing Advisory](https://example.com/notices/2026/10/advisory.html)" in out
    assert "- [2] [Reuters: margin hike](https://www.reuters.com/markets/margins-2026-10-20/)" in out


def test_web_url_with_query_and_fragment_is_never_rewritten():
    url = "https://example.com/markets/equities/sp/contract.html?a=1&b=[2]#x"
    text = "Rules changed [1].\n\nSources:\n[1] [Specs](%s)\n" % url
    out = rag_filter._format_sources_list(rag_filter._renumber_citations(
        rag_filter._drop_uncited_sources(rag_filter._normalize_sources_block(text))))
    assert url in out


# ── found by the replay over cached chats ─────────────────────────────────────

def test_repeating_a_computation_is_a_new_run_not_reuse():
    q = ("Using the uploaded `sales_data, 5.csv` file, repeat the 9/21 "
         "crossover analysis in summary mode")
    assert rag_filter.classify_source(q, HISTORY_COMPUTE) != "C"
    assert rag_filter.classify_source(q, HISTORY_COMPUTE, has_files=True) == "K"


def test_a_turn_with_an_attached_file_is_k():
    for q in ("What time zone are the times in?", "Is the volume usable?",
              "What was the total for each month?"):
        assert rag_filter.classify_source(q, [], has_files=True) == "K", q


def test_a_code_follow_up_is_k():
    h = [{"role": "user", "content": "Write a script that marks the events"},
         {"role": "assistant", "content": "```python\nprint(1)\n```"}]
    assert rag_filter.classify_source("Now make it a strategy with a 1.5% stop and 3% target.", h) == "K"


# ---- P220 (live web run 2026-10-25): no date -> searches for "2025", `date` in the
# sandbox, 6-7 searches per question; gemini-3.7 curl'ed pypi from the terminal.
def test_web_rule_forbids_terminal_lookups_and_caps_searches():
    import rag_filter as rf
    r = rf.SOURCE_CHOICE_RULE
    assert "at most 2 searches" in r
    assert "never use the terminal" in r and "curl" in r


def test_today_note_carries_the_utc_date():
    import datetime
    import rag_filter as rf
    note = rf._today_note(datetime.datetime(2026, 10, 26, 1, 5, tzinfo=datetime.timezone.utc))
    assert "2026-10-26" in note and "Monday" in note and "UTC" in note
