"""Routing-accuracy and term-gate regression tests for rag_filter.py.

Three things live here, all pure-logic and offline (see test_properties.py's
docstring for why the filter files themselves cannot be pytest'd whole):

  1. A LABELLED query set measuring needs_code_execution() as precision/recall
     rather than spot checks. Phase 7 of the reliability plan asked for a number,
     because "it feels over-broad" is not something a later change can regress
     against. Baseline when this file was written (2026-09-24): precision 0.500,
     knowledge-only specificity 0.455.

  2. TERM-VARIANT tests for every regex gate in rag_filter.py — the P40
     generalization. P40 was a term gate missing plurals; auditing all ten
     gates the same way found the same blind spot in six more of them, so each
     gate now pins its own inflections. A gate that matches "the manual" but not
     "the manuals" downgrades retrieval silently, which is why this bug class
     keeps being found by report rather than by test.

  3. _sanitize_tool_call_ids() (P29) against a Gemini-origin history, to the
     extent that is checkable without a live cross-provider turn. What is NOT
     checkable here is marked in the test that covers it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import json

import pytest

import rag_filter
from rag_filter import (
    _TOOL_ID_OK_RE,
    _sanitize_tool_call_ids,
    is_conceptual_question,
    is_corpus_domain,
    is_script_generation_request,
    is_meta_capability_query,
    needs_code_execution,
    references_additional_stored_data,
    references_kb_material,
    references_stored_data,
)

# ── 1. labelled routing set ─────────────────────────────

# Weighted toward the ambiguous middle on purpose: the easy cases ("compute the
# totals of my data", "what is a lookup table?") were never the problem. Every
# "knowledge" row below except the last block contains at least one
# CODE_KEYWORDS substring, which is what made the old precision 0.500.
#
# Labels are the INTENT of the turn, not what the filter does with it:
#   "code"      — the answer requires running Python over data
#   "knowledge" — the answer is a definition/opinion the corpus or the model has
#   "generation" — a code-artifact request; tested separately below
KNOWLEDGE_WITH_CODE_KEYWORD = [
    'what is a moving average?',
    'what does variance mean?',
    'explain correlation vs regression',
    'is a correlation above 0.9 realistic?',
    'how is variance calculated?',
    'why do analysts use regression instead of correlation?',
    "what's a typical variance for survey results?",
    'what does the chart of a random walk look like?',
    'should I optimize my model parameters or not?',
    'what is a parameter sweep?',
    'what is grid search used for in model tuning?',
    'what is a sweep in load testing?',
    'what is a chart legend?',
    'which chart type is best for time series?',
    'explain how to compute a median by hand',
    'what is the point of a benchmark?',
    'do I need to benchmark before going live?',
    'what benchmark do professional teams target?',
    # F11 (P185 note): "chart" is a CODE_KEYWORD and "how should I" was no
    # conceptual frame, so this reading question reached the sandbox.
    'How should I read a 5 minute latency chart?',
]

KNOWLEDGE_PLAIN = [
    "what is a lookup table?",
    "explain indexing in databases",
    "what units does the report use?",
    "define a cache miss",
    "what is the difference between a stack and a queue?",
    "what is the two-pointer technique?",
    "explain the double-spend problem",
    "when is the maintenance window?",
    "what does throttling mean in networking?",
    "how do I identify a memory leak?",
    "what is the best retry strategy?",
    "does the API use webhooks?",
    "what is expectancy?",
    "how does a deadlock form?",
    "what is the optimal buffer size?",
]

CODE_WITH_KEYWORD = [
    'compute the totals of this report on my CSV',
    'calculate the average order value for this file',
    'run a parameter sweep over window sizes 5 to 30',
    'what is the correlation between price and quantity in the attached file?',
    'compute the refund rate for the orders in orders.csv',
    'optimize the discount threshold on the data',
    'plot the monthly revenue from that export',
    'how many records are in this file?',
    'count rows where the status is refunded',
    'run this on the uploaded data',
    'grid search the window and threshold values',
    'analyze data in the attached spreadsheet',
    'simulate 1000 customers arriving per hour',
    'chart the error rate over time for these requests',
    'give me the variance and moving average for this sales log',
    'run a benchmark and show me the results',
    'sweep the threshold from 0.1 to 0.5',
    'give me the revenue trend as a chart',
]

# Real data questions that match NO code keyword. These are the recall hole, and
# they are deliberately still failures here: _route() compensates by treating any
# attachment as reason enough to inject the tools block, so the turn works anyway
# (rag_filter.py's own comment on the `attached` branch says so). Pinned as a
# number rather than fixed, so that widening CODE_KEYWORDS to chase them — which
# would cost precision again — has to be a deliberate decision with a measurement.
CODE_WITHOUT_KEYWORD = [
    'how many rows are in this file?',
    "what's the date range of the attached csv?",
    'what were the biggest 5 refunds in my file?',
    'summarise the columns in this dataset',
    'resample this to weekly totals and show the result',
    'what is the average daily volume in the attached data?',
    'measure the median response time for these requests',
]

GENERATION_REQUESTS = [
    "write a script that plots the moving average",
    "create a program that charts the results",
    "build a function that computes the totals",
    "generate a script to count rows in the log",
    "make a script to run analysis of the numbers",
]

LABELLED = (
    [(q, "knowledge") for q in KNOWLEDGE_WITH_CODE_KEYWORD]
    + [(q, "knowledge") for q in KNOWLEDGE_PLAIN]
    + [(q, "code") for q in CODE_WITH_KEYWORD]
    + [(q, "code") for q in CODE_WITHOUT_KEYWORD]
    + [(q, "generation") for q in GENERATION_REQUESTS]
)


def _confusion():
    tp = fp = fn = tn = 0
    for query, label in LABELLED:
        if label == "generation":
            continue
        predicted = needs_code_execution(query)
        wanted = label == "code"
        if predicted and wanted:
            tp += 1
        elif predicted and not wanted:
            fp += 1
        elif wanted:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def test_knowledge_only_queries_stay_knowledge_only():
    """The Phase 7 acceptance number: >=90% of knowledge-only queries must NOT
    route to the sandbox.

    A false trigger is not just a wasted hint. `code` suppresses NO_KB_INFO in
    _route(), so a definition question the corpus cannot answer stops refusing
    and answers from training data instead, under a UI that presents the reply as
    knowledge-base backed. Measured 0.455 before is_conceptual_question() existed.
    """
    tp, fp, fn, tn = _confusion()
    specificity = tn / (tn + fp)
    assert specificity >= 0.90, (
        f"{specificity:.3f} — {fp} of {tn + fp} knowledge-only queries routed to "
        f"the sandbox: "
        + ", ".join(
            repr(q)
            for q, label in LABELLED
            if label == "knowledge" and needs_code_execution(q)
        )
    )


def test_code_routing_precision_and_recall_do_not_regress():
    """Both directions, pinned together so neither can be bought with the other.

    Recall is the weaker number (0.720) and is expected to be: CODE_WITHOUT_KEYWORD
    above is a documented, compensated-for gap. It is asserted anyway so that a
    change which improves precision by narrowing the keyword list has to show what
    it cost.
    """
    tp, fp, fn, tn = _confusion()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision >= 0.85, f"precision {precision:.3f} (tp={tp} fp={fp})"
    assert recall >= 0.70, f"recall {recall:.3f} (tp={tp} fn={fn})"


def test_generation_requests_route_to_the_code_leg():
    """A code-generation turn is recognised as an artifact turn AND routes to
    the code leg."""
    leaked = [q for q in GENERATION_REQUESTS if not needs_code_execution(q)]
    assert leaked == [], leaked
    missed = [q for q in GENERATION_REQUESTS if not is_script_generation_request(q)]
    assert missed == [], missed


# ── 2. term-gate variants (the P40 generalization) ──────

# Each row: the gate's predicate, a canonical phrase it has always matched, and the
# inflections of that same phrase a user is just as likely to type. Every variant
# here returned False at some point on 2026-09-24; the audit that produced this
# table covered all the regex gates in rag_filter.py.
TERM_VARIANTS = [
    # references_kb_material — the refusal gate. Plurals and compound spelling
    # are ordinary English; a miss silently downgrades the turn's scope rules.
    (references_kb_material, "the manual", ["the manuals", "the handbook"]),
    (references_kb_material, "knowledge base", ["knowledge bases",
                                                "knowledge-bases"]),
    (references_kb_material, "the documentation", ["the docs"]),
    (references_kb_material, "according to the manual",
     ["according to the documentation", "according to the guide"]),
    # references_stored_data
    (references_stored_data, "dataset", ["datasets"]),
    (references_stored_data, "csv", ["csvs"]),
    (references_stored_data, "historical", ["history"]),
    # is_meta_capability_query
    (is_meta_capability_query, "your knowledge base", ["your knowledge bases"]),
]


@pytest.mark.parametrize(
    "predicate,canonical,variants",
    TERM_VARIANTS,
    ids=[f"{p.__name__}:{c}" for p, c, _ in TERM_VARIANTS],
)
def test_term_gate_matches_every_inflection_of_a_term_it_already_matches(
    predicate, canonical, variants
):
    assert predicate(canonical), f"fixture is wrong: {canonical!r} should match"
    missed = [v for v in variants if not predicate(v)]
    assert missed == [], (
        f"{predicate.__name__} matches {canonical!r} but not {missed} — "
        "the P40 blind spot, in a new gate"
    )


def test_additional_stored_data_needs_both_halves_and_both_inflect():
    """references_additional_stored_data() is an AND of two gates, so a plural
    miss in either one silently drops the second dataset the user asked for."""
    assert references_additional_stored_data("also load my datasets for the last 10 years")
    assert references_additional_stored_data("also load my dataset for the last 10 years")
    # The higher bar is still a bar: a passing mention is not a command.
    assert not references_additional_stored_data("compute the totals on this data")


def test_conceptual_veto_never_fires_on_a_request_naming_data():
    """The veto's safety clause. Whenever the sentence points at something to
    compute over, the old behaviour wins — including for sentences that open with
    a definitional frame."""
    for query in [
        "what is the max value of the attached series?",
        "what is the mean of this file?",
        "explain the outlier in my records log",
        "why does this csv show a negative total?",
    ]:
        assert not is_conceptual_question(query), query


# ── 3. _sanitize_tool_call_ids (P29) ────────────────────

def _gemini_origin_id() -> str:
    """A LiteLLM/Gemini-shaped tool-call id: a thought_signature blob encoded into
    the id to carry it across turns (P14) — ~260 chars instead of ~30."""
    blob = json.dumps({"__thought__": "x" * 150}).encode()
    return "call_" + base64.b64encode(blob).decode()


def test_gemini_origin_id_is_rewritten_into_claudes_alphabet():
    gemini_id = _gemini_origin_id()
    assert len(gemini_id) > 200, len(gemini_id)
    assert not _TOOL_ID_OK_RE.match(gemini_id), "fixture no longer violates the pattern"

    messages = [
        {"role": "assistant", "tool_calls": [{"id": gemini_id, "type": "function"}]},
        {"role": "tool", "tool_call_id": gemini_id, "content": "42"},
    ]
    _sanitize_tool_call_ids(messages)

    new_id = messages[0]["tool_calls"][0]["id"]
    assert _TOOL_ID_OK_RE.match(new_id), new_id
    # Both sides must move together or the pairing breaks, which Claude rejects
    # just as hard as the original id did.
    assert messages[1]["tool_call_id"] == new_id
    # Short enough that no length limit can be the next thing to trip on. What
    # Claude's actual maximum id length is remains NOT ESTABLISHED from this repo —
    # settling it needs a live Messages API call, not an offline test.
    assert len(new_id) < 40, new_id


def test_anthropic_native_content_blocks_are_rewritten_too():
    """Which of the two message shapes Open WebUI hands a filter is not verified
    from this repo, so both are covered — see the function's own docstring."""
    gemini_id = _gemini_origin_id()
    messages = [
        {"role": "assistant",
         "content": [{"type": "tool_use", "id": gemini_id, "name": "t", "input": {}}]},
        {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": gemini_id, "content": "42"}]},
    ]
    _sanitize_tool_call_ids(messages)
    new_id = messages[0]["content"][0]["id"]
    assert _TOOL_ID_OK_RE.match(new_id), new_id
    assert messages[1]["content"][0]["tool_use_id"] == new_id


def test_rewrite_is_deterministic_and_idempotent():
    """A regenerate re-runs this on history that may already be rewritten. A
    counter or a fresh uuid would drift and break the pairing on the second pass."""
    gemini_id = _gemini_origin_id()

    def once():
        messages = [{"role": "assistant",
                     "tool_calls": [{"id": gemini_id, "type": "function"}]}]
        _sanitize_tool_call_ids(messages)
        return messages[0]["tool_calls"][0]["id"]

    first, second = once(), once()
    assert first == second

    already = [{"role": "assistant", "tool_calls": [{"id": first, "type": "function"}]}]
    _sanitize_tool_call_ids(already)
    assert already[0]["tool_calls"][0]["id"] == first


def test_a_trailing_newline_does_not_smuggle_an_id_past_the_validator():
    """`$` in Python also matches before a trailing newline, so "abc\\n" satisfied
    `^[a-zA-Z0-9_-]+$` and was passed through unrewritten — into the exact 400 this
    function exists to prevent, and silently, since the log line only fires on a
    rewrite. Fixed by anchoring with `\\Z`."""
    assert not _TOOL_ID_OK_RE.match("abc\n")
    messages = [{"role": "assistant",
                 "tool_calls": [{"id": "abc\n", "type": "function"}]}]
    _sanitize_tool_call_ids(messages)
    assert messages[0]["tool_calls"][0]["id"] != "abc\n"
    assert _TOOL_ID_OK_RE.match(messages[0]["tool_calls"][0]["id"])


def test_a_compliant_history_is_left_completely_alone():
    """The no-op path. Rewriting ids that were already valid would churn every
    cross-provider turn for nothing."""
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_abc123", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "42"},
    ]
    _sanitize_tool_call_ids(messages)
    assert messages[0]["tool_calls"][0]["id"] == "call_abc123"
    assert messages[1]["tool_call_id"] == "call_abc123"


def test_sanitizer_survives_malformed_history():
    """'Degrade, never break the chat' — a filter that raises here takes the whole
    turn down, and history shapes are not this repo's to guarantee."""
    for messages in ([None], [{"tool_calls": None}], [{"content": [None, "x"]}],
                     [{"tool_calls": ["not a dict"]}], [{"tool_call_id": 17}], []):
        _sanitize_tool_call_ids(list(messages))


# ── 4. scope routing: what may be answered without corpus support ──────

# Two predicates, two different questions:
#
#   is_corpus_domain()          — is the corpus authoritative for what it returned?
#   references_kb_material()    — does the question explicitly name the knowledge
#                                 base or its written material? (the refusal gate)
#
# The refusal tier is the only gate in this file built for PRECISION rather than
# recall. Every other gate's comment says the bias is deliberately toward matching
# because "a false positive only ever buys MORE context". That stops being true
# once the gate produces a refusal: a false positive here is not a risk of
# refusing, it is a guaranteed refusal on a zero-hit turn.

# Answered normally from the model's own knowledge. None of these may be protected.
GENERAL_KNOWLEDGE = [
    "what is the speed of light?",
    "what is the capital of Romania?",
    "who was Einstein?",
    "convert 5 miles to km",
    "how many liters is 2 tons of water?",
    "what is compound interest?",
    "how do options work?",
    "what is implied volatility?",
    "what is a limit order?",
    "explain contango and backwardation",
]

# The reason the refusal tier cannot reuse the max-recall construction style.
# Every one of these contains a phrase that is knowledge-base vocabulary in one
# sense and ordinary English in another. All would refuse 100% of the time.
MATERIAL_FALSE_FRIENDS = [
    "how does a manual transmission work?",
    "what is a guide dog?",
    "what is the course of a river?",
    "who wrote a textbook on anatomy?",
    "what is raw material?",
    "how many chapters does a novel have?",
]

# Explicit references to the knowledge base or its written material, where a
# miss means the model answers from training data what the corpus is the
# authoritative record of.
KB_MATERIAL = [
    "what does the manual say?",
    "according to the documentation, how do I set it up?",
    "summarise the knowledge base",
    "explain the material in the handbook",
    "what do the lessons cover?",
    "read the docs for me",
]

# API/code references, as a docs question types them. The corpus is the right
# source for these whatever the retrieval returned.
API_GROUNDED = [
    "what is pd.read_csv?",
    "how does os.path.join work?",
    "explain the json.dumps function",
    "what is str.format()?",
    "how do I use re.findall?",
]

# Requests to PRODUCE something. These must never be refused: the artifact is the
# answer, and a missing concept definition is named rather than used as grounds to
# decline.
GENERATION = [
    "write a script that detects the documented patterns",
    "code a pattern detector",
    "write a script that summarises the documentation",
    "build a detector with a fixed window that marks duplicates",
]

# The mirror image, and the reason _SCRIPT_REQUEST_RE cannot be the exemption on
# its own: it matches all of these, and none of them asks for code.
PEDAGOGY_NOT_CODE = [
    "explain how to build a strategy around the documented patterns",
    "how do I build a strategy using the two-pointer technique?",
    "can you make a strategy out of the double-spend problem?",
    "write up a strategy for the material in the manual",
]


@pytest.mark.parametrize("query", GENERAL_KNOWLEDGE)
def test_general_knowledge_is_never_protected(query):
    """The reported bug. Zero tolerance — a match here is a refusal the user sees."""
    assert not references_kb_material(query)


@pytest.mark.parametrize("query", MATERIAL_FALSE_FRIENDS)
def test_ordinary_senses_of_the_material_phrases_are_never_protected(query):
    """A manual transmission, a guide dog, the course of a river.

    Zero tolerance for the same reason: with a false positive the gate refuses
    a turn the user expected answered.
    """
    assert not references_kb_material(query)


@pytest.mark.parametrize("query", KB_MATERIAL)
def test_kb_material_references_are_protected(query):
    """Zero tolerance in the other direction: a miss answers the knowledge
    base's own material from training data, which is what the refusal exists for."""
    assert references_kb_material(query)


@pytest.mark.parametrize("query", API_GROUNDED)
def test_api_reference_questions_stay_in_the_corpus_domain(query):
    """Code-documentation questions are corpus questions whatever the retrieval
    returned — the dotted-name shape is what catches them."""
    assert is_corpus_domain(query)


@pytest.mark.parametrize("query", KB_MATERIAL)
def test_kb_material_questions_are_also_in_the_corpus_domain(query):
    assert is_corpus_domain(query)


@pytest.mark.parametrize("query", GENERAL_KNOWLEDGE)
def test_general_knowledge_is_outside_the_corpus_domain(query):
    assert not is_corpus_domain(query)


@pytest.mark.parametrize("query", GENERATION)
def test_generation_requests_are_recognised(query):
    assert is_script_generation_request(query)


@pytest.mark.parametrize("query", PEDAGOGY_NOT_CODE)
def test_pedagogy_is_not_a_generation_request(query):
    """_SCRIPT_REQUEST_RE matches every one of these. It is a necessary condition,
    never a sufficient one — exempting on it alone hands teaching questions an
    'answer from your own knowledge' instruction."""
    assert not is_script_generation_request(query)


# A bare request with no qualification at all is still an artifact request: the
# verb+noun shape is what says so.
UNQUALIFIED_SCRIPT_IS_GENERATION = [
    "write a script that detects outliers",
    "create a program for the slow queries",
    "can you build a script for the edge cases?",
    "make a function that marks duplicates",
]

# These name the same verb+noun shape but ALSO name sandbox/analysis work, and
# must keep routing there instead — the default must not swallow an
# already-decided case.
UNQUALIFIED_SCRIPT_STAYS_SANDBOX = [
    "write a script to chart the totals",
    "write a script that computes this on my CSV",
    "build a function and calculate its totals",
]


@pytest.mark.parametrize("query", UNQUALIFIED_SCRIPT_IS_GENERATION)
def test_unqualified_script_request_is_generation(query):
    assert is_script_generation_request(query), query


@pytest.mark.parametrize("query", UNQUALIFIED_SCRIPT_STAYS_SANDBOX)
def test_unqualified_script_request_with_sandbox_signal_stays_sandbox(query):
    # The turn stays a code/sandbox turn rather than a knowledge question.
    assert needs_code_execution(query), query


@pytest.mark.parametrize("query", PEDAGOGY_NOT_CODE)
def test_unqualified_default_does_not_reach_the_plan_nouns(query):
    """is_script_generation_request()'s own docstring is explicit that
    "strategy"/"study" must not count without sandbox work - the narrower
    _ARTIFACT_REQUEST_RE noun set (script/program/detector/function/algorithm/
    program/snippet) exists for exactly this reason. Re-asserted here, since
    PEDAGOGY_NOT_CODE's queries are the ones most likely to regress if that noun
    set is ever widened back to _SCRIPT_REQUEST_RE's."""
    assert not is_script_generation_request(query), query


def test_protected_tier_is_a_subset_of_the_corpus_domain():
    """is_corpus_domain() keeps its wider meaning so that grounding strictness,
    the retrieval skip and tools_block sizing are driven by the same signal as
    before. A protected phrase outside it would silently change all three."""
    escaped = [q for q in KB_MATERIAL if not is_corpus_domain(q)]
    assert escaped == [], escaped


# ── 5. collisions found by adversarial probing, 2026-09-24 ──────

# The reported examples were easy. These are the cases between the categories,
# and they found the defect class the first labelled set could not see:
# CODE_KEYWORDS is a substring list, and ordinary questions collide with it.

# A compute verb or a pointer at concrete data is what separates a compute
# request from a knowledge question - not the wording around it.
COMPUTE_STILL_COMPUTE = [
    "compute the totals of this report on my CSV",
    "compute the totals of this report on my data",
    "simulate the queue: 10 arrivals, serve after 3 ticks",
    "calculate how many rows matched in the attached file",
]


@pytest.mark.parametrize("query", COMPUTE_STILL_COMPUTE)
def test_compute_requests_keep_their_sandbox_routing(query):
    assert needs_code_execution(query)


# A mechanism question, not a compute request. `sweep` is a CODE_KEYWORD, so
# without a causal frame in _CONCEPTUAL_FRAME_RE this reached the sandbox.
MECHANISM_NOT_COMPUTE = [
    "what causes a gap in the series when a large order sweeps the book?",
    "what makes a cache fill so quickly on thin traffic?",
]


@pytest.mark.parametrize("query", MECHANISM_NOT_COMPUTE)
def test_causal_questions_are_not_compute_requests(query):
    assert not needs_code_execution(query)


# F11: a "how should I read/use X" question about a chart is methodology, not a
# compute request. Pinned on its own as well as in the labelled set, because the
# set only asserts a rate and one regression would not move it below the bar.
READING_QUESTIONS = [
    "How should I read volume on a 5 minute chart?",
    "how should I interpret the daily chart before the open?",
]


@pytest.mark.parametrize("query", READING_QUESTIONS)
def test_how_should_i_read_a_chart_is_not_a_compute_request(query):
    assert not needs_code_execution(query)


def test_how_should_i_with_concrete_data_keeps_the_sandbox():
    assert needs_code_execution("how should I plot the totals of my file?")
