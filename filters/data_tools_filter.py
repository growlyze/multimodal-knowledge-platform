"""
title: Data Tools Awareness
author: Growlyze
version: 1.0
description: Injects the data_tools library reference into the system prompt on turns
that need the code sandbox (a data question, or an attached CSV/TSV/Excel file), so any
model that reaches for Open Terminal's run_command imports the tested library for file
loading, file summaries, charts and exports instead of reimplementing it. Stands down
when the RAG Filter has already injected its own sandbox instructions on this turn, and
on plain knowledge questions.
"""

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

# Deliberately NOT setting `file_handler = True` here. That module-level flag is what
# tells Open WebUI to hand a filter the attachments and skip its own RAG pipeline -
# rag_filter.py owns it because it owns retrieval. A second function claiming it would
# make attachment handling depend on which filter Open WebUI happened to load last.

log = logging.getLogger("data-tools-filter")

# Names and signatures are copied from code-execution/data_tools/__init__.py
# (_SIGNATURES / __all__). A wrong signature here costs a failed sandbox run, so
# re-check against data_tools.list_functions() whenever the library changes.
DATA_TOOLS_BLOCK = """## Data Tools Library

BEFORE WRITING ANY CODE, check your tool list for a CALLABLE tool that can execute \
Python (run_command). Never call a tool to list tools, or any unlisted name. The \
functions below are REFERENCE DOCS, not callable tools - they need one of those tools \
to run. If none is in your tool list: reply ONLY "I can't run code right now - code \
execution isn't available on this conversation." then STOP. No code, no steps, no \
expected results. NEVER present numbers you did not produce by executing code.

EVERY NUMBER FROM CODE: each number, date or value you state must be in output your \
code printed in this conversation, or in the user's message. Compute any difference, \
sum, count, average, percentage or conversion in code and print it first; never count \
a printed list by eye.

The sandbox has a pre-installed, tested Python library. Import from it instead of \
writing loading, summary, chart or export code from scratch.

### Available imports

```python
from data_tools import (
    load_file,            # (filepath) CSV/TSV/Excel/gzip -> DataFrame
    load_dataset,         # (name, api_url=None) a stored dataset, the whole table
    resample,             # (df, timeframe='1min', agg=None) datetime-indexed frames
    describe_file,        # (path_or_df) every fact a file summary may state
    format_file_summary,  # (info) render describe_file()'s dict as text
    export_xlsx,          # (sheets, path='/tmp/export.xlsx', column_formats=None)
    plot_line,            # (frame, y=None, title='Line Chart') -> Figure
    plot_bar,             # (frame, x=None, y=None, title='Bar Chart') -> Figure
    plot_heatmap,         # (frame, x_col, y_col, value_col, title='Heatmap') -> Figure
    format_number, format_percent, format_money, format_plain,
    format_signed, format_ratio, format_table_text,
    validate_code,        # (source) -> (ok, message) guardrail for generated code
    list_functions,       # () the full catalogue, one signature per line
)
```

### Rules

1. Load files with `load_file(path)` and compute over the WHOLE DataFrame - never \
from .head(), .tail(), .sample() or a printed table read by eye. After loading, check \
`df.attrs.get('load_issues')` and tell the user what was degraded before answering.
2. `load_file()` may move a timestamp column into the index, so `df.columns` is not \
necessarily the file's columns: for column names or counts print \
`pd.read_csv(path, nrows=0).columns.tolist()`.
3. Charts: the plot_* functions return a matplotlib Figure. Save it with \
`fig.savefig('/home/user/outputs/chart.png')` and show it with display_file.
4. Downloads: write under /home/user/outputs/ (/app is read-only), e.g. \
`export_xlsx({"Summary": df}, "/home/user/outputs/report.xlsx")` or \
`df.to_csv('/home/user/outputs/out.csv')`, then call display_file with that path and \
inline=true. Never say a file is ready unless display_file confirmed it exists.
5. Before running a longer generated script, `ok, msg = validate_code(source)`; if \
`ok` is False, fix what `msg` names instead of working around it.
6. If a step raises, report the error plainly and stop; never fill in numbers.
7. `help(function_name)` gives the exact parameters. Check there before assuming a \
signature."""

# Added when a data file was attached on this turn. %s is the file list.
FILE_SUMMARY_HINT = """### Attached data file(s): %s

For "what's in this file?" or any summary / overview / description request, run \
first, exactly: `from data_tools import describe_file, format_file_summary; \
print(format_file_summary(describe_file(path)))` with the file's sandbox path, and \
build the answer from that printed output: report every fact and every problem it \
prints, add nothing it did not print, and say clean, continuous, no gaps or no \
missing values only for a check that printed zero. Never guess what the file holds \
from its filename. For any other question, load it with load_file(path) and compute \
the answer in code."""

# Idempotence marker. Open WebUI re-runs inlet() on a regenerate or an edited message,
# against a body that may already carry the block.
OWN_MARKER = "## Data Tools Library"

# rag_filter.py's CODE_TOOLS_HINT / CODE_TOOLS_LIGHT both open with this sentence when
# it routes a turn to the sandbox. Saying the same thing twice in one prompt does not
# help the model and pays for it on every code turn.
RAG_FILTER_MARKER = "BEFORE WRITING ANY CODE, check your tool list for a CALLABLE tool that can execute Python (here that is run_command)"

# A subset of rag_filter.py's CODE_KEYWORDS, duplicated because these are two
# separately deployed Open WebUI Functions with no shared import path - keep the two
# lists roughly in sync by hand. Without a gate, a plain conceptual question ("what is
# a moving average?") would be primed to reach for the sandbox.
CODE_KEYWORDS = (
    "simulate",
    "optimize",
    "parameter sweep",
    "benchmark",
    "calculate",
    "compute",
    "run analysis",
    "analyze",
    "analyse",
    "moving average",
    "percent change",
    "variance",
    "correlation",
    "regression",
    "how many records",
    "how many rows",
    "count rows",
    "plot",
    "chart",
    "graph",
    "export",
    "excel",
    "spreadsheet",
    "run this on",
    "grid search",
)

DATA_FILE_EXTS = ("csv", "tsv", "xlsx", "xls")
_DATA_FILE_NAME_RE = re.compile(
    r"[\w ,.!()-]*?\.(?:%s)\b" % "|".join(DATA_FILE_EXTS), re.I
)

# rag_filter.py's DATA_FILE_HINT wording. By the time this filter runs, rag_filter has
# usually stripped the attachment list from body["files"], so this text (appended to
# the user's own message) is the remaining sign a data file was attached this turn.
_FILE_ATTACHED_HINT_TEXT = "attached as data file"


def _needs_code(query: str) -> bool:
    lowered = (query or "").lower()
    return any(kw in lowered for kw in CODE_KEYWORDS)


def _last_user_text(messages: list) -> str:
    """The last user message's text, or "" if there is none to read.

    Multimodal content (a list of parts) is reduced to its text parts only.
    """
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""


def _system_message(messages: list) -> Optional[dict]:
    """The first system message with string content, or None.

    Non-string content is a multimodal system message (a list of parts). Appending a
    string to it would corrupt the shape, so the caller inserts a fresh one instead.
    """
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
        ):
            return message
    return None


def _file_name(item) -> str:
    """The display name of one body["files"] entry, across Open WebUI shapes."""
    if not isinstance(item, dict):
        return ""
    inner = item.get("file") if isinstance(item.get("file"), dict) else {}
    meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
    for value in (item.get("name"), inner.get("filename"), meta.get("name"),
                  item.get("filename")):
        if isinstance(value, str) and value:
            return value
    return ""


def is_data_file(name: str) -> bool:
    return (name or "").lower().rsplit(".", 1)[-1] in DATA_FILE_EXTS and "." in (name or "")


def attached_data_files(body: dict, query: str = "") -> list:
    """Names of the CSV/TSV/Excel files attached on this turn.

    Read from body["files"] when it is still there, else from rag_filter's
    DATA_FILE_HINT note on the user's message."""
    names = [n for n in (_file_name(f) for f in (body.get("files") or [])) if is_data_file(n)]
    if names:
        return names
    if _FILE_ATTACHED_HINT_TEXT in (query or "").lower():
        note = query[query.lower().rfind("[system note:"):] if "[system note:" in query.lower() else query
        found = [m.group(0).strip(" ,") for m in _DATA_FILE_NAME_RE.finditer(note)]
        return found or ["the attached file"]
    return []


def _user_enabled(user: Optional[dict]) -> bool:
    """The per-user override, if Open WebUI passed one through.

    UserValves arrives as a pydantic model in Open WebUI and as a plain dict in tests
    and older builds, so both are read.
    """
    valves = (user or {}).get("valves")
    if valves is None:
        return True
    enabled = getattr(valves, "enabled", None)
    if enabled is None and isinstance(valves, dict):
        enabled = valves.get("enabled")
    return True if enabled is None else bool(enabled)


class Filter:
    class Valves(BaseModel):
        # Open WebUI runs filters in ascending priority. rag_filter.py declares none, so
        # it sits at 0 and runs first; by the time this filter looks, the marker it
        # checks for is already there to be found.
        priority: int = Field(
            default=10,
            description="Filter order. Keep above rag_filter (0) so the duplicate "
            "check can see what rag_filter injected.",
        )
        enabled: bool = Field(
            default=True, description="Master switch for the injection."
        )
        skip_if_rag_filter_injected: bool = Field(
            default=True,
            description="Stay silent when rag_filter already injected its own "
            "sandbox instructions on this turn.",
        )
        skip_for_pure_knowledge_questions: bool = Field(
            default=True,
            description="Stay silent when the question matches no code keyword and "
            "no data file was attached this turn - a plain conceptual question "
            "should be answered from the knowledge base, not primed for the sandbox.",
        )
        inject_file_summary_hint: bool = Field(
            default=True,
            description="When a CSV/TSV/Excel file is attached, also tell the model "
            "to summarise it with describe_file()/format_file_summary().",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Turn the data_tools reference off for this user.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _skip_reason(
        self, system_text: str, user: Optional[dict], query: str = "",
        has_files: bool = False,
    ) -> Optional[str]:
        """Why this turn gets no injection, or None to inject."""
        v = self.valves
        if not v.enabled:
            return "disabled"
        if not _user_enabled(user):
            return "disabled for user"
        if OWN_MARKER in system_text:
            return "already injected"
        if v.skip_if_rag_filter_injected and RAG_FILTER_MARKER in system_text:
            return "rag_filter covered it"
        if (
            v.skip_for_pure_knowledge_questions
            and not has_files
            and not _needs_code(query)
        ):
            return "pure knowledge question, no code keyword"
        return None

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Best-effort: a surprise in the body shape costs the injection, never the chat.
        try:
            self._inject(body, __user__)
        except Exception:
            log.exception("data_tools injection failed - passing through")
        return body

    def build_block(self, files: list) -> str:
        block = DATA_TOOLS_BLOCK
        if files and self.valves.inject_file_summary_hint:
            block += "\n\n" + FILE_SUMMARY_HINT % ", ".join(files)
        return block

    def _inject(self, body: dict, user: Optional[dict]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return

        message = _system_message(messages)
        existing = message["content"] if message else ""
        query = _last_user_text(messages)
        files = attached_data_files(body, query)

        skip = self._skip_reason(existing, user, query, bool(files))
        if skip:
            log.debug("no data_tools block (%s)", skip)
            return

        block = self.build_block(files)
        if message is None:
            messages.insert(0, {"role": "system", "content": block})
        else:
            # Appended, never prepended: the model's or the user's own system prompt
            # states who the assistant is, and this is reference material for it.
            message["content"] = "\n\n".join(part for part in (existing, block) if part)
        log.info("data_tools reference injected (%d chars)", len(block))

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Nothing to do on the way out; the method exists because Open WebUI calls it.
        return body
