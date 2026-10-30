"""
Code chunker — source code.

A "block" is one definition plus the comment lines directly above it, because
the comment is the explanation and splitting it off the code makes both halves
useless. Blocks are indivisible; chunks are groups of whole blocks.

Target 200-400 tokens, 50 token overlap (config.CHUNK_SIZE_CODE / _OVERLAP_CODE).

Definition boundaries: Python def/class, JS/TS function and arrow assignments,
Rust fn, Go func, Java method signatures, and `name(args) =>` expression
functions (Julia, Scala, Kotlin, and similar).
"""

import logging
import os
import re

import config
from pipelines.chunkers.base import (
    BaseChunker,
    Unit,
    count_tokens,
    pack_units,
    units_to_chunks,
)

logger = logging.getLogger("haystack-api.ingest")

# A block bigger than this gets split at blank lines rather than becoming a
# single enormous chunk. Multiples of the target, so it moves with the config.
OVERSIZE_FACTOR = float(os.getenv("CODE_OVERSIZE_FACTOR", "2.5"))

_COMMENT = re.compile(r"^\s*(?://|#|--|/\*|\*|\*/)")
_DECORATOR = re.compile(r"^\s*@\w")

# Each alternative captures the definition's name in group "name" where one exists.
_DEFINITION = re.compile(
    r"""^\s*(?:
        (?:export\s+)?(?:method\s+)?(?P<expr_fn>\w+)\s*\([^()]*\)\s*=>   # expression function name(args) =>
      | (?:async\s+)?def\s+(?P<py_fn>\w+)                         # Python
      | class\s+(?P<cls>\w+)                                      # Python / JS / Java class
      | (?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(?P<js_fn>\w+)
      | (?:export\s+)?(?:const|let|var)\s+(?P<js_arrow>\w+)\s*=\s*(?:async\s*)?[\(f]
      | (?:pub\s+)?fn\s+(?P<rs_fn>\w+)                            # Rust
      | func\s+(?:\([^)]*\)\s*)?(?P<go_fn>\w+)                    # Go
      | (?:public|private|protected|internal)\s+[\w<>\[\],\s]*?(?P<java_fn>\w+)\s*\(
    )""",
    re.VERBOSE,
)

_EXT_LANGUAGE = {
    "py": "Python", "js": "JavaScript", "ts": "TypeScript",
    "jsx": "React JSX", "tsx": "React TSX", "java": "Java", "cpp": "C++", "c": "C",
    "h": "C/C++ Header", "cs": "C#", "go": "Go", "rs": "Rust", "rb": "Ruby",
    "php": "PHP", "swift": "Swift", "kt": "Kotlin", "scala": "Scala", "sh": "Shell",
    "bash": "Bash", "sql": "SQL", "r": "R",
}


def detect_language(source: str) -> str:
    ext = source.rsplit(".", 1)[-1].lower() if source and "." in source else ""
    return _EXT_LANGUAGE.get(ext, "code")


def definition_name(line: str):
    """Name of the thing defined on this line, or None if it defines nothing."""
    match = _DEFINITION.match(line)
    if not match:
        return None
    named = [v for v in match.groupdict().values() if v]
    return named[0] if named else "definition"


def symbol_names(text: str) -> list:
    """Function / class names defined in text, in order, without duplicates."""
    names = []
    for line in text.splitlines():
        match = _DEFINITION.match(line)
        if not match or match.groupdict().get("annotation"):
            continue
        name = next((v for v in match.groupdict().values() if v), None)
        if name and name not in names:
            names.append(name)
    return names


def block_ranges(text: str) -> list:
    """[(start_line, end_line, name), ...] — one entry per definition block.

    A run of comments or decorators directly above a definition belongs to that
    definition, not to the one before it: the comment is the explanation, and
    separating the two makes both halves useless in a retrieval result.

    The chunker builds its units from this, and evaluation/chunk_inspector.py
    checks the finished chunks against it, so both agree on what "one function"
    means by construction.
    """
    lines = text.splitlines()
    starts = [0]
    for i, line in enumerate(lines):
        if i == 0 or not definition_name(line):
            continue
        begin = i
        while begin > 0 and (_COMMENT.match(lines[begin - 1])
                             or _DECORATOR.match(lines[begin - 1])):
            begin -= 1
        if begin > 0 and begin != starts[-1]:
            starts.append(begin)

    starts = sorted(set(starts))
    ranges = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1        # trailing blank lines belong to nobody
        if end <= start:
            continue
        name = next((definition_name(ln) for ln in lines[start:end]
                     if definition_name(ln)), None)
        ranges.append((start, end, name))
    return ranges


class CodeChunker(BaseChunker):
    def __init__(self, max_tokens: int = None, overlap_tokens: int = None,
                 min_tokens: int = None, language: str = None):
        self.max_tokens = max_tokens or config.CHUNK_SIZE_CODE
        self.overlap_tokens = (
            config.CHUNK_OVERLAP_CODE if overlap_tokens is None else overlap_tokens
        )
        self.min_tokens = min_tokens or int(self.max_tokens * 0.66)
        self.language = language

    def chunk(self, text: str, metadata: dict = None) -> list:
        if not text or not text.strip():
            return []

        metadata = dict(metadata or {})
        metadata.setdefault(
            "language", self.language or detect_language(metadata.get("source", ""))
        )

        units = self._units(text)
        packed = pack_units(units, self.max_tokens, self.overlap_tokens, self.min_tokens)
        chunks = units_to_chunks(packed, metadata, joiner="\n")
        for chunk in chunks:
            names = symbol_names(chunk.content)
            if names:
                chunk.metadata["function_name"] = names[0]
                chunk.metadata["function_names"] = names
            else:
                chunk.metadata.pop("function_name", None)
        logger.info("code chunker: %d lines -> %d blocks -> %d chunks",
                    text.count("\n") + 1, len(units), len(chunks))
        return chunks

    # ── blocks ──────────────────────────────────────────

    def _units(self, text: str) -> list:
        """Source -> one Unit per definition block (comments included)."""
        lines = text.splitlines()
        units = []
        for start, end, name in block_ranges(text):
            body = "\n".join(lines[start:end]).strip("\n")
            if not body.strip():
                continue
            meta = {"start_line": start + 1, "end_line": end}
            if name:
                meta["function_name"] = name
            units.extend(self._split_oversize(body, meta))
        return units

    def _split_oversize(self, body: str, meta: dict) -> list:
        """One definition larger than OVERSIZE_FACTOR x target: split at blank
        lines so a chunk never begins mid-statement.

        ponytail: a real parser (tree-sitter) would split at statement level.
        Blank lines are where humans already put the seams and cost nothing.
        """
        limit = int(self.max_tokens * OVERSIZE_FACTOR)
        if count_tokens(body) <= limit:
            return [Unit(text=body, boundary=True, meta=meta)]

        parts, current = [], []
        for paragraph in re.split(r"\n\s*\n", body):
            candidate = current + [paragraph]
            if current and count_tokens("\n\n".join(candidate)) > self.max_tokens:
                parts.append("\n\n".join(current))
                current = [paragraph]
            else:
                current = candidate
        if current:
            parts.append("\n\n".join(current))

        return [Unit(text=p, boundary=(i == 0), meta=dict(meta))
                for i, p in enumerate(parts) if p.strip()]


if __name__ == "__main__":
    code = """# Flag a spike when a value exceeds the rolling mean by 3 sigma.
def detect_spike(values, window):
    mean = rolling_mean(values, window)
    return [v > 3 * m for v, m in zip(values, mean)]

# Smooth a series with an exponentially weighted moving average.
def smooth_series(values, alpha):
    out = []
    for v in values:
        out.append(alpha * v + (1 - alpha) * (out[-1] if out else v))
    return out

class Smoother:
    last = 0

    def step(self, value):
        self.last = value
        return self.last
"""
    # Deliberately small target so the file has to split into several chunks.
    out = CodeChunker(max_tokens=40, overlap_tokens=10).chunk(
        code, {"source": "spike.py", "source_type": "code"})
    assert len(out) > 1, "expected multiple chunks"
    names = {n for c in out for n in c.metadata.get("function_names", [])}
    assert {"detect_spike", "smooth_series"} <= names, names
    # A function body must not be split away from its definition.
    for fn, body in [("detect_spike", "return [v > 3 * m"),
                     ("smooth_series", "return out")]:
        owners = [c for c in out if re.search(r"^def %s\(" % fn, c.content, re.M)]
        assert owners and any(body in c.content for c in owners), fn
        # ...and the explanation comment stays with it.
        assert any("#" in c.content.split(fn)[0] for c in owners), fn
    assert all(c.metadata["language"] == "Python" for c in out)
    print("%d chunks, functions: %s" % (len(out), sorted(names)))
