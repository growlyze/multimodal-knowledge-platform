"""
Code extractor — source code files.

Reads as plain text. No processing needed.
Code files are small and text-native.
"""

import logging

logger = logging.getLogger("haystack-api.extract.code")


async def extract_code(filename: str, ext: str, file_bytes: bytes) -> str:
    """Read source code as plain text with language annotation."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    lang = _detect_language(ext)

    return f"// Source: {filename} ({lang})\n// Lines: {text.count(chr(10)) + 1}\n\n{text}"


def _detect_language(ext: str) -> str:
    """Map extension to language name."""
    lang_map = {
        "py": "Python",
        "js": "JavaScript",
        "ts": "TypeScript",
        "jsx": "React JSX",
        "tsx": "React TSX",
        "java": "Java",
        "cpp": "C++",
        "c": "C",
        "h": "C/C++ Header",
        "cs": "C#",
        "go": "Go",
        "rs": "Rust",
        "rb": "Ruby",
        "php": "PHP",
        "swift": "Swift",
        "kt": "Kotlin",
        "scala": "Scala",
        "sh": "Shell",
        "bash": "Bash",
        "sql": "SQL",
        "r": "R",
    }
    return lang_map.get(ext, ext.upper())
