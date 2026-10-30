"""
Smallest thing that fails if extraction breaks. Run: python tests/test_extraction.py

No framework on purpose — this checks the routing table and each extractor's
output shape, which is what silently rots when someone adds a file type.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractors.document import content_disposition
from extractors.router import route_extraction, _get_extension


CSV = b"time,open,high,low,close,volume\n2020-03-11 09:30,8620.5,8630.0,8615.25,8628.75,1420\n2020-03-11 09:35,8628.75,8640.0,8626.0,8637.5,1655\n"

CODE = b"use std::collections::HashMap;\n\n// Counts occurrences of each word.\nfn word_count(words: &[&str]) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in words {\n        *counts.entry(word).or_insert(0) += 1;\n    }\n    counts\n}\n"


async def main():
    assert _get_extension("sales_2020.csv") == "csv"
    assert _get_extension("Report.PDF") == "pdf"
    assert _get_extension("no_extension") == ""

    # CSV -> metadata summary, not the raw rows. This is the whole point of Phase 1.
    csv_out = await route_extraction("sales_2020.csv", "text/csv", CSV)
    assert "Columns (6)" in csv_out, csv_out
    assert "volume" in csv_out
    assert "code interpreter" in csv_out.lower()
    assert "1420" not in csv_out.split("Sample data")[0], "row data leaked into the header section"

    # Source code -> plain text, tagged with the language.
    code_out = await route_extraction("word_count.rs", "text/plain", CODE)
    assert "Rust" in code_out, code_out
    assert "word_count" in code_out, "source body must survive"

    # Image -> cheap placeholder, no bytes processing.
    img_out = await route_extraction("chart.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048)
    assert "Image file" in img_out and "chart.png" in img_out, img_out

    # A filename with a curly apostrophe — what Word and macOS produce, and what
    # "Pakistan’s economic development.pdf" arrives as. HTTP header values are
    # ASCII, so interpolating this straight into Content-Disposition raised
    # UnicodeEncodeError before the request left the process: Tika never saw the
    # file, and the exception text was stored as the document's own content.
    curly = "2fd54c74-1949-4f84-9eeb-6f3d74911b35_Pakistan’s development.pdf"
    header = content_disposition(curly)
    header.encode("ascii")                      # the assertion is that this works
    assert 'filename="2fd54c74-1949-4f84-9eeb-6f3d74911b35_Pakistans development.pdf"'         in header, header
    assert "filename*=UTF-8''" in header and "%E2%80%99" in header, header

    # Nothing ASCII survives, but Tika detects the type from the extension, so the
    # fallback has to keep one.
    assert 'filename="upload.pdf"' in content_disposition("中文.pdf")
    assert 'filename="upload.bin"' in content_disposition("中文")
    # A quote or a backslash would break out of the quoted-string parameter.
    safe = content_disposition('a"b' + chr(92) + "c.pdf")
    assert 'filename="a_b_c.pdf"' in safe, safe
    assert content_disposition("plain.pdf").startswith('attachment; filename="plain.pdf"')

    # The whole point: the request now reaches Tika instead of dying in httpx.
    # Tika is not running here, so a connection error is the success case and a
    # UnicodeEncodeError is the regression.
    curly_out = await route_extraction(curly, "application/pdf", b"%PDF-1.4 fake")
    assert "codec can't encode" not in curly_out, curly_out

    # Unknown extension, undecodable bytes -> falls through to the Tika path.
    # Tika is not running here, so we assert it degrades instead of raising.
    odd_out = await route_extraction("mystery.xyz", "application/octet-stream", b"\xff\xfe\x00\x01")
    assert "mystery.xyz" in odd_out, odd_out

    print("ok")


if __name__ == "__main__":
    asyncio.run(main())
