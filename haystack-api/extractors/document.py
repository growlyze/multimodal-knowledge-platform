"""
Document extractor — PDF, DOCX, TXT, HTML, Markdown.

For PDF/DOCX: forwards to Tika (it's good at these).
For plain text formats: reads directly (no processing needed).
"""

import logging
from urllib.parse import quote

import httpx

import config

logger = logging.getLogger("haystack-api.extract.document")

# Extensions that can be read as plain text
PLAIN_TEXT_EXTENSIONS = {"txt", "md", "html", "rtf"}

# /extract/process has to answer 200 with a non-empty body or Open WebUI fails the
# upload outright, so a failed extraction returns prose rather than raising. That
# prose is indistinguishable from content to everything downstream: ingestion
# chunked it and indexed it as the document, and the filter injected it as the
# file's text — which is how a header-encoding bug became an assistant confidently
# summarising its own error message. These markers are what extraction_failed()
# recognises; producer and predicate share them so they cannot drift apart.
FAILED_MARKER = "Extraction failed:"
TIMEOUT_MARKER = "Extraction timed out"


def extraction_failed(text: str) -> bool:
    """Is this one of our own failure placeholders rather than a document?

    Consumers that would otherwise store or quote it should treat it as no text
    at all. It stays a plain string on the wire because the upload depends on it.
    """
    return text.startswith("Document: ") and (
        FAILED_MARKER in text or TIMEOUT_MARKER in text)


async def extract_document(filename: str, ext: str, file_bytes: bytes) -> str:
    """Extract text from document files."""

    # Plain text — just decode
    if ext in PLAIN_TEXT_EXTENSIONS:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    # PDF, DOCX, etc. — forward to Tika
    return await _extract_via_tika(filename, file_bytes)


def content_disposition(filename: str) -> str:
    """RFC 6266 attachment header for a name that is not necessarily ASCII.

    HTTP header values are ASCII and httpx encodes them as such, so a filename
    carrying a curly apostrophe — what Word, macOS and most CMSes produce, and
    what a PDF named "Pakistan's economic development.pdf" arrives with — raised
    UnicodeEncodeError before the request left the process. Tika never saw the
    file; the exception text was returned as the document's content and stored
    as if the PDF had said it.

    Both halves are here for a reason: `filename*` is the real name for anything
    that reads RFC 6266, and the quoted fallback keeps the extension, which is
    the only part of this Tika uses — it detects the type from it.
    """
    ascii_name = filename.encode("ascii", "ignore").decode("ascii")
    # Neither may appear raw inside a quoted-string parameter.
    ascii_name = ascii_name.replace("\\", "_").replace('"', "_").strip()
    if not ascii_name or ascii_name.startswith("."):
        # A wholly non-Latin name leaves nothing behind. Keep the extension.
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
        ascii_name = "upload.%s" % (ext.encode("ascii", "ignore").decode("ascii") or "bin")
    return "attachment; filename=\"%s\"; filename*=UTF-8''%s" % (
        ascii_name, quote(filename, safe=""))


async def _extract_via_tika(filename: str, file_bytes: bytes) -> str:
    """
    Forward file to Tika for extraction.
    We use Tika only for file types it's actually good at (PDF, DOCX).
    """
    tika_url = f"{config.TIKA_URL}/tika"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.put(
                tika_url,
                content=file_bytes,
                headers={
                    "Content-Disposition": content_disposition(filename),
                    "Accept": "text/plain",
                },
            )
            response.raise_for_status()
            text = response.text.strip()
            logger.info(f"Tika extracted {len(text)} chars from {filename}")
            # B1's "cap the Tika fallback": Tika concatenates package formats
            # (including decompressed archives) without a bound of its own.
            # Ordinary PDFs are far under this and see no change.
            if len(text) > config.EXTRACT_MAX_TEXT_CHARS:
                logger.warning(
                    f"Tika output for {filename} truncated at "
                    f"{config.EXTRACT_MAX_TEXT_CHARS:,} chars (was {len(text):,})")
                text = (text[: config.EXTRACT_MAX_TEXT_CHARS]
                        + f"\n[extraction truncated at "
                          f"{config.EXTRACT_MAX_TEXT_CHARS:,} characters]")
            return text

    except httpx.TimeoutException:
        logger.error(f"Tika timeout for {filename}")
        return (f"Document: {filename}. {TIMEOUT_MARKER}. "
                "File is available for download.")

    except Exception as e:
        logger.error(f"Tika extraction failed for {filename}: {e}")
        return (f"Document: {filename} ({len(file_bytes):,} bytes). "
                f"{FAILED_MARKER} {e}")
