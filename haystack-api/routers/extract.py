"""
/extract — External Content Extraction endpoint for Open WebUI.

Open WebUI's ExternalDocumentLoader (v0.11.0) calls us like this:

    PUT  {EXTERNAL_DOCUMENT_LOADER_URL}/process   <- it appends /process itself
    body:    raw file bytes (NOT multipart)
    headers: Content-Type:  <mime>
             Authorization: Bearer <EXTERNAL_DOCUMENT_LOADER_API_KEY>
             X-Filename:    <url-quoted basename>   <- only source of the name
    expects: {"page_content": str, "metadata": dict}

Anything non-2xx, or an empty body, makes the upload fail in the UI.

We route by file type and return fast:
  - CSV/XLSX -> metadata summary (milliseconds)
  - PDF/DOCX -> forward to Tika (real extraction)
  - Images   -> placeholder (vision model handles it)
  - Code     -> read as plain text
"""

import logging
import secrets
from urllib.parse import unquote

from fastapi import APIRouter, Request, HTTPException

import config
from extractors.router import route_extraction

logger = logging.getLogger("haystack-api.extract")
router = APIRouter()


@router.put("/process")
async def process(request: Request):
    """Extract text from an uploaded file. Mounted at /extract, so: PUT /extract/process."""
    # Trust boundary — fail closed rather than serving unauthenticated extraction.
    if not config.EXTERNAL_LOADER_API_KEY:
        logger.error("EXTERNAL_LOADER_API_KEY is not set — refusing all requests")
        raise HTTPException(status_code=500, detail="EXTERNAL_LOADER_API_KEY not configured")

    if not secrets.compare_digest(
        request.headers.get("authorization", ""),
        f"Bearer {config.EXTERNAL_LOADER_API_KEY}",
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    # Open WebUI url-quotes the basename. No X-Filename -> no extension -> the
    # router's decode-then-Tika fallback handles it.
    filename = unquote(request.headers.get("x-filename", "")) or "unknown"
    content_type = request.headers.get("content-type", "")

    file_bytes = await request.body()

    try:
        text = await route_extraction(filename, content_type, file_bytes)
    except Exception as e:
        logger.error(f"Extraction failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"{filename} ({content_type}): {len(file_bytes):,} bytes -> {len(text):,} chars")

    # metadata must be a dict, never None — it goes straight into a langchain Document.
    return {"page_content": text, "metadata": {"source": filename}}
