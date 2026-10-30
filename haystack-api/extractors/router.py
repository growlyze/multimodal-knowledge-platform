"""
Routes files to the appropriate extractor based on file type.
Each extractor is a separate module — swappable, testable.

Compressed uploads (B1) are intercepted here before anything else: a
single-stream .gz/.bz2 is decompressed and routed as the file inside it
(data.csv.gz is data.csv), a .zip/.tar/.tgz container is unpacked into members
that each go through the same extractors below, and formats we never open
(rar/7z) are refused by name instead of falling through to Tika, which used to
decompress them wholesale into the request.
"""

from extractors.structured import extract_structured
from extractors.document import extract_document, FAILED_MARKER
from extractors.media import extract_media
from extractors.code import extract_code
from extractors import archive
from extractors.archive import ArchiveError, human_bytes

import config

# File extension → category mapping
STRUCTURED_EXTENSIONS = {"csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "doc", "txt", "rtf", "odt", "epub", "md", "html"}
MEDIA_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "svg",
                    "mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a"}
CODE_EXTENSIONS = {"py", "js", "ts", "jsx", "tsx", "java", "cpp", "c", "h",
                   "cs", "go", "rs", "rb", "php", "swift", "kt", "scala",
                   "sh", "bash", "zsh", "sql", "r", "m"}

# Two compression layers (x.csv.gz.gz) is already a curiosity; more is a
# deliberate resource attack and is refused rather than unwrapped forever.
_MAX_COMPRESSION_DEPTH = 2


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def classify_extension(ext: str):
    """Extension -> "structured" | "document" | "media" | "code" | None."""
    for kind, exts in (("structured", STRUCTURED_EXTENSIONS),
                       ("document", DOCUMENT_EXTENSIONS),
                       ("media", MEDIA_EXTENSIONS),
                       ("code", CODE_EXTENSIONS)):
        if ext in exts:
            return kind
    return None


def _looks_like_text(data: bytes) -> bool:
    """Plain text, or something a model should never be handed as text?

    UTF-8 alone does not tell the two apart - control bytes decode fine - so
    NUL bytes and a control-character ratio are part of the test. This is what
    keeps a .exe member out of the prompt.
    """
    if b"\x00" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    return control <= max(2, len(text) // 100)


def classify_member(name: str, data: bytes):
    """(kind, skip_reason) for one file unpacked from an archive.

    kind None means the member is skipped and skip_reason says why — the same
    explanation is shown to the caller whatever the model or pipeline is.
    Structured/document/code members are eligible; media is skipped because
    the vision model needs the raw upload, not bytes buried in an archive;
    nested compressed files are skipped so unpacking stays bounded; unknown
    binary is skipped, but unknown *text* is still content and goes in as a
    document.
    """
    if archive.compression_suffix(name):
        return None, ("compressed member — nested archives are not unpacked; "
                      "upload this file on its own")
    if not data:
        return None, "file is empty"
    kind = classify_extension(_get_extension(name))
    if kind == "media":
        return None, ("media file — upload it directly so the vision model "
                      "can see it")
    if kind:
        return kind, ""
    if not _looks_like_text(data):
        return None, "unsupported file type (.%s)" % (_get_extension(name) or "?")
    return "document", ""


def _refusal(filename: str, size: int, reason: str) -> str:
    """The one failure shape every refusal uses.

    Shaped like extractors/document's failure placeholders on purpose:
    extraction_failed() recognises it, so ingestion fails the job with this
    reason instead of indexing it as a document, while /extract still answers
    200 with a non-empty body (its hard contract with Open WebUI).
    """
    return "Document: %s (%s). %s %s" % (filename, human_bytes(size),
                                         FAILED_MARKER, reason)


async def route_extraction(filename: str, content_type: str, file_bytes: bytes,
                           depth: int = 0) -> str:
    """
    Route file to the correct extractor.
    Returns extracted text content.
    """
    name = archive.basename(filename)

    # Compressed uploads first: they are identified by name and must never
    # reach the decode-then-Tika fallback, which decompresses them unbounded.
    suffix = archive.compression_suffix(name)
    if suffix is not None:
        return await _extract_compressed(filename, name, file_bytes, suffix, depth)

    ext = _get_extension(filename)

    if ext in STRUCTURED_EXTENSIONS:
        return await extract_structured(filename, ext, file_bytes)

    if ext in DOCUMENT_EXTENSIONS:
        return await extract_document(filename, ext, file_bytes)

    if ext in MEDIA_EXTENSIONS:
        return await extract_media(filename, ext, file_bytes)

    if ext in CODE_EXTENSIONS:
        return await extract_code(filename, ext, file_bytes)

    # Fallback: try to read as text, if that fails → send to Tika
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return await extract_document(filename, ext, file_bytes)


async def _extract_compressed(filename: str, name: str, file_bytes: bytes,
                              suffix: str, depth: int) -> str:
    """One compressed upload -> extracted text, or a named refusal."""
    if depth >= _MAX_COMPRESSION_DEPTH:
        return _refusal(filename, len(file_bytes),
                        "compressed more than %d layers deep — decompress it "
                        "and re-upload" % _MAX_COMPRESSION_DEPTH)

    try:
        if suffix in archive.SINGLE_STREAM_SUFFIXES:
            inner_name, inner_bytes = archive.decompress_single(name, file_bytes)
            if not archive.basename(inner_name).count("."):
                sniffed = _sniff_inner_name(inner_name, inner_bytes)
                if sniffed is None:
                    return _refusal(
                        filename, len(inner_bytes),
                        "decompressed content is not a supported file type - "
                        "rename %s to name what is inside it (e.g. %s.csv.gz), "
                        "or upload that file directly" % (name, inner_name))
                inner_name = sniffed
            return await route_extraction(inner_name, "", inner_bytes,
                                          depth=depth + 1)
        members = archive.read_members(name, file_bytes)
    except ArchiveError as exc:
        return _refusal(filename, len(file_bytes), str(exc))

    return await _extract_container(filename, file_bytes, members, depth)


def _sniff_inner_name(inner_name: str, data: bytes):
    """A bare "dump.gz" has no name for its content — recognise the type.

    Returns the name to route it as, or None when the content is unrecognised
    binary (which must be refused, not guessed at).
    """
    if data[:2] == b"\x1f\x8b":
        return inner_name + ".gz"          # compressed-in-compressed
    if len(data) > 262 and data[257:262] == b"ustar":
        return inner_name + ".tar"
    return inner_name + ".txt" if _looks_like_text(data) else None


async def _extract_container(filename: str, file_bytes: bytes, members,
                             depth: int) -> str:
    """Container -> one bounded, labelled block per eligible member.

    Members are routed through the same extractors a direct upload of that
    file would get — a CSV member produces the CSV summary, a PDF member goes
    to Tika, code is read as text — so an archive's contents land in exactly
    the shape every downstream consumer already understands.
    """
    total = config.ARCHIVE_MAX_TEXT_CHARS
    header = ["Compressed archive: %s (%d file(s), %s compressed -> %s expanded)"
              % (archive.basename(filename), len(members),
                 human_bytes(len(file_bytes)),
                 human_bytes(sum(m.size for m in members)))]
    used = sum(len(h) for h in header)
    body, overflow = [], False

    for index, member in enumerate(members, 1):
        tag = "--- [%d/%d] %s (%s) ---" % (index, len(members), member.name,
                                           human_bytes(member.size))
        kind, reason = classify_member(member.name, member.data)
        if member.status != "ok":
            section = "%s\nSkipped: %s" % (tag, member.reason)
        elif kind is None:
            section = "%s\nSkipped: %s" % (tag, reason)
        else:
            text = await route_extraction(member.name, "", member.data,
                                          depth=depth + 1)
            if overflow or used + len(text) > total:
                overflow = True
                section = ("%s\n[not included: reached the "
                           "ARCHIVE_MAX_TEXT_CHARS extraction budget]" % tag)
            else:
                section = "%s\n%s" % (tag, text)
                used += len(text)
        body.append(section)

    if overflow:
        body.append("[archive text truncated at %d characters "
                    "(ARCHIVE_MAX_TEXT_CHARS)]" % total)
    return "\n\n".join(header + body)
