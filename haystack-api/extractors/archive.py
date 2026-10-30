"""
Shared preprocessing layer for compressed uploads (B1) - validate, decompress
and unpack .zip / .gz / .tar-family uploads into bounded member files.

This module is mechanics only (stdlib only - zipfile/tarfile/gzip/bz2, never an
external tool): naming, limits, traversal checks and byte accounting. What a
member *means* is decided by extractors/router.py, which routes each member
through the same extractors a direct upload would get.

Three rules shape everything here:

* **Nothing is written to disk.** Members are streamed out of the in-memory
  archive buffer, so archive path traversal cannot happen by construction.
  Member names are still validated (safe_member_name) and traversal-shaped
  names rejected explicitly, so the guarantee survives a future change that
  does write files.
* **Headers are not trusted.** Declared sizes and compression ratios are
  checked, and the streamed bytes are counted again as they arrive. A member
  that expands past what its header declared is treated as a decompression
  bomb and refuses the whole archive.
* **Failures are named, never silent.** Every refusal raises ArchiveError with
  a message worth showing a user; callers turn it into the shared
  "Document: ... Extraction failed: ..." placeholder (see extractors/document),
  which /extract can safely return and ingestion refuses to index.

Limits live in config.py (ARCHIVE_MAX_*): they are the file-count, expanded
-size and compression-ratio guards this task requires, and they are
configurable per deployment via environment variables.
"""

import bz2
import gzip
import io
import logging
import stat
import tarfile
import zipfile
from dataclasses import dataclass, field

import config

logger = logging.getLogger("haystack-api.extract.archive")

# Single-stream compressors: one file inside, so "data.csv.gz" is "data.csv".
SINGLE_STREAM_SUFFIXES = frozenset({"gz", "bz2"})
# Containers: a directory tree of members.
CONTAINER_SUFFIXES = frozenset({"zip", "tar", "tgz", "tar.gz", "tar.bz2"})
# Formats we name but never open: no extractor is confirmed for them anywhere
# in this stack (the sandbox hint draws the same line for rar/7z).
UNSUPPORTED_SUFFIXES = frozenset({"rar", "7z"})

_KNOWN_SUFFIXES = frozenset({"zip", "tar", "gz", "tgz", "bz2", "rar", "7z"})

# A header may understate by a little without lying; anything beyond this is a lie.
_DECLARED_SLACK = 64 * 1024
_READ_CHUNK = 1024 * 1024


class ArchiveError(ValueError):
    """The upload is not a usable compressed file. Message goes to the user."""


@dataclass
class Member:
    """One file inside an archive, already bounded and unpacked."""
    name: str
    size: int = 0
    compressed_size: int = 0
    data: bytes = b""
    status: str = "ok"        # "ok" | "skipped"
    reason: str = ""
    # True when the member came from a single-stream .gz/.bz2 (its "archive"
    # has exactly this one file in it).
    single: bool = field(default=False)


# ── naming ──────────────────────────────────────────────────────────────────


def basename(filename: str) -> str:
    return (filename or "").replace("\\", "/").rsplit("/", 1)[-1]


def compression_suffix(filename: str):
    """The compression suffix of *filename*, or None.

    "x.zip" -> "zip", "x.tar.gz" -> "tar.gz", "data.csv.gz" -> "gz",
    "x.csv" -> None. Only the basename counts.
    """
    name = basename(filename).lower()
    for suffix in ("tar.gz", "tar.bz2"):
        if name.endswith("." + suffix):
            return suffix
    if "." not in name:
        return None
    ext = name.rsplit(".", 1)[-1]
    return ext if ext in _KNOWN_SUFFIXES else None


def split_compression(filename: str):
    """("data.csv.gz",) -> ("data.csv", "gz"); ("x.csv",) -> ("x.csv", None)."""
    name = basename(filename)
    suffix = compression_suffix(name)
    if suffix is None:
        return name, None
    return name[: -(len(suffix) + 1)], suffix


def is_container(filename: str) -> bool:
    return compression_suffix(filename) in CONTAINER_SUFFIXES


def safe_member_name(name: str):
    """(ok, reason) for a member name. Traversal-shaped names are refused.

    In-memory extraction cannot be escaped even by a hostile name, but "PK
    ../etc/passwd" is still not a file we should ever report, extract or
    index - refused outright rather than silently flattened.
    """
    if not name or "\x00" in name:
        return False, "empty or NUL-containing name"
    norm = name.replace("\\", "/")
    if norm.startswith("/"):
        return False, "absolute path"
    if len(norm) > 1 and norm[1] == ":":
        return False, "drive-absolute path"
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return False, "'..' in path"
    return True, ""


# ── limits ──────────────────────────────────────────────────────────────────


def _reject_ratio(uncompressed: int, compressed: int, what: str) -> None:
    """Compression-ratio guard, the classic zip-bomb shape.

    Not applied below ARCHIVE_RATIO_FLOOR_BYTES: tiny members can compress
    1000:1 harmlessly and the absolute caps already bound them.
    """
    floor = config.ARCHIVE_RATIO_FLOOR_BYTES
    if uncompressed < floor:
        return
    ratio = uncompressed / max(compressed, 1)
    if ratio > config.ARCHIVE_MAX_RATIO:
        raise ArchiveError(
            "%s expands at %.0f:1 - over the ARCHIVE_MAX_RATIO limit of %.0f:1 "
            "(possible decompression bomb; raise ARCHIVE_MAX_RATIO if this file "
            "is legitimate)" % (what, ratio, config.ARCHIVE_MAX_RATIO))


def _reject_total(total: int) -> None:
    if total > config.ARCHIVE_MAX_TOTAL_BYTES:
        raise ArchiveError(
            "the archive expands to %s - over the ARCHIVE_MAX_TOTAL_BYTES limit "
            "of %s (possible decompression bomb)"
            % (human_bytes(total), human_bytes(config.ARCHIVE_MAX_TOTAL_BYTES)))


def _reject_count(count: int) -> None:
    if count > config.ARCHIVE_MAX_MEMBERS:
        raise ArchiveError(
            "the archive contains %d entries - over the ARCHIVE_MAX_MEMBERS "
            "limit of %d" % (count, config.ARCHIVE_MAX_MEMBERS))


def human_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return "%.1f MB" % (n / (1024 * 1024))
    if n >= 1024:
        return "%.1f KB" % (n / 1024)
    return "%d bytes" % n


def _read_limited(stream, limit: int, what: str) -> bytes:
    """Read at most *limit* bytes, counting as we go.

    Exceeding the limit means the header lied or the member is oversized -
    either way the bytes stop here and the archive is refused.
    """
    out = bytearray()
    while True:
        chunk = stream.read(_READ_CHUNK)
        if not chunk:
            return bytes(out)
        out += chunk
        if len(out) > limit:
            raise ArchiveError(
                "%s expands past its declared size / the per-file limit of %s "
                "(possible decompression bomb)"
                % (what, human_bytes(limit)))


# ── single-stream .gz / .bz2 ────────────────────────────────────────────────


def decompress_single(filename: str, file_bytes: bytes):
    """One compressed stream -> (inner filename, decompressed bytes).

    Bounded while streaming: the expanded size cap, the compression-ratio cap
    and the declared-lie check all apply as bytes arrive, not after.
    """
    name = basename(filename)
    base, suffix = split_compression(name)
    if suffix not in SINGLE_STREAM_SUFFIXES:
        raise ArchiveError("%s is not a single-stream compressed file" % name)

    opener = gzip.open if suffix == "gz" else bz2.open
    cap = config.ARCHIVE_MAX_MEMBER_BYTES
    allowed = max(config.ARCHIVE_RATIO_FLOOR_BYTES,
                  int(len(file_bytes) * config.ARCHIVE_MAX_RATIO))
    out = bytearray()
    try:
        with opener(io.BytesIO(file_bytes), "rb") as stream:
            while True:
                chunk = stream.read(_READ_CHUNK)
                if not chunk:
                    break
                out += chunk
                if len(out) > allowed:
                    raise ArchiveError(
                        "%s expands at more than %.0f:1 - over the "
                        "ARCHIVE_MAX_RATIO limit (possible decompression bomb)"
                        % (name, config.ARCHIVE_MAX_RATIO))
                if len(out) > cap:
                    raise ArchiveError(
                        "%s decompresses to more than %s - over the "
                        "ARCHIVE_MAX_MEMBER_BYTES limit"
                        % (name, human_bytes(cap)))
    except ArchiveError:
        raise
    except Exception as exc:  # BadGzipFile, EOFError, OSError, zlib.error, ...
        raise ArchiveError(
            "%s is not a readable %s stream: %s: %s"
            % (name, suffix, type(exc).__name__, exc))

    if not out:
        raise ArchiveError("%s decompresses to zero bytes" % name)
    logger.info("%s: %s -> %s (single stream)", name,
                human_bytes(len(file_bytes)), human_bytes(len(out)))
    return base, bytes(out)


# ── zip / tar containers ────────────────────────────────────────────────────


def _zip_member_problem(info: zipfile.ZipInfo):
    """Skip reason for a zip member we refuse to unpack, or None to unpack it."""
    ok, why = safe_member_name(info.filename)
    if not ok:
        return "unsafe path in archive (%s)" % why
    if info.flag_bits & 0x1:
        return "encrypted member"
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        return "symbolic link"
    if info.file_size > config.ARCHIVE_MAX_MEMBER_BYTES:
        return ("individual file is %s - over the ARCHIVE_MAX_MEMBER_BYTES "
                "limit of %s; upload it on its own"
                % (human_bytes(info.file_size),
                   human_bytes(config.ARCHIVE_MAX_MEMBER_BYTES)))
    return None


def _read_zip(name: str, file_bytes: bytes):
    try:
        archive = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as exc:  # BadZipFile and friends - named either way
        raise ArchiveError("%s is not a readable ZIP archive: %s" % (name, exc))

    with archive:
        infos = archive.infolist()
        _reject_count(len(infos))
        files = [i for i in infos if not i.is_dir()]
        if not files:
            raise ArchiveError("%s contains no files" % name)

        declared_total = sum(max(i.file_size, 0) for i in files)
        _reject_total(declared_total)
        _reject_ratio(declared_total, len(file_bytes), "the archive")

        members = []
        for info in files:
            member_name = info.filename.replace("\\", "/")
            problem = _zip_member_problem(info)
            if problem:
                members.append(Member(name=member_name, size=info.file_size,
                                      compressed_size=info.compress_size,
                                      status="skipped", reason=problem))
                continue
            _reject_ratio(info.file_size, info.compress_size,
                          "member %r" % member_name)
            limit = min(config.ARCHIVE_MAX_MEMBER_BYTES,
                        info.file_size + _DECLARED_SLACK)
            with archive.open(info) as stream:
                blob = _read_limited(stream, limit, "member %r" % member_name)
            members.append(Member(name=member_name, size=len(blob),
                                  compressed_size=info.compress_size, data=blob))
        return members


def _tar_member_problem(member: tarfile.TarInfo):
    if member.isdir():
        return "directory"
    ok, why = safe_member_name(member.name)
    if not ok:
        return "unsafe path in archive (%s)" % why
    if member.issym() or member.islnk():
        return "symbolic/hard link"
    if not member.isfile():
        return "not a regular file"
    if member.size > config.ARCHIVE_MAX_MEMBER_BYTES:
        return ("individual file is %s - over the ARCHIVE_MAX_MEMBER_BYTES "
                "limit of %s; upload it on its own"
                % (human_bytes(member.size),
                   human_bytes(config.ARCHIVE_MAX_MEMBER_BYTES)))
    return None


def _read_tar(name: str, file_bytes: bytes):
    try:
        archive = tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*")
    except Exception as exc:  # TarError and friends - named either way
        raise ArchiveError("%s is not a readable tar archive: %s" % (name, exc))

    with archive:
        infos = archive.getmembers()
        _reject_count(len(infos))
        files = [m for m in infos if m.isfile()]
        if not files:
            raise ArchiveError("%s contains no files" % name)

        declared_total = sum(max(m.size, 0) for m in files)
        _reject_total(declared_total)
        _reject_ratio(declared_total, len(file_bytes), "the archive")

        members = []
        for info in infos:
            problem = _tar_member_problem(info)
            if problem == "directory":
                continue
            if problem:
                members.append(Member(name=info.name.replace("\\", "/"),
                                      size=info.size,
                                      compressed_size=0,
                                      status="skipped", reason=problem))
                continue
            _reject_ratio(info.size, max(info.size // 2, 1),
                          "member %r" % info.name)
            limit = min(config.ARCHIVE_MAX_MEMBER_BYTES,
                        info.size + _DECLARED_SLACK)
            stream = archive.extractfile(info)
            if stream is None:
                members.append(Member(name=info.name, size=info.size,
                                      status="skipped",
                                      reason="not a readable member"))
                continue
            blob = _read_limited(stream, limit, "member %r" % info.name)
            members.append(Member(name=info.name.replace("\\", "/"),
                                  size=len(blob), compressed_size=0, data=blob))
        return members


def read_members(filename: str, file_bytes: bytes):
    """Compressed upload -> bounded member list.

    Containers yield every eligible file; a single-stream .gz/.bz2 yields the
    one file inside it. Raises ArchiveError for anything unusable (corrupt,
    empty, over the limits, or a format we name but never open).
    """
    name = basename(filename)
    suffix = compression_suffix(name)
    try:
        if suffix == "zip":
            members = _read_zip(name, file_bytes)
        elif suffix in ("tar", "tgz", "tar.gz", "tar.bz2"):
            members = _read_tar(name, file_bytes)
        elif suffix in SINGLE_STREAM_SUFFIXES:
            inner, blob = decompress_single(name, file_bytes)
            return [Member(name=inner, size=len(blob),
                           compressed_size=len(file_bytes), data=blob,
                           single=True)]
        elif suffix in UNSUPPORTED_SUFFIXES:
            raise ArchiveError(
                ".%s archives are not supported - no extractor for that format "
                "is available in this stack. Re-upload the files inside it "
                "directly, or as a .zip" % suffix)
        else:
            raise ArchiveError("%s is not a supported compressed file" % name)
    except ArchiveError:
        raise
    except Exception as exc:
        # A member that corrupts mid-stream (CRC, truncated deflate, ...) lands
        # here as its raw library error. Name it like every other refusal so the
        # job fails with a reason instead of a traceback.
        raise ArchiveError("%s could not be unpacked: %s: %s"
                           % (name, type(exc).__name__, exc))

    logger.info("%s: %d member(s), %s compressed -> %s expanded",
                name, len(members), human_bytes(len(file_bytes)),
                human_bytes(sum(m.size for m in members)))
    return members
