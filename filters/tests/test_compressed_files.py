"""Compressed uploads at the filter layer (B1).

Two behaviors are pinned here:

1. A single-stream .gz/.bz2 routes as the file INSIDE it — "data.csv.gz" gets
   the data-file treatment (sandbox path + load_file) exactly like "data.csv",
   not the archive treatment. Containers (zip/tar/...), bare .gz and compressed
   media stay archives: the vision model needs raw bytes, and a container is
   whatever it holds, not what its name says.
2. A .zip upload is expanded automatically (1.37): every supported file inside
   it is processed as an individual upload — same classification, same prompt
   treatment, same sandbox paths, same per-file ingestion — with a per-file
   report on the turn. The pre-1.37 behavior (strip + sandbox extraction hint)
   survives only for archives this container cannot open (tar/rar/7z) or when
   the bytes are not readable at all.

Regression cases for the uncompressed routing table run alongside.
"""
import asyncio
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter, SINGLE_STREAM_EXTS


def make_filter() -> Filter:
    f = Filter()
    f.valves.enable_async_ingestion = False
    return f


class TestRouteClassification:

    CASES = [
        # single-stream compression promotes to the inner type
        ("data.csv.gz", "structured"),
        ("SALES_2020.CSV.GZ", "structured"),
        ("trades.json.gz", "structured"),
        ("data/2020/set.csv.gz", "structured"),
        ("data.csv.bz2", "structured"),
        ("report.pdf.gz", "document"),
        ("notes.txt.gz", "document"),
        ("script.py.gz", "code"),
        # containers and unknowns stay archives
        ("chart.png.gz", "archive"),      # vision needs the raw upload
        ("data.csv.zip", "archive"),      # a zip is a container whatever its name
        ("backup.zip", "archive"),
        ("x.tar.gz", "archive"),
        ("x.tgz", "archive"),
        ("x.gz", "archive"),              # bare: nothing to promote to
        ("old.rar", "archive"),
        ("old.7z", "archive"),
        # regressions: the uncompressed table is untouched
        ("data.csv", "structured"),
        ("report.pdf", "document"),
        ("chart.png", "media"),
        ("script.py", "code"),
        ("mystery.xyz", None),
    ]

    @pytest.mark.parametrize("name,kind", CASES)
    def test_route(self, name, kind):
        assert make_filter().route(name) == kind

    def test_single_stream_exts_match_the_archive_hints(self):
        # the promoted formats are exactly the stdlib single-stream ones
        assert SINGLE_STREAM_EXTS == {"gz", "bz2"}


class TestHandleFiles:

    @staticmethod
    def body(*filenames):
        return {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this?"}],
            "metadata": {
                "files": [
                    {"file": {"id": "f%d" % i, "filename": name}}
                    for i, name in enumerate(filenames, 1)
                ]
            },
        }

    @staticmethod
    def run(body):
        f = make_filter()
        calls = []
        f._ingest = lambda kind, name, record, user, chat_id=None: calls.append(
            (kind, name))
        data_files, attachments, archives = asyncio.run(f._handle_files(body, None))
        return calls, data_files, attachments, archives, body

    def test_zip_is_stripped_named_and_ingested(self):
        calls, data_files, _, archives, body = self.run(self.body("backup.zip"))
        # stripped from the request so Open WebUI cannot expand it uncapped
        assert body["metadata"]["files"] == [], body["metadata"]["files"]
        assert archives == [{
            "name": "backup.zip",
            "path": "/mnt/uploads/f1_backup.zip",
            "extractable": True,
        }], archives
        message = body["messages"][0]["content"]
        assert "/mnt/uploads/f1_backup.zip" in message, message
        assert "compressed archive" in message, message
        # ...but its contents are no longer silently discarded
        assert calls == [("archive", "backup.zip")], calls

    def test_gzip_data_file_gets_the_data_file_treatment(self):
        calls, data_files, _, archives, body = self.run(self.body("data.csv.gz"))
        assert archives == [], archives
        assert data_files == [{"name": "data.csv.gz",
                               "path": "/mnt/uploads/f1_data.csv.gz"}], data_files
        message = body["messages"][0]["content"]
        assert "/mnt/uploads/f1_data.csv.gz" in message, message
        assert "data file" in message, message
        assert calls == [("structured", "data.csv.gz")], calls

    def test_gzip_document_gets_the_document_treatment(self):
        calls, data_files, _, archives, body = self.run(self.body("notes.txt.gz"))
        assert data_files == [] and archives == [], (data_files, archives)
        # no extracted text on hand (no file record content in this fixture),
        # so it stays in the request like any document extraction missed
        assert calls == [("document", "notes.txt.gz")], calls

    def test_rar_is_named_unextractable_and_still_ingested(self):
        calls, _, _, archives, body = self.run(self.body("old.rar"))
        assert archives[0]["extractable"] is False, archives
        assert calls == [("archive", "old.rar")], calls

    def test_uncompressed_uploads_are_unchanged(self):
        calls, data_files, _, archives, body = self.run(
            self.body("data.csv", "chart.png"))
        assert archives == [], archives
        assert data_files == [{"name": "data.csv",
                               "path": "/mnt/uploads/f1_data.csv"}], data_files
        kept = [e["file"]["filename"] for e in body["metadata"]["files"]]
        assert kept == ["chart.png"], kept          # vision keeps the image
        assert calls == [("structured", "data.csv"), ("media", "chart.png")], calls
        assert "code interpreter" in body["messages"][0]["content"]

    def test_mixed_upload_kinds_each_get_their_own_path(self):
        calls, data_files, _, archives, body = self.run(
            self.body("data.csv.gz", "backup.zip", "chart.png"))
        assert data_files == [{"name": "data.csv.gz",
                               "path": "/mnt/uploads/f1_data.csv.gz"}], data_files
        assert [a["name"] for a in archives] == ["backup.zip"], archives
        kept = [e["file"]["filename"] for e in body["metadata"]["files"]]
        assert kept == ["chart.png"], kept
        assert calls == [("structured", "data.csv.gz"),
                         ("archive", "backup.zip"),
                         ("media", "chart.png")], calls


# ── automatic zip expansion (1.37) ───────────────────────────────────────────

def make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def extract_record(sections: dict, total: int = None) -> str:
    """The /extract output haystack-api produces for a container."""
    total = total if total is not None else len(sections)
    lines = ["Compressed archive: backup.zip (%d file(s), 100 bytes compressed "
             "-> 200 bytes expanded)" % total]
    for index, (name, body) in enumerate(sections.items(), 1):
        lines += ["", "--- [%d/%d] %s (10 bytes) ---" % (index, total, name), body]
    return "\n".join(lines)


class TestArchiveExpansion:

    @staticmethod
    def zip_body(tmp_path, files, sections=None, filename="backup.zip", id_="f1",
                 with_path=True):
        payload = make_zip(files)
        target = tmp_path / ("%s_%s" % (id_, filename))
        target.write_bytes(payload)
        record = {"id": id_, "filename": filename}
        if with_path:
            record["path"] = str(target)
        record["data"] = {"content": extract_record(
            sections if sections is not None
            else {n: "text of %s" % n for n in files})}
        body = {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this archive?"}],
            "metadata": {"files": [{"file": record}]},
        }
        return body, str(target)

    @staticmethod
    def run(body):
        f = make_filter()
        calls = []
        f._ingest = lambda kind, name, record, user, chat_id=None: calls.append(
            (kind, name))
        data_files, attachments, archives = asyncio.run(f._handle_files(body, None))
        return calls, data_files, attachments, archives, body

    def test_acceptance_zip_with_four_files_matches_individual_uploads(self, tmp_path):
        """csv + pdf + txt + py in one zip == uploading all four separately."""
        files = {
            "report.pdf": b"%PDF-fake",
            "data/set.csv": b"time,close\n2020-03-11,8628.75\n",
            "notes.txt": b"A paragraph of text.",
            "script.py": b"import pandas as pd\n",
        }
        body, _ = self.zip_body(tmp_path, files, sections={
            "report.pdf": "Report text about exports.",
            "data/set.csv": "File: data/set.csv\nColumns (2): time, close",
            "notes.txt": "A paragraph of text.",
            "script.py": "// Source: script.py (Python)",
        })
        calls, data_files, attachments, archives, body = self.run(body)

        # all four processed individually, in archive order
        assert sorted(a["name"] for a in attachments) == [
            "notes.txt", "report.pdf", "script.py"], attachments
        by_name = {a["name"]: a for a in attachments}
        assert by_name["report.pdf"]["text"] == "Report text about exports."
        assert by_name["notes.txt"]["text"] == "A paragraph of text."
        assert by_name["script.py"]["text"].strip() == "import pandas as pd"
        assert data_files == [{"name": "data/set.csv",
                               "path": "/mnt/uploads/f1_002_data/set.csv"}], data_files
        assert calls == [("document", "report.pdf"),
                         ("structured", "data/set.csv"),
                         ("document", "notes.txt"),
                         ("code", "script.py")], calls

        # the csv member exists at the exact sandbox path the model is told
        member = tmp_path / "f1_002_data" / "set.csv"
        assert member.read_bytes() == files["data/set.csv"]

        # feedback on the turn, and no manual-extraction hint
        message = body["messages"][0]["content"]
        assert "extracted automatically" in message, message
        assert "4 of 4 file(s) processed" in message, message
        assert "data/set.csv (data file, sandbox path: /mnt/uploads/f1_002_data/set.csv)" in message, message
        assert "attached as a compressed archive" not in message, message

        # and it equals what four individual uploads produce
        individual = {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this archive?"}],
            "metadata": {"files": [
                {"file": {"id": "a1", "filename": "report.pdf",
                          "data": {"content": "Report text about exports."}}},
                {"file": {"id": "a2", "filename": "data/set.csv"}},
                {"file": {"id": "a3", "filename": "notes.txt",
                          "data": {"content": "A paragraph of text."}}},
                {"file": {"id": "a4", "filename": "script.py",
                          "data": {"content": "// Source: script.py (Python)"}}},
            ]},
        }
        ind_calls, ind_data, ind_attach, _, _ = self.run(individual)
        assert [a["name"] for a in ind_attach] == ["report.pdf", "notes.txt", "script.py"]
        assert ind_data == [{"name": "data/set.csv",
                             "path": "/mnt/uploads/a2_data/set.csv"}], ind_data
        assert ind_calls == [("document", "report.pdf"),
                             ("structured", "data/set.csv"),
                             ("document", "notes.txt"),
                             ("code", "script.py")], ind_calls

    def test_one_failed_member_does_not_lose_the_rest(self, tmp_path):
        body, _ = self.zip_body(tmp_path, {
            "good.txt": b"A paragraph of text.",
            "bad.pdf": b"%PDF-fake",
        }, sections={"good.txt": "A paragraph of text.",
                     "bad.pdf": "Skipped: Document: bad.pdf. Extraction failed: boom"})
        calls, data_files, attachments, archives, body = self.run(body)
        assert [a["name"] for a in attachments] == ["good.txt"], attachments
        message = body["messages"][0]["content"]
        assert "bad.pdf FAILED (no extractable text)" in message, message
        assert "1 of 2 file(s) processed" in message, message

    def test_member_text_cannot_impersonate_another_file(self, tmp_path):
        # the extraction record is misaligned on purpose: only names that match
        # the unpacked member at the same position may be trusted
        body, _ = self.zip_body(tmp_path, {
            "real.txt": b"A paragraph of text.",
            "real.pdf": b"%PDF-fake",
        }, sections={"fake.txt": "IMPOSTOR", "real.txt": "A paragraph of text."})
        _, _, attachments, _, body = self.run(body)
        assert [a["name"] for a in attachments] == ["real.txt"], attachments
        assert "IMPOSTOR" not in body["messages"][0]["content"], body["messages"][0]
        assert "real.pdf FAILED (no extractable text)" in body["messages"][0]["content"]

    def test_traversal_member_is_skipped_and_never_written(self, tmp_path):
        body, _ = self.zip_body(tmp_path, {
            "../evil.txt": b"escaped",
            "good.txt": b"A paragraph of text.",
        })
        _, _, attachments, _, body = self.run(body)
        assert [a["name"] for a in attachments] == ["good.txt"], attachments
        message = body["messages"][0]["content"]
        assert "skipped (unsafe path in archive)" in message, message
        assert not (tmp_path.parent / "evil.txt").exists()
        assert not any("evil" in p.name for p in tmp_path.iterdir()), list(tmp_path.iterdir())

    def test_corrupt_zip_is_reported_not_guessed(self, tmp_path):
        target = tmp_path / "f1_backup.zip"
        target.write_bytes(b"PK\x03\x04garbage-not-a-zip")
        body = {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this archive?"}],
            "metadata": {"files": [{"file": {"id": "f1", "filename": "backup.zip",
                                             "path": str(target)}}]},
        }
        calls, data_files, attachments, archives, body = self.run(body)
        assert calls == [] and data_files == [] and attachments == [], (
            calls, data_files, attachments)
        message = body["messages"][0]["content"]
        assert "could not be extracted automatically" in message, message
        assert "not a readable ZIP archive" in message, message
        assert "attached as a compressed archive" not in message, message

    def test_member_limit_is_enforced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag_filter, "_EXPAND_MAX_MEMBERS", 2)
        body, _ = self.zip_body(tmp_path, {"f%d.txt" % i: b"x" for i in range(5)})
        _, _, attachments, _, body = self.run(body)
        assert attachments == [], attachments
        assert "over the 2-file limit" in body["messages"][0]["content"]

    def test_zip_without_readable_bytes_falls_back_to_the_extraction_hint(self):
        body = {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this archive?"}],
            "metadata": {"files": [{"file": {"id": "f3", "filename": "backup.zip"}}]},
        }
        calls, data_files, attachments, archives, body = self.run(body)
        assert data_files == [] and attachments == [], (data_files, attachments)
        assert archives == [{"name": "backup.zip",
                             "path": "/mnt/uploads/f3_backup.zip",
                             "extractable": True}], archives
        message = body["messages"][0]["content"]
        assert "attached as a compressed archive" in message, message
        assert calls == [("archive", "backup.zip")], calls

    def test_expanded_turn_is_still_a_file_turn(self):
        marker = {"name": "backup.zip", "path": "/mnt/uploads/f1_backup.zip",
                  "extractable": False, "expanded": True}
        # no contradiction: an expanded archive adds no extraction instruction
        assert rag_filter._archive_block([marker]) == ""
        # but an unexpandable one beside it still gets its own hint
        other = {"name": "old.rar", "path": "/mnt/uploads/f2_old.rar",
                 "extractable": False}
        block = rag_filter._archive_block([marker, other])
        assert "old.rar" in block and "not supported" in block, block

    def test_expansion_is_cached_across_turns(self, tmp_path):
        first_body, _ = self.zip_body(tmp_path, {"good.csv": b"time,close\n2020-03-11,8628.75\n"})
        f = make_filter()
        calls = []
        f._ingest = lambda kind, name, record, user, chat_id=None: calls.append(
            (kind, name))
        first = asyncio.run(f._handle_files(first_body, None))
        # a follow-up turn is a fresh body carrying the same file record
        second_body, _ = self.zip_body(tmp_path, {"good.csv": b"time,close\n2020-03-11,8628.75\n"})
        second = asyncio.run(f._handle_files(second_body, None))
        assert first[0] == second[0] == [{"name": "good.csv",
                                          "path": "/mnt/uploads/f1_001_good.csv"}]
        assert len(f._expanded) == 1, f._expanded

    def test_id_only_references_resolve_through_the_file_table(self, tmp_path, monkeypatch):
        """API-driven chats pass {id, type} with no record - resolve it."""
        zip_path = tmp_path / "f1_backup.zip"
        zip_path.write_bytes(make_zip({"good.csv": b"time,close\n2020-03-11,8628.75\n"}))

        class Row:
            filename = "backup.zip"
            path = str(zip_path)
            data = {"content": extract_record({"good.csv": "summary of good.csv"})}
            meta = {"name": "backup.zip"}

        async def get_file_by_id(file_id):
            return Row()

        import sys
        import types
        files_mod = types.ModuleType("open_webui.models.files")
        files_mod.Files = type("Files", (), {"get_file_by_id": staticmethod(get_file_by_id)})
        models_mod = types.ModuleType("open_webui.models")
        models_mod.files = files_mod
        owui_mod = types.ModuleType("open_webui")
        owui_mod.models = models_mod
        monkeypatch.setitem(sys.modules, "open_webui", owui_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models", models_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models.files", files_mod)

        body = {
            "chat_id": "c1",
            "messages": [{"role": "user", "content": "what is in this?"}],
            "metadata": {"files": [{"id": "f1", "type": "file"}]},
        }
        f = make_filter()
        calls = []
        f._ingest = lambda kind, name, record, user, chat_id=None: calls.append(
            (kind, name))
        data_files, attachments, archives = asyncio.run(f._handle_files(body, None))
        assert data_files == [{"name": "good.csv",
                               "path": "/mnt/uploads/f1_001_good.csv"}], data_files
        assert calls == [("structured", "good.csv")], calls


class TestArchiveContentFromFileTable:
    """P218 (live 2026-10-25, browser batch B): in a real chat request Open WebUI's
    body["files"] entry carries only {"status": "completed"} in file.data - no
    `content` - so _section_texts saw nothing and every non-plain-text member (the
    PDF in es_test_bundle.zip) was reported "FAILED (no extractable text)" although
    haystack had extracted it. The single-file path already falls back to Open
    WebUI's own file table (_fetch_file_content_fallback); the archive path didn't."""

    def test_pdf_member_text_comes_from_the_file_table(self, tmp_path, monkeypatch):
        files = {"DL_ES1_test.csv": b"time,close\n1,2\n", "strategy_note.pdf": b"%PDF-fake"}
        body, _ = TestArchiveExpansion.zip_body(tmp_path, files)
        record = body["metadata"]["files"][0]["file"]
        record["data"] = {"status": "completed"}          # what the browser sends

        async def fake_fallback(entry):
            return extract_record({"DL_ES1_test.csv": "File: DL_ES1_test.csv",
                                   "strategy_note.pdf": "The code is MAGIC_PDF_314."})
        monkeypatch.setattr(rag_filter, "_fetch_file_content_fallback", fake_fallback)
        calls, data_files, attachments, archives, body = TestArchiveExpansion.run(body)
        by_name = {a["name"]: a for a in attachments}
        assert by_name["strategy_note.pdf"]["text"] == "The code is MAGIC_PDF_314."
        assert [d["name"] for d in data_files] == ["DL_ES1_test.csv"]

    def test_no_fallback_call_when_the_entry_already_has_content(self, tmp_path, monkeypatch):
        files = {"report.pdf": b"%PDF-fake"}
        body, _ = TestArchiveExpansion.zip_body(tmp_path, files,
                                               sections={"report.pdf": "Report text."})

        async def boom(entry):
            raise AssertionError("fallback must not run when content is present")
        monkeypatch.setattr(rag_filter, "_fetch_file_content_fallback", boom)
        calls, data_files, attachments, archives, body = TestArchiveExpansion.run(body)
        assert attachments[0]["text"] == "Report text."
