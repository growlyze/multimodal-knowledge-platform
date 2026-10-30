"""Video results must get strict grounding (P101), with the source_type values
production actually sends.

P101's fix matched {"transcript", "ocr", "visual", "od"}, but ingestion writes
"transcript_segment", "visual_frame", "ocr_frame" and "od_frame" (and the
library answer is "video_library") -- so the check never fired live, and a
video question without domain vocabulary got the open rules that let the
model ignore the video it was handed. The self-test fixture used the short
name, which is why it passed. Offline, _search stubbed.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_filter
from rag_filter import Filter

# No knowledge-base vocabulary: only the video hit can make this turn strict.
QUESTION = "What happens to a person's ability to see their process working when they lack confidence?"


def _hit(source_type):
    return {"content": "when you lack confidence you can't see it working",
            "source": "video", "source_type": source_type, "score": 0.81,
            "citation": "[Video @ 1:05](https://youtube.com/watch?v=x&t=65)", "metadata": {}}


def _context_after_inlet(source_type):
    async def _search(*_a, **_kw):
        return [_hit(source_type)], None, "Sources:\n[1] [Video @ 1:05](https://youtube.com/watch?v=x&t=65)"
    f = Filter()
    f._search = _search
    body = {"chat_id": "vg-" + source_type, "messages": [{"role": "user", "content": QUESTION}]}
    asyncio.run(f.inlet(body, __model__={"id": "m", "meta": {"capabilities": {"terminal": True}}}))
    return "\n".join(str(m.get("content")) for m in body["messages"])


def test_question_has_no_corpus_vocabulary():
    assert not rag_filter.is_corpus_domain(QUESTION)


def test_every_production_video_source_type_gets_strict_grounding():
    for st in ("transcript_segment", "visual_frame", "ocr_frame", "od_frame", "video_library"):
        ctx = _context_after_inlet(st)
        assert rag_filter.GROUNDING_RULES in ctx, "%s: strict rules missing" % st
        assert rag_filter.GROUNDING_RULES_OPEN not in ctx, "%s: got the open rules" % st


def test_document_hits_on_a_general_question_stay_open():
    ctx = _context_after_inlet("document")
    assert rag_filter.GROUNDING_RULES_OPEN in ctx
