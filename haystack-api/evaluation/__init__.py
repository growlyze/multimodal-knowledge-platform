"""
Quality evaluation, three layers deep.

    chunk_inspector      Layer 1 — rules, runs inside every ingestion job
    coherence_judge      Layer 2 — LLM judge on a sample, opt-in
    retrieval_evaluator  Layer 3 — the retrieval test set against GET /search

Layer 1 is wired into pipelines/ingestion.py. Layers 2 and 3 run on demand:

    python -m evaluation.retrieval_evaluator --url http://localhost:8000
"""

from evaluation.chunk_inspector import check_chunk, inspect_chunks

__all__ = ["inspect_chunks", "check_chunk"]
