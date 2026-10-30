"""
GET /costs/retrieval — what chat questions' retrieval has cost (P161 item 3).

Its own file, with its own APIRouter(prefix="/costs"), so GET /costs/chat can
live in routers/costs.py without the two colliding; FastAPI merges routers that
share a prefix. Read by the admin panel's Costs page, which shows ingestion,
chat and retrieval as three separate sections.
"""

from fastapi import APIRouter, Depends, Query

from evaluation.eval_db import get_eval_db
from pipelines import retrieval_cost
from services.auth import require_admin

router = APIRouter(prefix="/costs", tags=["costs"], dependencies=[Depends(require_admin)])


@router.get("/retrieval")
async def retrieval_costs(days: int = Query(30, ge=1, le=365,
                                            description="Look-back window; raw rows are kept 90 days")):
    """Totals, per service, per day and per search for the last `days` days.

    available=false (with zeros) when the cost database is unreachable, so the
    page can say so instead of showing a misleading $0.
    """
    return await retrieval_cost.summarize(days, db=get_eval_db())
