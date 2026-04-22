"""Student listing routes backed by lazy parquet queries."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException, Query, status

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["students"])

# ✅ FIXED PATH (same as stats.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "students.parquet"

# fallback target
TARGET = "placed"


@router.get("/students")
async def get_students(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    cgpa_min: float | None = None,
    cgpa_max: float | None = None,
    internships_min: int | None = None,
    internships_max: int | None = None,
    placement_status: int | None = Query(None, ge=0, le=1),
    sort_by: str | None = None,
    sort_order: Literal["asc", "desc"] = "asc",
) -> dict[str, Any]:
    """Paginated students endpoint (optimized for 1M rows)."""

    try:
        lf = pl.scan_parquet(PARQUET_PATH)

        # ✅ FILTERS
        if cgpa_min is not None:
            lf = lf.filter(pl.col("cgpa") >= cgpa_min)

        if cgpa_max is not None:
            lf = lf.filter(pl.col("cgpa") <= cgpa_max)

        if internships_min is not None:
            lf = lf.filter(pl.col("internships") >= internships_min)

        if internships_max is not None:
            lf = lf.filter(pl.col("internships") <= internships_max)

        if placement_status is not None:
            lf = lf.filter(pl.col(TARGET) == placement_status)

        # ✅ SORTING
        if sort_by:
            lf = lf.sort(by=sort_by, descending=(sort_order == "desc"))

        # ✅ PAGINATION
        offset = (page - 1) * limit
        data_lf = lf.slice(offset, limit)

        # ✅ TOTAL COUNT (lazy)
        total_lf = lf.select(pl.len().alias("total"))

        # ✅ EXECUTE ONCE
        data = data_lf.collect().to_dicts()
        total = total_lf.collect().item()

    except Exception as exc:
        LOGGER.exception("Failed to fetch /students")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Students service is temporarily unavailable",
        ) from exc

    total_pages = math.ceil(total / limit) if total > 0 else 0

    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }