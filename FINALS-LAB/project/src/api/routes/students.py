"""Student listing routes backed by lazy parquet queries."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal
from pathlib import Path

import duckdb
from fastapi import APIRouter, HTTPException, Query, status

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["students"])

# ✅ FIXED PATH (same as stats.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "students.parquet"

# fallback target
TARGET = "placed"

# Initialize DuckDB connection and create view
con = duckdb.connect()
parquet_path_str = str(PARQUET_PATH).replace("\\", "/")
con.execute(f"CREATE VIEW students AS SELECT * FROM read_parquet('{parquet_path_str}')")


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
        where_clauses = []
        if cgpa_min is not None:
            where_clauses.append(f"cgpa >= {cgpa_min}")
        if cgpa_max is not None:
            where_clauses.append(f"cgpa <= {cgpa_max}")
        if internships_min is not None:
            where_clauses.append(f"internships >= {internships_min}")
        if internships_max is not None:
            where_clauses.append(f"internships <= {internships_max}")
        if placement_status is not None:
            where_clauses.append(f"{TARGET} = {placement_status}")
            
        where_str = ""
        if where_clauses:
            where_str = "WHERE " + " AND ".join(where_clauses)
            
        order_str = ""
        if sort_by:
            sort_by_clean = "".join(c for c in sort_by if c.isalnum() or c == "_")
            desc_str = "DESC" if sort_order == "desc" else "ASC"
            order_str = f"ORDER BY {sort_by_clean} {desc_str}"
            
        offset = (page - 1) * limit
        limit_str = f"LIMIT {limit} OFFSET {offset}"
        
        query = f"SELECT * FROM students {where_str} {order_str} {limit_str}"
        data = con.execute(query).fetch_arrow_table().to_pylist()
        
        count_query = f"SELECT COUNT(*) FROM students {where_str}"
        total = con.execute(count_query).fetchone()[0]

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