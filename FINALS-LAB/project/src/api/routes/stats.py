"""Statistics routes backed by lazy parquet aggregation."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException, status

# If TARGET is defined in your settings, you can keep it.
# Otherwise we fallback safely to "placed"
try:
    from src.config.settings import TARGET
except:
    TARGET = "placed"

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["stats"])


def _as_float(value: float | None, *, precision: int = 6) -> float:
    if value is None:
        return 0.0
    return round(float(value), precision)


# ✅ FIXED: correct path to parquet
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "students.parquet"


@router.get("/stats/overview")
async def get_stats_overview() -> dict[str, float | int]:
    """Return dataset-level overview statistics from parquet."""

    try:
        agg_lf = (
            pl.scan_parquet(PARQUET_PATH)
            .select(
                [
                    pl.len().alias("total_students"),
                    pl.col(TARGET).mean().alias("placement_rate"),
                    pl.col("cgpa").mean().alias("avg_cgpa"),
                    pl.col("internships").mean().alias("avg_internships"),
                ]
            )
        )

        result = agg_lf.collect()

    except Exception as exc:
        LOGGER.exception(
            "Failed to compute /stats/overview from parquet: %s", PARQUET_PATH
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stats service is temporarily unavailable",
        ) from exc

    if result.is_empty():
        return {
            "total_students": 0,
            "placement_rate": 0.0,
            "avg_cgpa": 0.0,
            "avg_internships": 0.0,
        }

    row = result.row(0, named=True)

    return {
        "total_students": int(row["total_students"] or 0),
        "placement_rate": _as_float(row["placement_rate"], precision=6),
        "avg_cgpa": _as_float(row["avg_cgpa"], precision=4),
        "avg_internships": _as_float(row["avg_internships"], precision=4),
    }