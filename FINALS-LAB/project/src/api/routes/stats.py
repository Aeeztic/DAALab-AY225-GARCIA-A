"""Statistics routes backed by lazy parquet aggregation."""

from __future__ import annotations

import logging
from functools import lru_cache
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

EXPECTED_COLUMNS = [
    "age",
    "gender",
    "branch",
    "college_tier",
    "city_tier",
    "family_income",
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "attendance",
    "python_skill",
    "c++_skill",
    "java_skill",
    "ml_skill",
    "web_dev_skill",
    "communication_skill",
    "aptitude_score",
    "logical_reasoning",
    "problem_solving",
    "teamwork",
    "leadership",
    "time_management",
    "internships",
    "projects",
    "github_projects",
    "hackathons",
    "certifications",
    "coding_contest_rating",
    "placed",
]

HEATMAP_COLUMNS = [
    "age",
    "cgpa",
    "backlogs",
    "attendance",
    "tenth_percentage",
    "twelfth_percentage",
    "communication_skill",
    "aptitude_score",
    "logical_reasoning",
    "problem_solving",
    "internships",
    "projects",
    "placed",
]


def _as_float(value: float | None, *, precision: int = 6) -> float:
    if value is None:
        return 0.0
    return round(float(value), precision)


def _scan_students() -> pl.LazyFrame:
    return pl.scan_parquet(PARQUET_PATH)


def _frame_to_distribution(frame: pl.DataFrame, key: str) -> list[dict[str, int | str]]:
    return [
        {
            key: str(row[key]) if row[key] is not None else "Unknown",
            "count": int(row["count"] or 0),
        }
        for row in frame.to_dicts()
    ]


@lru_cache(maxsize=1)
def _get_dataset_profile() -> dict[str, object]:
    lf = _scan_students()
    schema = lf.collect_schema()
    actual_columns = schema.names()

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in actual_columns
    ]
    extra_columns = [
        column for column in actual_columns if column not in EXPECTED_COLUMNS
    ]

    overview = (
        lf.select(
            [
                pl.len().alias("row_count"),
                pl.col(TARGET).sum().cast(pl.Int64).alias("placed_count"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    row_count = int(overview["row_count"] or 0)
    placed_count = int(overview["placed_count"] or 0)
    not_placed_count = max(0, row_count - placed_count)

    branch_distribution = _frame_to_distribution(
        lf.group_by("branch")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(),
        "branch",
    )

    gender_distribution = _frame_to_distribution(
        lf.group_by("gender")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(),
        "gender",
    )

    heatmap_labels = [column for column in HEATMAP_COLUMNS if column in actual_columns]
    correlation_matrix: list[list[float]] = []
    if heatmap_labels:
        corr_df = lf.select(heatmap_labels).collect().corr()
        correlation_matrix = [
            [
                _as_float(corr_df.item(row_index, col_index), precision=6)
                for col_index in range(len(heatmap_labels))
            ]
            for row_index in range(len(heatmap_labels))
        ]

    return {
        "row_count": row_count,
        "column_count": len(actual_columns),
        "columns": actual_columns,
        "expected_column_count": len(EXPECTED_COLUMNS),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "placement_counts": {
            "placed": placed_count,
            "not_placed": not_placed_count,
        },
        "branch_distribution": branch_distribution,
        "gender_distribution": gender_distribution,
        "correlation": {
            "labels": heatmap_labels,
            "matrix": correlation_matrix,
        },
    }


# ✅ FIXED: correct path to parquet
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "students.parquet"


@router.get("/stats/overview")
async def get_stats_overview() -> dict[str, float | int]:
    """Return dataset-level overview statistics from parquet."""

    try:
        agg_lf = (
            _scan_students()
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


@router.get("/stats/dataset")
async def get_dataset_stats() -> dict[str, object]:
    """Return schema and full-dataset analytics computed from parquet."""

    try:
        return _get_dataset_profile()
    except Exception as exc:
        LOGGER.exception(
            "Failed to compute /stats/dataset from parquet: %s", PARQUET_PATH
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset analytics service is temporarily unavailable",
        ) from exc
