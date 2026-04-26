"""Statistics routes backed by lazy parquet aggregation."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import duckdb
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


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_PATH = BASE_DIR / "data" / "students.parquet"

con = duckdb.connect()
parquet_path_str = str(PARQUET_PATH).replace("\\", "/")
con.execute(f"CREATE VIEW students AS SELECT * FROM read_parquet('{parquet_path_str}')")


@lru_cache(maxsize=1)
def _get_dataset_profile() -> dict[str, object]:
    desc = con.execute("DESCRIBE students").fetchall()
    actual_columns = [row[0] for row in desc]

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in actual_columns
    ]
    extra_columns = [
        column for column in actual_columns if column not in EXPECTED_COLUMNS
    ]

    overview = con.execute(f"SELECT COUNT(*) as row_count, SUM(CAST({TARGET} AS BIGINT)) as placed_count FROM students").fetchone()

    row_count = int(overview[0] or 0)
    placed_count = int(overview[1] or 0)
    not_placed_count = max(0, row_count - placed_count)

    branch_distribution = [
        {
            "branch": str(row[0]) if row[0] is not None else "Unknown",
            "count": int(row[1] or 0),
        }
        for row in con.execute("SELECT branch, COUNT(*) as count FROM students GROUP BY branch ORDER BY count DESC").fetchall()
    ]

    gender_distribution = [
        {
            "gender": str(row[0]) if row[0] is not None else "Unknown",
            "count": int(row[1] or 0),
        }
        for row in con.execute("SELECT gender, COUNT(*) as count FROM students GROUP BY gender ORDER BY count DESC").fetchall()
    ]

    heatmap_labels = [column for column in HEATMAP_COLUMNS if column in actual_columns]
    correlation_matrix: list[list[float]] = []
    if heatmap_labels:
        n = len(heatmap_labels)
        select_exprs = []
        for i in range(n):
            for j in range(n):
                c1 = heatmap_labels[i]
                c2 = heatmap_labels[j]
                select_exprs.append(f"corr({c1}, {c2})")
        corr_query = "SELECT " + ", ".join(select_exprs) + " FROM students"
        corr_result = con.execute(corr_query).fetchone()
        
        idx = 0
        for i in range(n):
            row_corrs = []
            for j in range(n):
                val = corr_result[idx]
                if i == j:
                    row_corrs.append(1.0)
                else:
                    row_corrs.append(_as_float(val, precision=6) if val is not None else 0.0)
                idx += 1
            correlation_matrix.append(row_corrs)

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


@router.get("/stats/overview")
async def get_stats_overview() -> dict[str, float | int]:
    """Return dataset-level overview statistics from parquet."""

    try:
        query = f"""
        SELECT 
            COUNT(*) as total_students, 
            AVG({TARGET}) as placement_rate, 
            AVG(cgpa) as avg_cgpa, 
            AVG(internships) as avg_internships 
        FROM students
        """
        result = con.execute(query).fetchone()

    except Exception as exc:
        LOGGER.exception(
            "Failed to compute /stats/overview from parquet: %s", PARQUET_PATH
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stats service is temporarily unavailable",
        ) from exc

    if not result or result[0] == 0:
        return {
            "total_students": 0,
            "placement_rate": 0.0,
            "avg_cgpa": 0.0,
            "avg_internships": 0.0,
        }

    return {
        "total_students": int(result[0] or 0),
        "placement_rate": _as_float(result[1], precision=6),
        "avg_cgpa": _as_float(result[2], precision=4),
        "avg_internships": _as_float(result[3], precision=4),
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
