"""Student listing routes backed by lazy parquet queries."""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from typing import Any, Literal

import polars as pl
from fastapi import APIRouter, HTTPException, Query, status

from src.config.settings import DATA_DIR, TARGET

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["students"])


@lru_cache(maxsize=1)
def _load_branch_encoding() -> dict[str, int]:
    """Load and normalize branch encoding from feature metadata."""
    metadata_path = DATA_DIR / "feature_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    raw_mapping = metadata.get("branch_encoding")
    if not isinstance(raw_mapping, dict):
        raise RuntimeError("feature_metadata.json is missing a valid branch_encoding mapping")

    normalized_mapping: dict[str, int] = {}
    for label, encoded in raw_mapping.items():
        if not isinstance(label, str):
            continue
        try:
            normalized_mapping[label.strip().upper()] = int(encoded)
        except (TypeError, ValueError):
            continue

    if not normalized_mapping:
        raise RuntimeError("feature_metadata.json contains an empty branch_encoding mapping")

    return normalized_mapping


@router.get("/students")
async def get_students(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    cgpa_min: float | None = Query(None),
    cgpa_max: float | None = Query(None),
    internships_min: int | None = Query(None),
    internships_max: int | None = Query(None),
    branch: str | None = Query(None),
    placement_status: int | None = Query(None, ge=0, le=1),
    search: str | None = Query(None),
    sort_by: str | None = Query(None),
    sort_order: Literal["asc", "desc"] = Query("asc"),
) -> dict[str, Any]:
    """Return paginated students with optional filtering, search, and sorting."""
    parquet_path = str(DATA_DIR / "test.parquet")

    try:
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()
        schema_items = list(schema.items())
        available_columns = {name for name, _ in schema_items}
        branch_encoding = _load_branch_encoding()

        if cgpa_min is not None:
            lf = lf.filter(pl.col("cgpa") >= cgpa_min)
        if cgpa_max is not None:
            lf = lf.filter(pl.col("cgpa") <= cgpa_max)

        if internships_min is not None:
            lf = lf.filter(pl.col("internships") >= internships_min)
        if internships_max is not None:
            lf = lf.filter(pl.col("internships") <= internships_max)

        if branch:
            normalized_branch = branch.strip().upper()
            if "branch" in available_columns:
                lf = lf.filter(pl.col("branch") == normalized_branch)
            elif "branch_encoded" in available_columns:
                encoded_branch = branch_encoding.get(normalized_branch)
                if encoded_branch is None:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"branch must be one of: {', '.join(sorted(branch_encoding))}",
                    )
                lf = lf.filter(pl.col("branch_encoded") == encoded_branch)
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="branch filter is unavailable for this dataset",
                )

        if placement_status is not None:
            lf = lf.filter(pl.col(TARGET) == int(placement_status))

        if search:
            search_value = search.strip().lower()
            if search_value:
                text_columns = [name for name, dtype in schema_items if dtype == pl.Utf8]
                if text_columns:
                    search_expr = pl.any_horizontal(
                        [
                            pl.col(column)
                            .cast(pl.Utf8)
                            .str.to_lowercase()
                            .str.contains(search_value, literal=True)
                            for column in text_columns
                        ]
                    )
                    lf = lf.filter(search_expr)
                elif "branch_encoded" in available_columns:
                    matching_codes = [
                        encoded
                        for label, encoded in branch_encoding.items()
                        if search_value in label.lower()
                    ]
                    if matching_codes:
                        lf = lf.filter(pl.col("branch_encoded").is_in(matching_codes))

        if sort_by:
            if sort_by not in available_columns:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"sort_by must be one of: {', '.join(sorted(available_columns))}",
                )
            lf = lf.sort(by=sort_by, descending=(sort_order == "desc"), nulls_last=True)

        offset = (page - 1) * limit
        paged_lf = lf.slice(offset, limit)

        meta_lf = lf.select(
            [
                pl.lit("__meta__").alias("__row_type"),
                pl.len().cast(pl.Int64).alias("__total"),
            ]
        )

        data_lf = paged_lf.with_columns(
            [
                pl.lit("__data__").alias("__row_type"),
                pl.lit(None, dtype=pl.Int64).alias("__total"),
            ]
        )

        # Single execution point after all filters, sorting, and pagination are defined.
        result = pl.concat([meta_lf, data_lf], how="diagonal_relaxed").collect()

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to fetch /students from parquet: %s", parquet_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Students service is temporarily unavailable",
        ) from exc

    total = 0
    data: list[dict[str, Any]] = []
    for row in result.to_dicts():
        row_type = row.get("__row_type")
        if row_type == "__meta__":
            total = int(row.get("__total") or 0)
            continue

        row.pop("__row_type", None)
        row.pop("__total", None)
        data.append(row)

    total_pages = int(math.ceil(total / limit)) if total > 0 else 0

    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }