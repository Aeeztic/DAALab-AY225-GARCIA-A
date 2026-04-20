"""Batch prediction routes for CSV uploads."""

from __future__ import annotations

import io
import logging
from typing import Any

import polars as pl
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from src.api.schemas.request_models import PredictRequest
from src.ml.predict import (
	ArtifactLoadError,
	InferenceError,
	InputValidationError,
	PredictionError,
	PreprocessingError,
	predict_batch_fast,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["batch"])

MAX_BATCH_ROWS = 1000
CSV_CONTENT_TYPES = {
	"text/csv",
	"application/csv",
	"application/vnd.ms-excel",
}


def _format_row_validation_error(exc: ValidationError) -> str:
	"""Convert pydantic validation details into a compact row error string."""
	issues: list[str] = []
	for error in exc.errors():
		location = ".".join(str(part) for part in error.get("loc", []))
		message = error.get("msg", "Invalid value")
		issues.append(f"{location}: {message}" if location else message)
	return "; ".join(issues) if issues else "Invalid row payload"


def _parse_csv_records(payload: bytes) -> list[dict[str, Any]]:
	"""Parse CSV bytes safely into list-of-dict records."""
	if not payload or not payload.strip():
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Uploaded file is empty",
		)

	try:
		dataframe = pl.read_csv(io.BytesIO(payload), infer_schema_length=200)
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail="Invalid CSV file",
		) from exc

	if dataframe.height == 0:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="CSV file contains no data rows",
		)

	if dataframe.height > MAX_BATCH_ROWS:
		raise HTTPException(
			status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
			detail=f"CSV row limit exceeded (max {MAX_BATCH_ROWS})",
		)

	return dataframe.to_dicts()


@router.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)) -> dict[str, Any]:
	"""Run batch prediction from an uploaded CSV file."""
	if not file.filename:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="CSV file is required",
		)

	if not file.filename.lower().endswith(".csv"):
		raise HTTPException(
			status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
			detail="Only .csv files are supported",
		)

	normalized_content_type = ""
	if file.content_type:
		normalized_content_type = file.content_type.split(";", 1)[0].strip().lower()

	if normalized_content_type and normalized_content_type not in CSV_CONTENT_TYPES:
		raise HTTPException(
			status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
			detail=(
				"Unsupported file content type. "
				f"Expected one of: {', '.join(sorted(CSV_CONTENT_TYPES))}"
			),
		)

	try:
		raw_bytes = await file.read()
		records = _parse_csv_records(raw_bytes)
		total_rows = len(records)

		invalid_predictions: dict[int, dict[str, Any]] = {}
		valid_rows: list[dict[str, Any]] = []
		valid_row_indices: list[int] = []

		# Validate each row against PredictRequest schema while preserving index order.
		for row_index, row in enumerate(records):
			try:
				validated = PredictRequest.model_validate(row)
				valid_rows.append(validated.to_model_input())
				valid_row_indices.append(row_index)
			except ValidationError as exc:
				invalid_predictions[row_index] = {
					"row_index": row_index,
					"probability": None,
					"prediction": None,
					"error": _format_row_validation_error(exc),
				}

		valid_batch_predictions: list[dict[str, Any]] = []
		if valid_rows:
			batch_result = predict_batch_fast(valid_rows)
			for item in batch_result.get("predictions", []):
				remapped = dict(item)
				internal_index = int(remapped.get("row_index", -1))
				if internal_index < 0 or internal_index >= len(valid_row_indices):
					continue
				remapped["row_index"] = valid_row_indices[internal_index]
				valid_batch_predictions.append(remapped)

		merged_predictions: list[dict[str, Any] | None] = [None] * total_rows
		for row_index, payload in invalid_predictions.items():
			merged_predictions[row_index] = payload

		for payload in valid_batch_predictions:
			row_index = int(payload["row_index"])
			if 0 <= row_index < total_rows:
				merged_predictions[row_index] = payload

		# Safety fill in case any row was not materialized.
		for row_index, payload in enumerate(merged_predictions):
			if payload is None:
				merged_predictions[row_index] = {
					"row_index": row_index,
					"probability": None,
					"prediction": None,
					"error": "Unknown batch prediction failure",
				}

		predictions = [item for item in merged_predictions if item is not None]
		success_rows = [item for item in predictions if item.get("error") is None]

		processed = len(success_rows)
		failed = total_rows - processed
		placed_count = int(sum(int(item["prediction"]) for item in success_rows))
		not_placed_count = int(processed - placed_count)
		placement_rate = round((placed_count / processed), 6) if processed else 0.0

		return {
			"total": total_rows,
			"processed": processed,
			"failed": failed,
			"placed_count": placed_count,
			"not_placed_count": not_placed_count,
			"placement_rate": placement_rate,
			"predictions": predictions,
		}

	except HTTPException:
		raise
	except (InputValidationError, PreprocessingError) as exc:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail=str(exc),
		) from exc
	except ArtifactLoadError as exc:
		LOGGER.exception("Inference artifacts unavailable during /batch-predict")
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Prediction service is temporarily unavailable",
		) from exc
	except PredictionError as exc:
		LOGGER.exception("Prediction runtime failure during /batch-predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Batch prediction failed due to an internal error",
		) from exc
	except InferenceError as exc:
		LOGGER.exception("Unhandled inference error during /batch-predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Batch prediction failed due to an internal error",
		) from exc
	except Exception as exc:
		LOGGER.exception("Unexpected failure during /batch-predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Batch prediction failed due to an internal error",
		) from exc
	finally:
		await file.close()
