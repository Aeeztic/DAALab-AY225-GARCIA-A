"""Prediction routes for single-student inference."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.request_models import PredictRequest, PredictResponse
from src.ml.predict import (
    ArtifactLoadError,
    InferenceError,
    InputValidationError,
    PredictionError,
    PreprocessingError,
    predict_single,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


@router.post(
	"/predict",
	response_model=PredictResponse,
	response_model_exclude_none=True,
)
async def predict(payload: PredictRequest) -> PredictResponse:
	"""Run placement prediction for a single student payload."""
	try:
		model_input = payload.to_model_input()
		result = predict_single(
			raw_input=model_input,
			include_explanation=payload.include_explanation,
		)
		return PredictResponse(**result)

	except InputValidationError as exc:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail=str(exc),
		) from exc

	except PreprocessingError as exc:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail=str(exc),
		) from exc

	except ArtifactLoadError as exc:
		LOGGER.exception("Inference artifacts unavailable during /predict")
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Prediction service is temporarily unavailable",
		) from exc

	except PredictionError as exc:
		LOGGER.exception("Prediction runtime failure during /predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Prediction failed due to an internal error",
		) from exc

	except InferenceError as exc:
		LOGGER.exception("Unhandled inference error during /predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Prediction failed due to an internal error",
		) from exc

	except Exception as exc:
		LOGGER.exception("Unexpected failure during /predict")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Prediction failed due to an internal error",
		) from exc
