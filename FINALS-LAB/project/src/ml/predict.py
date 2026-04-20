"""
src/ml/predict.py
-----------------
Production inference module for placement prediction.

Responsibilities:
  - Load model and metadata once at process startup
  - Validate and normalize raw prediction input
  - Reproduce training-time feature engineering and encoding
  - Enforce strict model feature schema and feature order
  - Provide single and batch prediction APIs
  - Compute SHAP explanations only when requested
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import shap
import xgboost as xgb

from src.config.settings import DATA_DIR, MODEL_DIR, SCHEMA, TARGET


# -------------------------------------------------
# Exceptions
# -------------------------------------------------


class InferenceError(Exception):
    """Base exception for inference failures."""


class ArtifactLoadError(InferenceError):
    """Raised when model or metadata artifacts are unavailable/invalid."""


class InputValidationError(InferenceError):
    """Raised when incoming payload is missing or invalid."""


class PreprocessingError(InferenceError):
    """Raised when model features cannot be produced correctly."""


class PredictionError(InferenceError):
    """Raised when model inference or explanation fails."""


@dataclass(frozen=True)
class InferenceArtifacts:
    model: xgb.XGBClassifier
    model_meta: dict[str, Any]
    feature_meta: dict[str, Any]
    feature_cols: tuple[str, ...]
    threshold: float
    branch_encoding: dict[str, int]
    gender_encoding: dict[str, int]


# -------------------------------------------------
# Runtime constants
# -------------------------------------------------


_INT_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_FLOAT_DTYPES = {pl.Float32, pl.Float64}

_REQUIRED_RAW_FIELDS = tuple(col for col in SCHEMA if col != TARGET)

_EXPLAINER: shap.TreeExplainer | None = None
_EXPLAINER_LOCK = threading.Lock()


# -------------------------------------------------
# Artifact loading (once per process)
# -------------------------------------------------


def _load_model(path: Path) -> xgb.XGBClassifier:
    try:
        if not path.exists():
            raise ArtifactLoadError(f"Model artifact not found: {path}")

        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model
    except ArtifactLoadError:
        raise
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to load model from {path}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            raise ArtifactLoadError(f"Artifact file not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ArtifactLoadError(f"Invalid JSON payload in {path}: expected object")

        return payload
    except ArtifactLoadError:
        raise
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to load JSON artifact {path}: {exc}") from exc


def _validate_artifacts(
    model_meta: dict[str, Any],
    feature_meta: dict[str, Any],
) -> tuple[tuple[str, ...], float, dict[str, int], dict[str, int]]:
    feature_cols = model_meta.get("feature_columns")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ArtifactLoadError("model_metadata.json missing non-empty 'feature_columns'")
    if not all(isinstance(col, str) and col.strip() for col in feature_cols):
        raise ArtifactLoadError("model_metadata.json has invalid entries in 'feature_columns'")

    threshold_raw = model_meta.get("threshold")
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError("model_metadata.json has invalid 'threshold'") from exc

    if not (0.0 <= threshold <= 1.0):
        raise ArtifactLoadError("model_metadata.json threshold must be between 0 and 1")

    branch_encoding = feature_meta.get("branch_encoding")
    if not isinstance(branch_encoding, dict) or not branch_encoding:
        raise ArtifactLoadError("feature_metadata.json missing non-empty 'branch_encoding'")
    if "OTHER" not in branch_encoding:
        raise ArtifactLoadError("feature_metadata.json branch_encoding must contain 'OTHER'")

    gender_encoding = feature_meta.get("gender_encoding")
    if not isinstance(gender_encoding, dict) or not gender_encoding:
        raise ArtifactLoadError("feature_metadata.json missing non-empty 'gender_encoding'")

    feature_meta_cols = feature_meta.get("feature_columns")
    if not isinstance(feature_meta_cols, list) or not feature_meta_cols:
        raise ArtifactLoadError("feature_metadata.json missing non-empty 'feature_columns'")

    if list(feature_cols) != list(feature_meta_cols):
        raise ArtifactLoadError(
            "Feature column order mismatch between model_metadata.json and feature_metadata.json"
        )

    try:
        branch_encoding_int = {str(key): int(value) for key, value in branch_encoding.items()}
        gender_encoding_int = {str(key): int(value) for key, value in gender_encoding.items()}
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError("Categorical encodings must map labels to integers") from exc

    return tuple(feature_cols), threshold, branch_encoding_int, gender_encoding_int


def _load_artifacts() -> InferenceArtifacts:
    model = _load_model(MODEL_DIR / "model.ubj")
    model_meta = _load_json(MODEL_DIR / "model_metadata.json")
    feature_meta = _load_json(DATA_DIR / "feature_metadata.json")

    feature_cols, threshold, branch_encoding, gender_encoding = _validate_artifacts(
        model_meta,
        feature_meta,
    )

    return InferenceArtifacts(
        model=model,
        model_meta=model_meta,
        feature_meta=feature_meta,
        feature_cols=feature_cols,
        threshold=threshold,
        branch_encoding=branch_encoding,
        gender_encoding=gender_encoding,
    )


ARTIFACTS = _load_artifacts()

# Compatibility aliases for future API route imports.
FEATURE_COLS = ARTIFACTS.feature_cols
THRESHOLD = ARTIFACTS.threshold
BRANCH_ENCODING = ARTIFACTS.branch_encoding
GENDER_ENCODING = ARTIFACTS.gender_encoding


def load_model() -> xgb.XGBClassifier:
    """Compatibility helper that returns the already loaded model."""
    return ARTIFACTS.model


def load_model_metadata() -> dict[str, Any]:
    """Compatibility helper that returns loaded model metadata."""
    return ARTIFACTS.model_meta


def load_feature_metadata() -> dict[str, Any]:
    """Compatibility helper that returns loaded feature metadata."""
    return ARTIFACTS.feature_meta


def get_runtime_status() -> dict[str, Any]:
    """Expose lightweight readiness information for startup/health checks."""
    return {
        "model_loaded": True,
        "feature_count": len(FEATURE_COLS),
        "threshold": THRESHOLD,
    }


# -------------------------------------------------
# Input validation and normalization
# -------------------------------------------------


def _coerce_int(field_name: str, value: Any) -> int:
    if value is None:
        raise InputValidationError(f"Field '{field_name}' cannot be null")
    if isinstance(value, bool):
        raise InputValidationError(f"Field '{field_name}' must be an integer")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise InputValidationError(f"Field '{field_name}' must be an integer")
        return int(numeric)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise InputValidationError(f"Field '{field_name}' cannot be empty")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise InputValidationError(f"Field '{field_name}' must be an integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise InputValidationError(f"Field '{field_name}' must be an integer")
        return int(numeric)

    raise InputValidationError(f"Field '{field_name}' must be an integer")


def _coerce_float(field_name: str, value: Any) -> float:
    if value is None:
        raise InputValidationError(f"Field '{field_name}' cannot be null")
    if isinstance(value, bool):
        raise InputValidationError(f"Field '{field_name}' must be numeric")

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise InputValidationError(f"Field '{field_name}' cannot be empty")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise InputValidationError(f"Field '{field_name}' must be numeric") from exc
    else:
        raise InputValidationError(f"Field '{field_name}' must be numeric")

    if not np.isfinite(numeric):
        raise InputValidationError(f"Field '{field_name}' must be finite")

    return numeric


def _normalize_branch(value: Any) -> str:
    if value is None:
        raise InputValidationError("Field 'branch' cannot be null")

    branch = str(value).strip().upper()
    if not branch:
        raise InputValidationError("Field 'branch' cannot be empty")

    return branch if branch in BRANCH_ENCODING else "OTHER"


def _normalize_gender(value: Any) -> str:
    if value is None:
        raise InputValidationError("Field 'gender' cannot be null")

    gender_raw = str(value).strip()
    if not gender_raw:
        raise InputValidationError("Field 'gender' cannot be empty")

    lookup = {label.casefold(): label for label in GENDER_ENCODING}
    canonical = lookup.get(gender_raw.casefold())
    if canonical is None:
        allowed = ", ".join(sorted(GENDER_ENCODING.keys()))
        raise InputValidationError(
            f"Invalid gender '{gender_raw}'. Allowed values: {allowed}"
        )

    return canonical


def _validate_and_normalize_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_input, dict):
        raise InputValidationError("Input must be a JSON object")

    required = set(_REQUIRED_RAW_FIELDS)
    provided = set(raw_input.keys())

    missing = sorted(required - provided)
    extra = sorted(provided - required)

    errors: list[str] = []
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected fields: {', '.join(extra)}")

    if errors:
        raise InputValidationError("; ".join(errors))

    normalized: dict[str, Any] = {}

    for field_name in _REQUIRED_RAW_FIELDS:
        value = raw_input[field_name]

        if field_name == "branch":
            normalized[field_name] = _normalize_branch(value)
            continue

        if field_name == "gender":
            normalized[field_name] = _normalize_gender(value)
            continue

        dtype = SCHEMA[field_name]
        if dtype in _INT_DTYPES:
            normalized[field_name] = _coerce_int(field_name, value)
        elif dtype in _FLOAT_DTYPES:
            normalized[field_name] = _coerce_float(field_name, value)
        else:
            if value is None:
                raise InputValidationError(f"Field '{field_name}' cannot be null")
            normalized[field_name] = value

    return normalized


# -------------------------------------------------
# Feature engineering and schema enforcement
# -------------------------------------------------


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering logic kept in parity with src/data/pipeline.py."""
    return df.with_columns([
        (
            (pl.col("python_skill") + pl.col("c++_skill") + pl.col("java_skill") +
             pl.col("ml_skill") + pl.col("web_dev_skill")) / 5.0
        ).alias("avg_technical_skill").cast(pl.Float32),

        (
            (pl.col("communication_skill") + pl.col("teamwork") +
             pl.col("leadership") + pl.col("time_management")) / 4.0
        ).alias("avg_soft_skill").cast(pl.Float32),

        (
            (pl.col("aptitude_score") + pl.col("logical_reasoning") +
             pl.col("problem_solving")) / 3.0
        ).alias("avg_problem_solving").cast(pl.Float32),

        (
            pl.col("internships") + pl.col("projects") + pl.col("github_projects") +
            pl.col("hackathons") + pl.col("certifications")
        ).alias("activity_score").cast(pl.Int32),

        (pl.col("twelfth_percentage") - pl.col("tenth_percentage"))
        .alias("academic_growth").cast(pl.Float32),

        (
            (pl.col("tenth_percentage") / 100.0 +
             pl.col("twelfth_percentage") / 100.0 +
             pl.col("cgpa") / 10.0) / 3.0
        ).alias("academic_avg").cast(pl.Float32),

        (pl.col("backlogs") == 0).cast(pl.Int32).alias("clean_record"),
        (pl.col("attendance") >= 75.0).cast(pl.Int32).alias("good_attendance"),
        (pl.col("internships") > 0).cast(pl.Int32).alias("has_internship"),
        (pl.col("coding_contest_rating") > 1000).cast(pl.Int32).alias("competitive_coder"),

        (
            pl.col("projects") /
            (pl.col("age") - 17).clip(lower_bound=1)
        ).alias("projects_per_year").cast(pl.Float32),
    ])


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("branch").str.to_uppercase().str.strip_chars().alias("branch")
    )
    df = df.with_columns(
        pl.when(pl.col("branch").is_in(list(BRANCH_ENCODING.keys())))
        .then(pl.col("branch"))
        .otherwise(pl.lit("OTHER"))
        .alias("branch")
    )
    df = df.with_columns(
        pl.col("branch").replace(BRANCH_ENCODING).cast(pl.Int32).alias("branch_encoded")
    )

    df = df.with_columns(
        pl.col("gender").str.strip_chars().replace(GENDER_ENCODING)
        .cast(pl.Int32).alias("gender_encoded")
    )

    return df.drop(["branch", "gender"])


def _enforce_feature_schema(df: pl.DataFrame) -> pl.DataFrame:
    expected = list(FEATURE_COLS)
    produced = df.columns

    missing = sorted(set(expected) - set(produced))
    if missing:
        raise PreprocessingError(
            f"Preprocessing missing required model features: {', '.join(missing)}"
        )

    unexpected = sorted(set(produced) - set(expected))
    if unexpected:
        raise PreprocessingError(
            f"Preprocessing produced unexpected features: {', '.join(unexpected)}"
        )

    ordered = df.select(expected)
    if ordered.columns != expected:
        raise PreprocessingError("Feature order mismatch against model metadata")

    return ordered


def _preprocess_records(records: list[dict[str, Any]]) -> np.ndarray:
    try:
        df = pl.DataFrame(records)
        df = engineer_features(df)
        df = encode_categoricals(df)
        df = _enforce_feature_schema(df)
        return df.to_numpy()
    except InferenceError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Failed to preprocess records: {exc}") from exc


def preprocess(raw_input: dict[str, Any]) -> np.ndarray:
    """
    Validate one student payload and return a model-ready numpy matrix.

    Raises:
      - InputValidationError
      - PreprocessingError
    """
    normalized = _validate_and_normalize_input(raw_input)
    return _preprocess_records([normalized])


# -------------------------------------------------
# Prediction and optional SHAP explanation
# -------------------------------------------------


def _get_shap_explainer() -> shap.TreeExplainer:
    global _EXPLAINER

    if _EXPLAINER is None:
        with _EXPLAINER_LOCK:
            if _EXPLAINER is None:
                _EXPLAINER = shap.TreeExplainer(ARTIFACTS.model)

    return _EXPLAINER


def _compute_shap_explanation(X: np.ndarray) -> dict[str, float]:
    try:
        explainer = _get_shap_explainer()
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 2:
            row_values = shap_arr[0]
        elif shap_arr.ndim == 1:
            row_values = shap_arr
        else:
            raise PredictionError("Unexpected SHAP output shape")

        if len(row_values) != len(FEATURE_COLS):
            raise PredictionError("SHAP output length does not match feature count")

        shap_explanation = {
            feature: round(float(value), 4)
            for feature, value in zip(FEATURE_COLS, row_values)
        }

        return dict(
            sorted(shap_explanation.items(), key=lambda item: abs(item[1]), reverse=True)
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise PredictionError(f"Failed to compute SHAP explanation: {exc}") from exc


def predict_single(
    raw_input: dict[str, Any],
    include_explanation: bool = False,
) -> dict[str, Any]:
    """
    Predict one student record.

    Returns:
      {
        "probability": float,
        "prediction": 0 | 1,
        "shap_explanation": {...}   # only when include_explanation=True
      }

    Raises:
      - InputValidationError
      - PreprocessingError
      - PredictionError
    """
    try:
        X = preprocess(raw_input)
        probability = float(ARTIFACTS.model.predict_proba(X)[0][1])
        prediction = int(probability >= THRESHOLD)

        output: dict[str, Any] = {
            "probability": round(probability, 6),
            "prediction": prediction,
        }

        if include_explanation:
            output["shap_explanation"] = _compute_shap_explanation(X)

        return output
    except InferenceError:
        raise
    except Exception as exc:
        raise PredictionError(f"Failed to run single prediction: {exc}") from exc


def predict_batch(
    raw_inputs: list[dict[str, Any]],
    include_explanation: bool = False,
) -> list[dict[str, Any]]:
    """
    Row-wise batch prediction with per-row error isolation.

    Invalid rows do not fail the entire batch.
    """
    if not isinstance(raw_inputs, list):
        raise InputValidationError("Batch input must be a list of student objects")

    results: list[dict[str, Any]] = []

    for index, raw_input in enumerate(raw_inputs):
        try:
            if not isinstance(raw_input, dict):
                raise InputValidationError("Each batch row must be a JSON object")

            row_result = predict_single(raw_input, include_explanation=include_explanation)
            row_result["row_index"] = index
            row_result["error"] = None
        except InferenceError as exc:
            row_result = {
                "row_index": index,
                "probability": None,
                "prediction": None,
                "error": str(exc),
            }
            if include_explanation:
                row_result["shap_explanation"] = {}

        results.append(row_result)

    return results


def predict_batch_fast(raw_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Fast vectorized batch prediction without SHAP.

    Invalid rows are reported individually while valid rows are inferred
    in one vectorized model call.
    """
    if not isinstance(raw_inputs, list):
        raise InputValidationError("Batch input must be a list of student objects")

    total_rows = len(raw_inputs)
    if total_rows == 0:
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "placed_count": 0,
            "not_placed_count": 0,
            "placement_rate": 0.0,
            "predictions": [],
        }

    predictions: list[dict[str, Any] | None] = [None] * total_rows
    valid_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []

    for index, raw_input in enumerate(raw_inputs):
        try:
            if not isinstance(raw_input, dict):
                raise InputValidationError("Each batch row must be a JSON object")

            valid_rows.append(_validate_and_normalize_input(raw_input))
            valid_indices.append(index)
        except InferenceError as exc:
            predictions[index] = {
                "row_index": index,
                "probability": None,
                "prediction": None,
                "error": str(exc),
            }

    if valid_rows:
        try:
            X = _preprocess_records(valid_rows)
            probs = ARTIFACTS.model.predict_proba(X)[:, 1]
            labels = (probs >= THRESHOLD).astype(int)

            for local_idx, row_idx in enumerate(valid_indices):
                predictions[row_idx] = {
                    "row_index": row_idx,
                    "probability": round(float(probs[local_idx]), 6),
                    "prediction": int(labels[local_idx]),
                    "error": None,
                }
        except InferenceError as exc:
            for row_idx in valid_indices:
                predictions[row_idx] = {
                    "row_index": row_idx,
                    "probability": None,
                    "prediction": None,
                    "error": str(exc),
                }
        except Exception as exc:
            msg = f"Failed to run fast batch inference: {exc}"
            for row_idx in valid_indices:
                predictions[row_idx] = {
                    "row_index": row_idx,
                    "probability": None,
                    "prediction": None,
                    "error": msg,
                }

    # Safety fill in case any row slot is still empty.
    for index, payload in enumerate(predictions):
        if payload is None:
            predictions[index] = {
                "row_index": index,
                "probability": None,
                "prediction": None,
                "error": "Unknown batch prediction failure",
            }

    finalized = [item for item in predictions if item is not None]
    success_rows = [item for item in finalized if item["error"] is None]

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
        "predictions": finalized,
    }
