"""
src/ml/predict.py
-----------------
Inference wrapper — loaded once by FastAPI on startup.
Handles:
  - Single student prediction
  - Batch prediction (list of students)
  - SHAP explanation per prediction
"""

import json
import numpy as np
import polars as pl
import xgboost as xgb
import shap

from src.config.settings import MODEL_DIR, DATA_DIR


# ─────────────────────────────────────────────
# LOAD MODEL & METADATA (once on startup)
# ─────────────────────────────────────────────

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.ubj"))
    return model


def load_model_metadata():
    with open(MODEL_DIR / "model_metadata.json") as f:
        return json.load(f)


def load_feature_metadata():
    with open(DATA_DIR / "feature_metadata.json") as f:
        return json.load(f)


# These are module-level — loaded once when FastAPI starts
model           = load_model()
model_meta      = load_model_metadata()
feature_meta    = load_feature_metadata()
explainer       = shap.TreeExplainer(model)

FEATURE_COLS    = model_meta["feature_columns"]
THRESHOLD       = model_meta["threshold"]
BRANCH_ENCODING = feature_meta["branch_encoding"]
GENDER_ENCODING = feature_meta["gender_encoding"]


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# (mirrors pipeline.py — must stay in sync)
# ─────────────────────────────────────────────

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([

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

    return df


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    # Branch
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

    # Gender
    df = df.with_columns(
        pl.col("gender").str.strip_chars().replace(GENDER_ENCODING)
                        .cast(pl.Int32).alias("gender_encoded")
    )

    df = df.drop(["branch", "gender"])
    return df


def preprocess(raw_input: dict) -> np.ndarray:
    """
    Takes a raw student dict (from API request),
    runs feature engineering + encoding,
    returns numpy array ready for model.
    """
    df = pl.DataFrame([raw_input])
    df = engineer_features(df)
    df = encode_categoricals(df)

    # Select only the features the model was trained on, in the right order
    df = df.select(FEATURE_COLS)
    return df.to_numpy()


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_single(raw_input: dict) -> dict:
    """
    Predict placement for a single student.

    Args:
        raw_input: dict with all 29 input fields (no 'placed' column)

    Returns:
        {
            "placed": bool,
            "confidence": float,       # probability of being placed
            "threshold": float,
            "shap_explanation": {
                "feature_name": shap_value,  # positive = pushed toward placed
                ...
            }
        }
    """
    X = preprocess(raw_input)

    prob      = float(model.predict_proba(X)[0][1])
    placed    = prob >= THRESHOLD

    # SHAP explanation
    shap_vals = explainer.shap_values(X)[0]   # shape: (n_features,)
    shap_explanation = {
        feat: round(float(val), 4)
        for feat, val in zip(FEATURE_COLS, shap_vals)
    }

    # Sort by absolute impact for frontend display
    shap_explanation = dict(
        sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "placed":           bool(placed),
        "confidence":       round(prob, 4),
        "threshold":        THRESHOLD,
        "shap_explanation": shap_explanation,
    }


def predict_batch(raw_inputs: list[dict]) -> list[dict]:
    """
    Predict placement for a list of students.
    Used by the batch upload route.

    Returns list of prediction dicts, one per student.
    """
    results = []
    for raw_input in raw_inputs:
        try:
            result = predict_single(raw_input)
            result["error"] = None
        except Exception as e:
            result = {
                "placed":           None,
                "confidence":       None,
                "threshold":        THRESHOLD,
                "shap_explanation": {},
                "error":            str(e),
            }
        results.append(result)
    return results


def predict_batch_fast(raw_inputs: list[dict]) -> dict:
    """
    Faster batch prediction without per-row SHAP.
    Used for large CSV uploads where SHAP per row is too slow.
    Returns summary stats + per-row placed/confidence only.
    """
    df = pl.DataFrame(raw_inputs)
    df = engineer_features(df)
    df = encode_categoricals(df)
    X  = df.select(FEATURE_COLS).to_numpy()

    probs  = model.predict_proba(X)[:, 1]
    placed = (probs >= THRESHOLD).astype(int)

    return {
        "total":          len(raw_inputs),
        "placed_count":   int(placed.sum()),
        "not_placed_count": int((placed == 0).sum()),
        "placement_rate": round(float(placed.mean()), 4),
        "predictions": [
            {
                "placed":     bool(p),
                "confidence": round(float(c), 4),
            }
            for p, c in zip(placed, probs)
        ]
    }