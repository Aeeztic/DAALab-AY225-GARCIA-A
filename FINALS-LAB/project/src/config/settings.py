"""
src/config/settings.py
----------------------
All constants, paths, and schema definitions.
Imported by pipeline.py, train.py, predict.py, and FastAPI.
"""

import os
from pathlib import Path
import polars as pl


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

# Project root (portable across machines).
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_CSV    = BASE_DIR / "data" / "raw" / "students.csv"
DATA_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "src" / "ml" / "model"
PLOTS_DIR  = BASE_DIR / "src" / "ml" / "model" / "plots"


# ─────────────────────────────────────────────
# PIPELINE SETTINGS
# ─────────────────────────────────────────────

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15 (remainder)


# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────

TARGET   = "placed"
N_TRIALS = 50     # Optuna tuning trials — raise to 100 for better results
CV_FOLDS = 5


# ─────────────────────────────────────────────
# DATASET SCHEMA
# ─────────────────────────────────────────────

SCHEMA = {
    "age":                   pl.Int32,
    "cgpa":                  pl.Float32,
    "backlogs":              pl.Int32,
    "attendance":            pl.Float32,
    "tenth_percentage":      pl.Float32,
    "twelfth_percentage":    pl.Float32,
    "branch":                pl.Utf8,
    "college_tier":          pl.Int32,
    "python_skill":          pl.Int32,
    "c++_skill":             pl.Int32,
    "java_skill":            pl.Int32,
    "ml_skill":              pl.Int32,
    "web_dev_skill":         pl.Int32,
    "communication_skill":   pl.Int32,
    "aptitude_score":        pl.Float32,
    "logical_reasoning":     pl.Float32,
    "internships":           pl.Int32,
    "projects":              pl.Int32,
    "github_projects":       pl.Int32,
    "hackathons":            pl.Int32,
    "certifications":        pl.Int32,
    "coding_contest_rating": pl.Float32,
    "teamwork":              pl.Int32,
    "leadership":            pl.Int32,
    "problem_solving":       pl.Int32,
    "time_management":       pl.Int32,
    "gender":                pl.Utf8,
    "city_tier":             pl.Int32,
    "family_income":         pl.Float32,
    "placed":                pl.Int32,
}


# ─────────────────────────────────────────────
# CATEGORICAL ENCODINGS
# ─────────────────────────────────────────────

# Update this list to match your actual branch names exactly
BRANCH_ORDER = [
    "CSE", "IT", "ECE", "EEE", "MECH", "CIVIL", "CHEM", "BIO", "OTHER"
]

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}