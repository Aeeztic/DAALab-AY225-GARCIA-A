"""
src/data/pipeline.py
--------------------
Phase 1 — Data Foundation Pipeline
Steps:
  1. Load & validate cleaned CSV
  2. Feature engineering
  3. Encode categoricals
  4. Split and save to Parquet
  5. Save feature metadata JSON
"""

import json
import polars as pl
from src.config.settings import (
    RAW_CSV, DATA_DIR, TRAIN_RATIO, VAL_RATIO,
    RANDOM_SEED, SCHEMA, BRANCH_ORDER, GENDER_MAP, TARGET
)


# ─────────────────────────────────────────────
# STEP 1 — LOAD & VALIDATE
# ─────────────────────────────────────────────

def load_and_validate(path=RAW_CSV) -> pl.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(path)

    cast_exprs = [pl.col(col).cast(dtype) for col, dtype in SCHEMA.items() if col in df.columns]
    df = df.with_columns(cast_exprs)

    missing = set(SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.filter(pl.col(TARGET).is_not_null())

    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Placed: {df[TARGET].sum():,} | Not placed: {(df[TARGET] == 0).sum():,}")

    return df


# ─────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([

        # Composite: technical skill average (5 skills)
        (
            (pl.col("python_skill") + pl.col("c++_skill") + pl.col("java_skill") +
             pl.col("ml_skill") + pl.col("web_dev_skill")) / 5.0
        ).alias("avg_technical_skill").cast(pl.Float32),

        # Composite: soft skill average (4 skills)
        (
            (pl.col("communication_skill") + pl.col("teamwork") +
             pl.col("leadership") + pl.col("time_management")) / 4.0
        ).alias("avg_soft_skill").cast(pl.Float32),

        # Composite: problem-solving score
        (
            (pl.col("aptitude_score") + pl.col("logical_reasoning") +
             pl.col("problem_solving")) / 3.0
        ).alias("avg_problem_solving").cast(pl.Float32),

        # Activity score: sum of all extracurriculars
        (
            pl.col("internships") + pl.col("projects") + pl.col("github_projects") +
            pl.col("hackathons") + pl.col("certifications")
        ).alias("activity_score").cast(pl.Int32),

        # Academic growth: gap between 10th and 12th
        (pl.col("twelfth_percentage") - pl.col("tenth_percentage"))
        .alias("academic_growth").cast(pl.Float32),

        # Academic average: normalised mean of 10th, 12th, cgpa
        (
            (pl.col("tenth_percentage") / 100.0 +
             pl.col("twelfth_percentage") / 100.0 +
             pl.col("cgpa") / 10.0) / 3.0
        ).alias("academic_avg").cast(pl.Float32),

        # Flag: no backlogs
        (pl.col("backlogs") == 0).cast(pl.Int32).alias("clean_record"),

        # Flag: good attendance (>=75%)
        (pl.col("attendance") >= 75.0).cast(pl.Int32).alias("good_attendance"),

        # Flag: has internship experience
        (pl.col("internships") > 0).cast(pl.Int32).alias("has_internship"),

        # Flag: competitive coder (adjust threshold to match your data)
        (pl.col("coding_contest_rating") > 1000).cast(pl.Int32).alias("competitive_coder"),

        # Ratio: projects per year (age 18 as baseline)
        (
            pl.col("projects") /
            (pl.col("age") - 17).clip(lower_bound=1)
        ).alias("projects_per_year").cast(pl.Float32),

    ])

    return df


# ─────────────────────────────────────────────
# STEP 3 — ENCODE CATEGORICALS
# ─────────────────────────────────────────────

def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("branch").str.to_uppercase().str.strip_chars().alias("branch")
    )
    df = df.with_columns(
        pl.when(pl.col("branch").is_in(BRANCH_ORDER))
          .then(pl.col("branch"))
          .otherwise(pl.lit("OTHER"))
          .alias("branch")
    )

    branch_map = {b: i for i, b in enumerate(BRANCH_ORDER)}
    df = df.with_columns(
        pl.col("branch").replace(branch_map).cast(pl.Int32).alias("branch_encoded")
    )

    df = df.with_columns(
        pl.col("gender").str.strip_chars().replace(GENDER_MAP)
                        .cast(pl.Int32).alias("gender_encoded")
    )

    df = df.drop(["branch", "gender"])

    return df


# ─────────────────────────────────────────────
# STEP 4 — SPLIT & SAVE
# ─────────────────────────────────────────────

def split_and_save(df: pl.DataFrame):
    df = df.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)

    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = df[:n_train]
    val   = df[n_train : n_train + n_val]
    test  = df[n_train + n_val :]

    train.write_parquet(DATA_DIR / "train.parquet", compression="snappy")
    val.write_parquet(DATA_DIR   / "val.parquet",   compression="snappy")
    test.write_parquet(DATA_DIR  / "test.parquet",  compression="snappy")

    print(f"✓ Saved splits → train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")
    print(f"  Output: {DATA_DIR.resolve()}")

    return train, val, test


# ─────────────────────────────────────────────
# STEP 5 — SAVE FEATURE METADATA
# ─────────────────────────────────────────────

def save_feature_metadata(train: pl.DataFrame):
    feature_cols     = [c for c in train.columns if c != TARGET]
    numeric_features = [c for c in feature_cols if train[c].dtype in
                        (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    engineered_features = [
        "avg_technical_skill", "avg_soft_skill", "avg_problem_solving",
        "activity_score", "academic_growth", "academic_avg",
        "clean_record", "good_attendance", "has_internship",
        "competitive_coder", "projects_per_year",
    ]

    meta = {
        "target":               TARGET,
        "feature_columns":      feature_cols,
        "numeric_features":     numeric_features,
        "engineered_features":  engineered_features,
        "branch_encoding":      {b: i for i, b in enumerate(BRANCH_ORDER)},
        "gender_encoding":      GENDER_MAP,
        "train_stats": {
            col: {
                "mean": round(train[col].mean(), 4),
                "std":  round(train[col].std(),  4),
                "min":  round(train[col].min(),  4),
                "max":  round(train[col].max(),  4),
            }
            for col in numeric_features
        },
        "class_balance": {
            "placed":     int(train[TARGET].sum()),
            "not_placed": int((train[TARGET] == 0).sum()),
            "total":      len(train),
        }
    }

    meta_path = DATA_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Saved feature metadata → {meta_path.resolve()}")
    return meta


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline(csv_path=RAW_CSV):
    print("\n── Phase 1: Data Foundation Pipeline ──\n")

    df = load_and_validate(csv_path)
    df = engineer_features(df)
    df = encode_categoricals(df)

    train, val, test = split_and_save(df)
    meta = save_feature_metadata(train)

    print("\n── Done ✓ ──")
    print("Next: run src/ml/train.py\n")

    return train, val, test, meta


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else RAW_CSV
    run_pipeline(csv_path)