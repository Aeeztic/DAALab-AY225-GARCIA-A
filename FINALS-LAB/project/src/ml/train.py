"""
src/ml/train.py
---------------
Phase 2 — Model Training, Evaluation & Explanation
Steps:
  1. Load train/val/test parquet files
  2. Hyperparameter tuning (Optuna)
  3. Train final XGBoost model
  4. Find optimal threshold
  5. Evaluate on val + test
  6. SHAP feature importance
  7. Save model artifacts
"""

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

from src.config.settings import (
    DATA_DIR, MODEL_DIR, PLOTS_DIR,
    TARGET, RANDOM_SEED, N_TRIALS, CV_FOLDS
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_splits():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(DATA_DIR / "train.parquet")
    val   = pl.read_parquet(DATA_DIR / "val.parquet")
    test  = pl.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "feature_metadata.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]

    X_train = train.select(feature_cols).to_numpy()
    y_train = train[TARGET].to_numpy()

    X_val   = val.select(feature_cols).to_numpy()
    y_val   = val[TARGET].to_numpy()

    X_test  = test.select(feature_cols).to_numpy()
    y_test  = test[TARGET].to_numpy()

    print(f"✓ Loaded splits")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, meta


# ─────────────────────────────────────────────
# STEP 2 — HYPERPARAMETER TUNING (Optuna)
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train):
    print(f"\n── Tuning hyperparameters ({N_TRIALS} trials) ──")

    dtrain = xgb.DMatrix(X_train, label=y_train)

    def objective(trial):
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "seed":             RANDOM_SEED,
            "verbosity":        0,
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
        }

        cv_result = xgb.cv(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            nfold=CV_FOLDS,
            early_stopping_rounds=20,
            verbose_eval=False,
            seed=RANDOM_SEED,
        )
        return cv_result["test-auc-mean"].iloc[-1]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\n✓ Best params (AUC={study.best_value:.4f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    return best


# ─────────────────────────────────────────────
# STEP 3 — TRAIN FINAL MODEL
# ─────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, best_params):
    print("\n── Training final model ──")

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"✓ Trained — best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# STEP 4 — FIND OPTIMAL THRESHOLD
# ─────────────────────────────────────────────

def find_optimal_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.30, 0.75, 0.01)

    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n✓ Optimal threshold: {best_thresh:.2f}  (val F1={best_f1:.4f})")
    return float(best_thresh)


# ─────────────────────────────────────────────
# STEP 5 — EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, X, y, threshold, split_name="Test"):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)
    cm  = confusion_matrix(y, preds)

    print(f"\n── {split_name} Results ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y, preds, target_names=["Not Placed", "Placed"], zero_division=0))

    return {"accuracy": acc, "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist()}


# ─────────────────────────────────────────────
# STEP 6 — PLOTS
# ─────────────────────────────────────────────

def save_roc_curve(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#534AB7", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Placement Prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print("✓ Saved ROC curve")


def save_shap_plots(model, X_train, feature_cols):
    print("\n── Computing SHAP values ──")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title("SHAP Beeswarm — Feature Impact Direction")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("✓ Saved SHAP plots")

    shap_df = pl.DataFrame(shap_values, schema=feature_cols)
    shap_df.write_parquet(MODEL_DIR / "shap_values_train.parquet", compression="snappy")
    print("✓ Saved SHAP values parquet (for API/frontend)")

    return shap_values


# ─────────────────────────────────────────────
# STEP 7 — SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(model, threshold, feature_cols, val_metrics, test_metrics, best_params):
    with open(MODEL_DIR / "placement_model.pkl", "wb") as f:
        pickle.dump(model, f)

    model.save_model(str(MODEL_DIR / "model.ubj"))

    model_meta = {
        "feature_columns": feature_cols,
        "threshold":       threshold,
        "best_params":     best_params,
        "best_iteration":  int(model.best_iteration),
        "val_metrics": {
            "accuracy": round(val_metrics["accuracy"], 4),
            "f1":       round(val_metrics["f1"], 4),
            "roc_auc":  round(val_metrics["roc_auc"], 4),
        },
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "f1":       round(test_metrics["f1"], 4),
            "roc_auc":  round(test_metrics["roc_auc"], 4),
        },
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    print(f"\n✓ Saved artifacts to {MODEL_DIR.resolve()}/")
    print("  model.ubj")
    print("  placement_model.pkl")
    print("  model_metadata.json")
    print("  shap_values_train.parquet")
    print("  plots/roc_curve.png")
    print("  plots/shap_importance.png")
    print("  plots/shap_beeswarm.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_training():
    print("\n── Phase 2: Model Training Pipeline ──\n")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, meta = load_splits()

    best_params  = tune_hyperparameters(X_train, y_train)
    model        = train_model(X_train, y_train, X_val, y_val, best_params)
    threshold    = find_optimal_threshold(model, X_val, y_val)

    val_metrics  = evaluate(model, X_val,  y_val,  threshold, split_name="Validation")
    test_metrics = evaluate(model, X_test, y_test, threshold, split_name="Test")

    save_roc_curve(model, X_test, y_test)
    save_shap_plots(model, X_train, feature_cols)
    save_artifacts(model, threshold, feature_cols, val_metrics, test_metrics, best_params)

    print("\n── Done ✓ ──")
    print("Next: Phase 3 — build src/api/main.py\n")


if __name__ == "__main__":
    run_training()