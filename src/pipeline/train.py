# -*- coding: utf-8 -*-
"""src.pipeline.train

Stage 2: Train 1-3 models + compare + select best

[OK] Split train / val / test
[OK] Pipeline (num/cat/text) + GridSearchCV
[OK] Compare on VALIDATION
[OK] Refit best on train+val, final metrics on TEST
[OK] Save best_model.pkl + train_meta.json + run snapshot runs/run_...

IMPORTANT FIX:
- No nested functions and no lambdas in pipeline (pickle/joblib safe)
- No named_transformers_/transformers_ access before fit
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    mean_absolute_error, r2_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.config.dataset_config import load_dataset_config
from src.config.model_config import load_model_config

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

import sys

def _safe_print(msg: str) -> None:
    """Print sans crash sur Windows (console cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = msg.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
        print(safe)



DATA_PATH = Path("data") / "processed" / "dataset.csv"
SPLIT_DIR = Path("data") / "processed" / "splits"
MODEL_PATH = Path("models") / "best_model.pkl"
META_PATH = Path("reports") / "train_meta.json"
RUNS_DIR = Path("runs")


# -------------------------------------------------------------------
# Pickle-safe TOP-LEVEL functions (NO lambdas, NO nested functions)
# -------------------------------------------------------------------
def join_text_cols(X):
    """Join multiple text columns row-wise -> array of strings (pickle-safe)."""
    if hasattr(X, "iloc"):
        vals = X.astype(str).fillna("").values
    else:
        vals = np.asarray(X).astype(str)
    joined = [" ".join(row) for row in vals]
    return np.array(joined)


def to_dense_matrix(X):
    """Convert sparse matrix to dense if needed (pickle-safe)."""
    return X.toarray() if hasattr(X, "toarray") else X


# -------------------------------------------------------------------
# Version-safe Bagging constructors (sklearn changed base_estimator -> estimator)
# -------------------------------------------------------------------
def _make_bagging_classifier():
    base = DecisionTreeClassifier(random_state=42)
    try:
        return BaggingClassifier(estimator=base, random_state=42)
    except TypeError:  # older sklearn
        return BaggingClassifier(base_estimator=base, random_state=42)


def _make_bagging_regressor():
    base = DecisionTreeRegressor(random_state=42)
    try:
        return BaggingRegressor(estimator=base, random_state=42)
    except TypeError:  # older sklearn
        return BaggingRegressor(base_estimator=base, random_state=42)


# -------------------------------------------------------------------
# Models that require dense matrix (trees, KNN, NB, MLP, etc.)
# -------------------------------------------------------------------
_DENSE_MODEL_KEYS = {
    # classification
    "random_forest", "knn", "decision_tree", "gradient_boosting", "bagging",
    "naive_bayes", "mlp", "svc",
    # regression
    "linear_regression", "lasso", "poly_ridge",
    "random_forest", "knn_reg", "decision_tree", "gradient_boosting",
    "adaboost", "bagging", "mlp",
}


def _needs_dense(model_key: str) -> bool:
    return model_key in _DENSE_MODEL_KEYS


def _build_model_pipeline(model_key: str, estimator, preprocess) -> Pipeline:
    """Build the full sklearn Pipeline for a given model_key (keep it pickle-safe)."""
    steps = [("prep", preprocess)]
    if _needs_dense(model_key):
        steps.append(("dense", _dense_step()))
    # Polynomial regression: only makes sense on dense numeric features
    if model_key == "poly_ridge":
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


# ----------------------------
# Helpers
# ----------------------------
def _now_run_id() -> str:
    from datetime import datetime
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_ohe():
    # sklearn compatibility: sparse_output vs sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _infer_cols(df: pd.DataFrame, x_cols: List[str], text_cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in x_cols if c not in num_cols and c not in text_cols]
    return num_cols, cat_cols


def _build_text_pipeline(text_cols: List[str]) -> Optional[Pipeline]:
    if not text_cols:
        return None
    return Pipeline(steps=[
        ("join", FunctionTransformer(join_text_cols, validate=False)),
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True
        )),
    ])


def _build_preprocess(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _ensure_ohe()),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    txt_pipe = _build_text_pipeline(text_cols)
    if txt_pipe is not None:
        transformers.append(("text", txt_pipe, text_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def _dense_step():
    return FunctionTransformer(to_dense_matrix, accept_sparse=True)


def _score_classification(y_true, y_pred, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # roc_auc only when binary + proba exists
    try:
        uniq = np.unique(y_true)
        if y_proba is not None and len(uniq) == 2:
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    except Exception:
        pass
    return out


def _score_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def _save_splits(X_train, X_val, X_test, y_train, y_val, y_test) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(SPLIT_DIR / "X_train.csv", index=False)
    X_val.to_csv(SPLIT_DIR / "X_val.csv", index=False)
    X_test.to_csv(SPLIT_DIR / "X_test.csv", index=False)

    pd.DataFrame({"y": y_train}).to_csv(SPLIT_DIR / "y_train.csv", index=False)
    pd.DataFrame({"y": y_val}).to_csv(SPLIT_DIR / "y_val.csv", index=False)
    pd.DataFrame({"y": y_test}).to_csv(SPLIT_DIR / "y_test.csv", index=False)



# ----------------------------
# Robust split helpers (avoid stratify crash / auto task switch)
# ----------------------------
def _maybe_stratify(y_arr: np.ndarray) -> Optional[np.ndarray]:
    """Return y_arr for stratify only if each class has >=2 samples; else None."""
    try:
        if y_arr is None or len(y_arr) < 4:
            return None
        if not np.issubdtype(np.asarray(y_arr).dtype, np.integer):
            return None
        counts = np.bincount(np.asarray(y_arr, dtype=int))
        if counts.size == 0:
            return None
        return y_arr if int(counts.min()) >= 2 else None
    except Exception:
        return None


def _looks_continuous_numeric(y_raw: pd.Series, unique_threshold: int = 20, ratio_threshold: float = 0.2) -> bool:
    """Heuristic: numeric Y + many unique values => likely regression."""
    try:
        if not pd.api.types.is_numeric_dtype(y_raw):
            return False
        y_nn = y_raw.dropna()
        n = int(y_nn.shape[0])
        if n == 0:
            return False
        nunique = int(y_nn.nunique(dropna=True))
        return (nunique > unique_threshold) and (nunique / max(1, n) > ratio_threshold)
    except Exception:
        return False

# ----------------------------
# Models + grids
# ----------------------------
def _get_models(task: str) -> Dict[str, Any]:
    if task == "classification":
        models: Dict[str, Any] = {
            "logreg": LogisticRegression(max_iter=2000, solver="saga"),
            "linear_svc": LinearSVC(),
            "svc": SVC(),  # kernel SVM (peut être lent selon dataset)
            "sgd": SGDClassifier(random_state=42),
            "naive_bayes": GaussianNB(),
            "random_forest": RandomForestClassifier(random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "bagging": _make_bagging_classifier(),
            "knn": KNeighborsClassifier(),
            "mlp": MLPClassifier(max_iter=400, random_state=42),
        }
        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1
            )
        return models

    # regression
    models: Dict[str, Any] = {
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=5000),
        "linear_regression": LinearRegression(),
        "poly_ridge": Ridge(),  # PolynomialFeatures step ajouté dans le pipeline
        "random_forest": RandomForestRegressor(random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "knn_reg": KNeighborsRegressor(),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "adaboost": AdaBoostRegressor(random_state=42),
        "bagging": _make_bagging_regressor(),
        "mlp": MLPRegressor(max_iter=500, random_state=42),
    }

    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(random_state=42)

    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(random_state=42, n_jobs=1)

    return models


def _get_param_grid(model_key: str, task: str) -> Dict[str, List[Any]]:
    if task == "classification":
        if model_key == "logreg":
            return {"model__C": [0.5, 1.0, 2.0]}
        if model_key == "linear_svc":
            return {"model__C": [0.5, 1.0, 2.0]}
        if model_key == "svc":
            return {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"]}
        if model_key == "sgd":
            return {
                "model__loss": ["hinge", "log_loss"],
                "model__alpha": [1e-4, 1e-3],
                "model__penalty": ["l2"],
            }
        if model_key == "naive_bayes":
            return {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}
        if model_key == "random_forest":
            return {
                "model__n_estimators": [200],
                "model__max_depth": [None, 12],
                "model__min_samples_split": [2, 5],
            }
        if model_key == "decision_tree":
            return {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5, 10],
            }
        if model_key == "gradient_boosting":
            return {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            }
        if model_key == "bagging":
            return {
                "model__n_estimators": [50, 200],
                "model__max_samples": [0.7, 1.0],
            }
        if model_key == "knn":
            return {"model__n_neighbors": [3, 5, 7]}
        if model_key == "mlp":
            return {
                "model__hidden_layer_sizes": [(50,), (100,)],
                "model__alpha": [1e-4, 1e-3],
            }
        if model_key == "xgboost":
            return {
                "model__n_estimators": [200],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
            }
        return {}

    # regression
    if model_key == "ridge":
        return {"model__alpha": [0.1, 1.0, 10.0]}
    if model_key == "lasso":
        return {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
    if model_key == "linear_regression":
        return {}
    if model_key == "poly_ridge":
        return {"poly__degree": [2], "model__alpha": [0.1, 1.0, 10.0]}
    if model_key == "random_forest":
        return {"model__n_estimators": [300], "model__max_depth": [None, 12]}
    if model_key == "decision_tree":
        return {"model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5, 10]}
    if model_key == "knn_reg":
        return {"model__n_neighbors": [3, 5, 7]}
    if model_key == "gradient_boosting":
        return {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    if model_key == "adaboost":
        return {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1]}
    if model_key == "bagging":
        return {"model__n_estimators": [50, 200], "model__max_samples": [0.7, 1.0]}
    if model_key == "mlp":
        return {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__alpha": [1e-4, 1e-3],
        }
    if model_key == "lightgbm":
        return {
            "model__n_estimators": [300, 600],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 8],
        }
    if model_key == "xgboost":
        return {
            "model__n_estimators": [300],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        }
    return {}


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}. Run prepare_data first.")

    cfg = load_dataset_config()
    mcfg = load_model_config()

    task = (getattr(cfg, "task", None) or "classification").lower()
    y_col = cfg.target_col
    id_col = cfg.id_col if getattr(cfg, "id_col", None) else None
    text_col = getattr(cfg, "text_col", None)

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if y_col not in df.columns:
        raise KeyError(f"Target column '{y_col}' not found in processed dataset.")

    # X columns
    feature_cols = getattr(cfg, "feature_cols", None) or []
    if feature_cols:
        x_cols = [c for c in feature_cols if c in df.columns and c != y_col and c != id_col]
    else:
        x_cols = [c for c in df.columns if c not in {y_col, id_col}]

    # text cols (0..1 col currently)
    text_cols: List[str] = []
    if text_col and text_col in df.columns and text_col in x_cols:
        text_cols = [text_col]

    # infer types
    num_cols, cat_cols = _infer_cols(df, x_cols, text_cols)

    # target y
    y_raw = df[y_col]
    task_in = task
    task_note = None

    # Auto-switch if user selected classification but Y looks continuous
    if task == "classification" and _looks_continuous_numeric(y_raw):
        task = "regression"
        task_note = "Auto-switch: Y ressemble à une variable continue → entraînement en régression."
    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()

    X = df[x_cols].copy()

    # ---------------- Split
    test_size = 0.20
    val_size_of_remaining = 0.20
    random_state = 42

    split_warning = None

    if task == "classification":
        strat1 = _maybe_stratify(y)
        if strat1 is None:
            split_warning = "Stratify désactivé (au moins une classe a <2 exemples)."

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat1
        )

        strat2 = _maybe_stratify(y_temp)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_of_remaining, random_state=random_state, stratify=strat2
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_of_remaining, random_state=random_state
        )

    _save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    # ---------------- CV folds robust
    requested_folds = int(getattr(mcfg, "cv_folds", 3) or 3)
    cv_warning = None

    if task == "classification":
        counts = np.bincount(y_train) if len(y_train) else np.array([0])
        min_count = int(counts.min()) if len(counts) else 0

        # StratifiedKFold requires each class to have >= n_splits.
        # If the dataset is tiny / unbalanced (min_count < 2), we fallback to KFold to avoid crashing.
        if min_count < 2:
            cv_used = min(requested_folds, len(X_train))
            if cv_used < 2:
                cv_used = 2
            cv_warning = (
                f"CV fallback: StratifiedKFold impossible (min class count={min_count}) -> KFold(n_splits={cv_used})."
            )
            cv = KFold(n_splits=cv_used, shuffle=True, random_state=random_state)
        else:
            cv_used = min(requested_folds, min_count)
            if cv_used < 2:
                cv_used = 2
                cv_warning = "Dataset trop petit: fallback CV=2."
            if cv_used != requested_folds:
                cv_warning = f"CV ajustée: requested={requested_folds} -> used={cv_used} (min class count={min_count})."
            cv = StratifiedKFold(n_splits=cv_used, shuffle=True, random_state=random_state)

        scoring = "f1_macro"
    else:
        cv_used = max(2, min(requested_folds, len(X_train)))
        if cv_used != requested_folds:
            cv_warning = f"CV ajustée: requested={requested_folds} -> used={cv_used}."
        cv = KFold(n_splits=cv_used, shuffle=True, random_state=random_state)
        scoring = "r2"

    n_jobs = int(getattr(mcfg, "n_jobs", -1))

    # ---------------- Model selection
    selected_models = getattr(mcfg, "selected_models", None) or []
    selected_models = [m for m in selected_models if isinstance(m, str)]
    if not selected_models:
        selected_models = ["logreg", "random_forest", "linear_svc"] if task == "classification" else ["ridge", "random_forest", "xgboost"]
    selected_models = selected_models[:3]

    available = _get_models(task)
    preprocess = _build_preprocess(num_cols, cat_cols, text_cols)

    comparison: List[Dict[str, Any]] = []
    best_choice: Optional[Dict[str, Any]] = None
    best_params: Dict[str, Any] = {}

    for mk in selected_models:
        if mk not in available:
            comparison.append({"model": mk, "error": "Model not available (missing dependency?)"})
            continue

        try:
            estimator = available[mk]
            pipe = _build_model_pipeline(mk, estimator, preprocess)

            grid = _get_param_grid(mk, task)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )

            gs.fit(X_train, y_train)

            best_pipe = gs.best_estimator_
            best_cv = float(gs.best_score_)

            # Validation metrics
            if task == "classification":
                y_pred_val = best_pipe.predict(X_val)
                try:
                    y_proba_val = best_pipe.predict_proba(X_val)
                except Exception:
                    y_proba_val = None
                val_metrics = _score_classification(y_val, y_pred_val, y_proba_val)
                val_primary = float(val_metrics["f1_macro"])
            else:
                y_pred_val = best_pipe.predict(X_val)
                val_metrics = _score_regression(y_val, y_pred_val)
                val_primary = float(val_metrics["r2"])

            row = {
                "model": mk,
                "cv_best_score": best_cv,
                "val_metrics": val_metrics,
                "val_primary_metric": val_primary,
                "best_params": gs.best_params_,
            }
            comparison.append(row)

            if best_choice is None or val_primary > float(best_choice["val_primary_metric"]):
                best_choice = row
                best_params = gs.best_params_

        except Exception as e:
            comparison.append({"model": mk, "error": str(e)})

    if best_choice is None:
        raise RuntimeError("No model trained successfully. Check dataset columns / selected models.")

    # ---------------- Refit best on train+val, evaluate on test
    best_key = best_choice["model"]
    best_estimator = available[best_key]
    best_pipe = _build_model_pipeline(best_key, best_estimator, preprocess)

    if best_params:
        best_pipe.set_params(**best_params)

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    best_pipe.fit(X_trainval, y_trainval)

    if task == "classification":
        y_pred_test = best_pipe.predict(X_test)
        try:
            y_proba_test = best_pipe.predict_proba(X_test)
        except Exception:
            y_proba_test = None
        test_metrics = _score_classification(y_test, y_pred_test, y_proba_test)
        best_test_primary = float(test_metrics["f1_macro"])
    else:
        y_pred_test = best_pipe.predict(X_test)
        test_metrics = _score_regression(y_test, y_pred_test)
        best_test_primary = float(test_metrics["r2"])

    # ---------------- Save artifacts
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # [OK] This now works (pickle-safe pipeline)
    joblib.dump(best_pipe, MODEL_PATH, compress=3)

    run_id = _now_run_id()
    run_dir = RUNS_DIR / run_id
    run_reports_dir = run_dir / "reports"
    run_models_dir = run_dir / "models"
    run_splits_dir = run_dir / "data" / "processed" / "splits"
    run_plots_dir = run_reports_dir / "plots"

    run_reports_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    run_splits_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir.mkdir(parents=True, exist_ok=True)

    justification = (
        "Choix du meilleur modèle: comparaison sur le **jeu validation** (même X, même preprocessing). "
        f"On retient celui qui maximise `{scoring}` sur validation. "
        "Ensuite, on ré-entraine ce modèle sur (train+val) et on reporte les métriques finales sur test."
    )

    meta = {
        "run_id": run_id,
        "task": task,
        "task_input": task_in,
        "task_effective": task,
        "task_note": task_note,
        "split_warning": split_warning,
        "target_col": y_col,
        "x_cols": x_cols,
        "id_col": id_col,
        "text_cols": text_cols,
        "feature_types": {"num": num_cols, "cat": cat_cols, "text": text_cols},
        "selected_models": selected_models,
        "split": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "test_size": test_size,
            "val_size_of_remaining": val_size_of_remaining,
        },
        "cv_folds_requested": requested_folds,
        "cv_folds_used": cv_used,
        "cv_warning": cv_warning,
        "n_jobs": n_jobs,
        "scoring": scoring,
        "comparison": comparison,
        "best_model": best_key,
        "best_val_primary_metric": float(best_choice["val_primary_metric"]),
        "best_test_metrics": test_metrics,
        "best_test_primary_metric": best_test_primary,
        "justification": justification,
        "best_params": best_params,
        "artifacts": {
            "run_dir": str(run_dir),
            "run_reports_dir": str(run_reports_dir),
            "run_plots_dir": str(run_plots_dir),
            "model_path": str(MODEL_PATH),
            "meta_path": str(META_PATH),
            "splits_dir": str(SPLIT_DIR),
        },
    }

    if task == "classification" and label_encoder is not None:
        meta["label_mapping"] = {int(i): str(lbl) for i, lbl in enumerate(label_encoder.classes_)}

    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    shutil.copy2(MODEL_PATH, run_models_dir / "best_model.pkl")
    _copy_tree(SPLIT_DIR, run_splits_dir)

    _safe_print(f"[OK] run_id = {run_id}")
    _safe_print(f"[OK] Best model (val): {best_key} | val_primary={meta['best_val_primary_metric']}")
    _safe_print(f"[OK] Final test primary: {meta['best_test_primary_metric']}")
    if cv_warning:
        _safe_print(f"[WARN] {cv_warning}")
    _safe_print(f"[OK] Saved model: {MODEL_PATH}")
    _safe_print(f"[OK] Meta: {META_PATH}")
    _safe_print(f"[OK] Snapshot: {run_dir}")


if __name__ == "__main__":
    main()
