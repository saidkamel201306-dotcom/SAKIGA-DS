# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

DEFAULT_MODEL_CONFIG_PATH = Path("data") / "config" / "model_config.json"

# Keys used across UI + training pipeline
# (Tu peux en sélectionner max 3 dans l’UI)
CLASSIFICATION_MODELS = [
    "logreg",
    "random_forest",
    "linear_svc",
    "svc",                # SVC (RBF)
    "sgd",                # SGDClassifier
    "naive_bayes",        # GaussianNB
    "knn",
    "decision_tree",
    "gradient_boosting",
    "bagging",
    "mlp",                # MLPClassifier
    "xgboost",
]

REGRESSION_MODELS = [
    "ridge",
    "lasso",
    "linear_regression",
    "poly_ridge",         # PolynomialFeatures + Ridge (numérique uniquement recommandé)
    "random_forest",
    "decision_tree",
    "knn_reg",
    "gradient_boosting",
    "adaboost",
    "bagging",
    "mlp",                # MLPRegressor
    "lightgbm",           # optionnel (si lightgbm est installé)
    "xgboost",
]


@dataclass
class ModelConfig:
    # choose up to 3 model keys
    task_type: str = "classification"   # "classification" | "regression"
    selected_models: List[str] = None

    # GridSearchCV config
    cv_folds: int = 3
    n_jobs: int = -1

    def __post_init__(self):
        if self.selected_models is None:
            # safe defaults (dépendent du type de tâche)
            if (self.task_type or "classification").lower() == "regression":
                self.selected_models = ["ridge", "random_forest", "xgboost"]
            else:
                self.selected_models = ["logreg", "random_forest", "linear_svc"]


def load_model_config(path: Path = DEFAULT_MODEL_CONFIG_PATH) -> ModelConfig:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = ModelConfig()
        save_model_config(cfg, path)
        return cfg

    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = ModelConfig(**data)

    # sanitize
    if (cfg.task_type or "classification").lower() == "classification":
        allowed = set(CLASSIFICATION_MODELS)
    else:
        allowed = set(REGRESSION_MODELS)

    cfg.selected_models = [m for m in (cfg.selected_models or []) if m in allowed]
    if not cfg.selected_models:
        # reset with safe defaults for this task_type
        cfg = ModelConfig(task_type=cfg.task_type)

    cfg.selected_models = (cfg.selected_models or [])[:3]
    return cfg


def save_model_config(cfg: ModelConfig, path: Path = DEFAULT_MODEL_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
