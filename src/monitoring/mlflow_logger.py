# -*- coding: utf-8 -*-
"""src.monitoring.mlflow_logger

MLflow logging helpers (BONUS)

Goal:
- Log the current pipeline artifacts (metrics, plots, model) into MLflow.
- Keep it dataset-agnostic and safe (works even if only some artifacts exist).

Requires:
  pip install -r requirements_mlops.txt
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, Dict, Any

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def log_current_run(experiment_name: str = "SAKIGA-DS", run_name: Optional[str] = None, tracking_uri: Optional[str] = None) -> str:
    try:
        import mlflow
    except Exception as e:
        raise RuntimeError("MLflow not installed. Install optional stack: pip install -r requirements_mlops.txt") from e

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    reports = Path("reports")
    models = Path("models")
    data_cfg = Path("data/config/dataset_config.json")

    metrics_path = reports / "metrics.json"
    meta_path = reports / "train_meta.json"

    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    meta = _read_json(meta_path) if meta_path.exists() else {}
    cfg = _read_json(data_cfg) if data_cfg.exists() else {}

    with mlflow.start_run(run_name=run_name):
        # Params (compact)
        for k in ["task", "target_col", "sep", "raw_csv"]:
            v = cfg.get(k)
            if v is not None:
                mlflow.log_param(k, v)

        best_model = meta.get("best_model")
        if best_model:
            mlflow.log_param("best_model", best_model)

        # Metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(k, float(v))

        # Artifacts
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        if meta_path.exists():
            mlflow.log_artifact(str(meta_path), artifact_path="reports")

        plots_dir = reports / "plots"
        if plots_dir.exists():
            for p in sorted(plots_dir.glob("*.png")):
                mlflow.log_artifact(str(p), artifact_path="plots")

        model_path = models / "best_model.pkl"
        if model_path.exists():
            # Safe: log as file artifact (no sklearn signature required)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        # pointer for convenience
        return mlflow.active_run().info.run_id
