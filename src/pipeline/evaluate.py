# -*- coding: utf-8 -*-
"""src.pipeline.evaluate

Stage 3: Evaluate best model + generate plots + metrics + run snapshot.

Reads:
- models/best_model.pkl
- data/processed/splits/X_test.csv, y_test.csv
- reports/train_meta.json

Writes (latest):
- reports/metrics.json
- reports/predictions_test.csv
- reports/confusion_matrix.csv (classification)
- reports/plots/*.png

Writes (snapshot):
- runs/<run_id>/meta.json
- runs/<run_id>/models/best_model.pkl
- runs/<run_id>/reports/train_meta.json
- runs/<run_id>/reports/metrics.json
- runs/<run_id>/reports/predictions_test.csv
- runs/<run_id>/reports/confusion_matrix.csv (classification)
- runs/<run_id>/reports/plots/*.png
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from src.pipeline.train import join_text_cols, to_dense_matrix



from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)
import sys

def _safe_print(msg: str) -> None:
    """Print sans crash sur Windows (console cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = msg.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
        print(safe)


MODEL_PATH = Path("models") / "best_model.pkl"
META_PATH = Path("reports") / "train_meta.json"
SPLIT_DIR = Path("data") / "processed" / "splits"

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

RUNS_ROOT = Path("runs")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bar_compare(comparison: list, metric_name: str, out_path: Path) -> None:
    """comparison rows from train.py store val_primary_metric and val_metrics."""
    if not comparison:
        return
    models = [c.get("model", "?") for c in comparison]
    vals = []
    for c in comparison:
        # Prefer validation primary metric, fallback to any known keys
        v = c.get("val_primary_metric")
        if v is None:
            v = c.get("primary_metric")
        if v is None and isinstance(c.get("val_metrics"), dict):
            v = c["val_metrics"].get(metric_name)
        if v is None:
            v = c.get(metric_name, 0.0)
        vals.append(float(v) if v is not None else 0.0)

    plt.figure(figsize=(7, 3.2))
    plt.bar(models, vals)
    plt.title(f"Comparaison des modèles (val {metric_name})")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20, ha="right")
    _safe_savefig(out_path)


def _label_inv_map(label_mapping: Optional[dict]) -> Optional[Dict[str, str]]:
    """
    Supports BOTH formats:
      A) {int_code: label}  (current train.py)
      B) {label: int_code}
    Returns {str(int_code): label}
    """
    if not isinstance(label_mapping, dict) or not label_mapping:
        return None

    # Detect format
    k0 = next(iter(label_mapping.keys()))
    v0 = next(iter(label_mapping.values()))

    # A) keys are numeric -> already code->label
    if isinstance(k0, int) or (isinstance(k0, str) and k0.isdigit()):
        return {str(k): str(v) for k, v in label_mapping.items()}

    # B) values are numeric -> label->code => invert
    if isinstance(v0, int) or (isinstance(v0, str) and str(v0).isdigit()):
        return {str(v): str(k) for k, v in label_mapping.items()}

    # Fallback: don't transform
    return {str(k): str(v) for k, v in label_mapping.items()}


def _pretty_labels_from_codes(raw_codes: List[str], inv_map: Optional[Dict[str, str]]) -> List[str]:
    if not inv_map:
        return raw_codes
    return [inv_map.get(c, c) for c in raw_codes]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/best_model.pkl not found. Run Train first.")
    if not (SPLIT_DIR / "X_test.csv").exists():
        raise FileNotFoundError("Missing test split. Run Train first.")
    if not META_PATH.exists():
        raise FileNotFoundError("reports/train_meta.json not found. Run Train first.")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    task = (meta.get("task") or "classification").lower()
    best_model_name = meta.get("best_model", "best_model")

    inv_map = _label_inv_map(meta.get("label_mapping"))

    X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")["y"]
    model = joblib.load(MODEL_PATH)

    comparison = meta.get("comparison") or []

    # Create run snapshot folder (standard structure for UI)
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")

    run_base = RUNS_ROOT / run_id
    run_reports = run_base / "reports"
    run_plots = run_reports / "plots"
    run_models = run_base / "models"
    run_plots.mkdir(parents=True, exist_ok=True)

    # Base metrics from train meta (these exist even if plots/roc fail)
    primary_metric = meta.get("primary_metric") or ("f1_macro" if task == "classification" else "r2")
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "task": task,
        "best_model": best_model_name,
        "primary_metric": primary_metric,
        "val_default": meta.get("best_val_metrics") or {},
        "test_default": meta.get("best_test_metrics") or {},
        "best_params": meta.get("best_params") or {},
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Predict on test ---
    if task == "classification":
        y_pred = model.predict(X_test)

        # stable string codes
        y_true_s = y_test.astype(str)
        y_pred_s = pd.Series(y_pred).astype(str)

        raw_codes = sorted(y_true_s.unique().tolist())
        display_labels = _pretty_labels_from_codes(raw_codes, inv_map)

        # Confusion matrix + csv
        cm = confusion_matrix(y_true_s, y_pred_s, labels=raw_codes)
        cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
        cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", index=True)
        cm_df.to_csv(run_reports / "confusion_matrix.csv", index=True)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        plt.figure(figsize=(5.2, 4.4))
        disp.plot(values_format="d", cmap=None)
        plt.title(f"Matrice de confusion (test) — {best_model_name}")
        _safe_savefig(PLOTS_DIR / "confusion_matrix.png")
        _safe_savefig(run_plots / "confusion_matrix.png")

        # Predict proba for binary ROC/PR
        proba_pos = None
        try:
            p = model.predict_proba(X_test)
            if hasattr(p, "shape") and p.shape[1] == 2:
                proba_pos = p[:, 1]
        except Exception:
            pass

        # predictions_test.csv
        pred_df = pd.DataFrame({
            "y_true": y_true_s,
            "y_pred": y_pred_s,
        })
        if inv_map:
            pred_df["y_true_label"] = [inv_map.get(v, v) for v in y_true_s.tolist()]
            pred_df["y_pred_label"] = [inv_map.get(v, v) for v in y_pred_s.tolist()]
        if proba_pos is not None:
            pred_df["proba_pos"] = proba_pos
        pred_df.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)
        pred_df.to_csv(run_reports / "predictions_test.csv", index=False)

        if proba_pos is not None and len(raw_codes) == 2:
            pos_code = raw_codes[1]
            pos_name = inv_map.get(pos_code, pos_code) if inv_map else pos_code

            y_bin = (y_true_s == pos_code).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, proba_pos)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(5.4, 3.6))
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1])
            plt.title(f"Courbe ROC — {best_model_name} (AUC={roc_auc:.3f}) | positif={pos_name}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            _safe_savefig(PLOTS_DIR / "roc_curve.png")
            _safe_savefig(run_plots / "roc_curve.png")

            prec, rec, _ = precision_recall_curve(y_bin, proba_pos)
            plt.figure(figsize=(5.4, 3.6))
            plt.plot(rec, prec)
            plt.title(f"Precision–Recall — {best_model_name} | positif={pos_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            _safe_savefig(PLOTS_DIR / "pr_curve.png")
            _safe_savefig(run_plots / "pr_curve.png")

            metrics["roc_auc"] = float(roc_auc)

    else:
        # Regression
        y_pred = model.predict(X_test)
        y_true = pd.to_numeric(y_test, errors="coerce").fillna(0).to_numpy()
        y_pred = np.asarray(y_pred, dtype=float)

        pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        pred_df.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)
        pred_df.to_csv(run_reports / "predictions_test.csv", index=False)

        plt.figure(figsize=(5.4, 3.8))
        plt.scatter(y_true, y_pred, s=8)
        plt.title(f"Prédictions vs Réel (test) — {best_model_name}")
        plt.xlabel("Réel")
        plt.ylabel("Prédit")
        _safe_savefig(PLOTS_DIR / "pred_vs_actual.png")
        _safe_savefig(run_plots / "pred_vs_actual.png")

        resid = y_true - y_pred
        plt.figure(figsize=(5.4, 3.8))
        plt.hist(resid, bins=30)
        plt.title(f"Résidus (test) — {best_model_name}")
        plt.xlabel("Résidu")
        plt.ylabel("Nombre")
        _safe_savefig(PLOTS_DIR / "residuals.png")
        _safe_savefig(run_plots / "residuals.png")

    # Model comparison plot (based on validation score)
    metric_name = "f1_macro" if task == "classification" else "r2"
    _bar_compare(comparison, metric_name, PLOTS_DIR / "model_compare.png")
    if (PLOTS_DIR / "model_compare.png").exists():
        _safe_copy(PLOTS_DIR / "model_compare.png", run_plots / "model_compare.png")

    # Save metrics (latest + run)
    _write_json(REPORTS_DIR / "metrics.json", metrics)
    _write_json(run_reports / "metrics.json", metrics)
    # legacy (optional)
    _write_json(run_base / "metrics.json", metrics)

    # Save train meta into run (for history UI)
    _safe_copy(META_PATH, run_reports / "train_meta.json")
    _safe_copy(META_PATH, run_base / "train_meta.json")  # legacy

    # Copy model into run (proof/repro)
    _safe_copy(MODEL_PATH, run_models / "best_model.pkl")

    # Root meta.json (quick overview)
    run_meta = {
        "run_id": run_id,
        "created_at": created_at,
        "task": task,
        "best_model": best_model_name,
        "primary_metric": primary_metric,
    }
    _write_json(run_base / "meta.json", run_meta)

    _safe_print(f"[OK] run_id = {run_id}")
    _safe_print(f"[OK] latest metrics: {REPORTS_DIR/'metrics.json'}")
    _safe_print(f"[OK] latest plots: {PLOTS_DIR}")
    _safe_print(f"[OK] run saved: {run_base}")


if __name__ == "__main__":
    main()
