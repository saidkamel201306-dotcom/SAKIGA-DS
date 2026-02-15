# -*- coding: utf-8 -*-
"""
Generate an Evidently report (optional) for our TEXT classification pipeline.
Requires: pip install -r requirements_mlops.txt

Run:
  python -m src.monitoring.evidently_report
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

def main():
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset, DataQualityPreset
    except Exception as e:
        raise RuntimeError("Evidently not installed. Run: pip install -r requirements_mlops.txt") from e

    processed = Path("data/processed/dataset.csv")
    if not processed.exists():
        raise FileNotFoundError("data/processed/dataset.csv not found. Run prepare/train first.")

    df = pd.read_csv(processed)
    if "text" not in df.columns or "gender_group" not in df.columns:
        raise ValueError("Expected columns: text, gender_group in dataset.csv")

    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("models/best_model.pkl not found. Train first.")

    import joblib
    model = joblib.load(model_path)

    X = df["text"].astype(str)
    y_true = df["gender_group"].astype(str)
    y_pred = model.predict(X)

    # Evidently prefers tabular data: we provide simple features derived from text
    cur = pd.DataFrame({
        "text_len": X.str.len(),
        "target": y_true,
        "prediction": pd.Series(y_pred).astype(str),
    })

    report = Report(metrics=[DataQualityPreset(), ClassificationPreset()])
    report.run(current_data=cur, reference_data=None, column_mapping=None)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "evidently_classification_report.html"
    json_path = out_dir / "evidently_classification_report.json"
    report.save_html(str(html_path))
    json_path.write_text(report.json(), encoding="utf-8")

    (out_dir / "evidently_pointer.json").write_text(json.dumps({
        "html": str(html_path),
        "json": str(json_path)
    }, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
