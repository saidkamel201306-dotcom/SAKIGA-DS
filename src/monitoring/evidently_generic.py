# -*- coding: utf-8 -*-
"""src.monitoring.evidently_generic

Evidently report (BONUS) — dataset-agnostic

- Works with ANY processed dataset
- Uses train/test splits if available to create a "reference" vs "current" report
- Generates HTML + JSON in reports/

Requires (optional):
  pip install -r requirements_mlops.txt

Run:
  python -m src.monitoring.evidently_generic
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataQualityPreset, DataDriftPreset
    except Exception as e:
        raise RuntimeError(
            "Evidently not installed. Install optional stack: pip install -r requirements_mlops.txt"
        ) from e

    split_dir = Path("data/processed/splits")
    x_train = split_dir / "X_train.csv"
    x_test = split_dir / "X_test.csv"
    y_train = split_dir / "y_train.csv"
    y_test = split_dir / "y_test.csv"

    # Reference / current data
    ref = None
    cur = None

    if x_train.exists() and x_test.exists():
        ref = pd.read_csv(x_train)
        cur = pd.read_csv(x_test)

        # Append targets if present (nice-to-have)
        if y_train.exists():
            try:
                ref["_target"] = pd.read_csv(y_train).iloc[:, 0]
            except Exception:
                pass
        if y_test.exists():
            try:
                cur["_target"] = pd.read_csv(y_test).iloc[:, 0]
            except Exception:
                pass
    else:
        # fallback: compare processed dataset to itself (quality only)
        processed = Path("data/processed/dataset.csv")
        if not processed.exists():
            raise FileNotFoundError("No data found. Run Prepare first.")
        cur = pd.read_csv(processed)
        ref = None

    metrics = [DataQualityPreset()]
    if ref is not None:
        metrics.append(DataDriftPreset())

    report = Report(metrics=metrics)
    report.run(current_data=cur, reference_data=ref, column_mapping=None)

    html_path = REPORTS_DIR / "evidently_report.html"
    json_path = REPORTS_DIR / "evidently_report.json"

    report.save_html(str(html_path))
    json_path.write_text(report.json(), encoding="utf-8")

    (REPORTS_DIR / "evidently_pointer.json").write_text(
        json.dumps({"html": str(html_path), "json": str(json_path)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
