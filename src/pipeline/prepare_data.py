# -*- coding: utf-8 -*-
"""src.pipeline.prepare_data

Stage 1: Prepare data (generic) + EDA report (lightweight but complete enough)

- Read configured CSV (data/config/dataset_config.json)
- Select [id(optional)] + X + Y (apply drop_cols)
- Cleaning:
  - normalize column names (strip)
  - drop duplicates
  - drop rows with missing Y
  - smart type casting (numeric-like strings -> numeric)
  - missing values handling (numeric: median, non-numeric: most_frequent / empty for text)
  - outliers handling (IQR clipping on numeric X, safe-guarded)
- Save standardized file: data/processed/dataset.csv

Artifacts (jury / reproducibility):
- reports/eda_summary.json
- reports/eda_missing.csv
- reports/eda_outliers.csv (IQR-based, numeric X only)
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from src.config.dataset_config import load_dataset_config


def _safe_print(msg: str) -> None:
    """Print sans crash sur Windows (console cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = msg.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
        print(safe)


OUT_PATH = Path("data") / "processed" / "dataset.csv"
REPORTS_DIR = Path("reports")
EDA_SUMMARY = REPORTS_DIR / "eda_summary.json"
EDA_MISSING = REPORTS_DIR / "eda_missing.csv"
EDA_OUTLIERS = REPORTS_DIR / "eda_outliers.csv"

# Values to treat as missing when reading CSV
NA_VALUES = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "-", "--", "?"]

# Outlier settings (safe defaults)
IQR_FACTOR = 1.5
MIN_NUMERIC_N_FOR_OUTLIERS = 20
MAX_OUTLIER_RATIO_TO_CLIP = 0.25  # if too many, we assume it's not "outliers" but distribution


def _to_numeric_if_possible(s: pd.Series, threshold: float = 0.90) -> tuple[pd.Series, bool]:
    """
    Try converting to numeric if most values are numeric-like.
    Returns (converted_series, did_convert).
    """
    if pd.api.types.is_numeric_dtype(s):
        return s, True

    # Only attempt on object/category
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)):
        return s, False

    s2 = s.copy()
    # replace comma decimal "12,3" => "12.3"
    s2 = s2.astype(str).str.replace(",", ".", regex=False)
    conv = pd.to_numeric(s2, errors="coerce")

    non_na = int(s.notna().sum())
    if non_na == 0:
        return s, False

    ratio = float(conv.notna().sum() / max(1, non_na))
    if ratio >= threshold:
        return conv, True
    return s, False


def _iqr_bounds(x: pd.Series, factor: float = IQR_FACTOR):
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return None
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = float(q3 - q1)
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return q1, q3, iqr, float(lower), float(upper)


def main() -> None:
    cfg = load_dataset_config()
    raw_path = Path(cfg.raw_csv)

    if not raw_path.exists():
        raise FileNotFoundError(f"CSV not found: {raw_path}")

    # Read CSV robustly
    df = pd.read_csv(
        raw_path,
        sep=cfg.sep,
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]

    # ---------------------------
    # Target column (Y) checks
    # ---------------------------
    y_col = (getattr(cfg, "target_col", None) or "").strip()
    if not y_col:
        raise KeyError("Target column (Y) vide. Va sur la page '📁 Data' et choisis Y, puis relance.")

    if y_col not in df.columns:
        raise KeyError(
            f"Target column (Y) is missing or not found: '{y_col}'. "
            f"Please choose Y in the Data page. Columns: {list(df.columns)}"
        )

    # ID/text cols (optional)
    id_col = cfg.id_col if getattr(cfg, "id_col", None) in df.columns else None
    text_col = cfg.text_col if getattr(cfg, "text_col", None) in df.columns else None

    # X columns: if empty => auto (all except Y and id)
    if cfg.feature_cols:
        x_cols = [c for c in cfg.feature_cols if c in df.columns and c != y_col and c != id_col]
    else:
        x_cols = [c for c in df.columns if c not in {y_col, id_col}]

    # Apply drop_cols
    drop_set = set(cfg.drop_cols or [])
    x_cols = [c for c in x_cols if c not in drop_set]

    # Final selection order
    cols = []
    if id_col:
        cols.append(id_col)
    cols += x_cols + [y_col]

    out_df = df[cols].copy()

    # ---------- Cleaning: duplicates ----------
    n_before = int(len(out_df))
    out_df = out_df.drop_duplicates()
    n_dups = int(n_before - len(out_df))

    # ---------- Cleaning: drop missing Y ----------
    n_before_y = int(len(out_df))
    out_df = out_df.dropna(subset=[y_col])
    n_drop_y = int(n_before_y - len(out_df))

    # ---------- Type casting (numeric-like strings -> numeric) ----------
    cast_info = {"converted_to_numeric": []}
    for c in x_cols:
        converted, did = _to_numeric_if_possible(out_df[c], threshold=0.90)
        if did and not pd.api.types.is_numeric_dtype(out_df[c]):
            out_df[c] = converted
            cast_info["converted_to_numeric"].append(c)

    # ---------- Missing values handling ----------
    # numeric: median ; non-numeric: most_frequent ; text_col: empty string
    impute_info = {"numeric_median": [], "categorical_mode": [], "text_empty": []}

    for c in x_cols:
        if pd.api.types.is_numeric_dtype(out_df[c]):
            med = pd.to_numeric(out_df[c], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(med)
            impute_info["numeric_median"].append(c)
        else:
            # For designated text column, keep it as string and fill with ""
            if text_col and c == text_col:
                out_df[c] = out_df[c].fillna("").astype(str)
                impute_info["text_empty"].append(c)
            else:
                # Mode imputation
                s = out_df[c]
                try:
                    mode = s.mode(dropna=True)
                    fill_val = mode.iloc[0] if len(mode) > 0 else ""
                except Exception:
                    fill_val = ""
                out_df[c] = s.fillna(fill_val)
                impute_info["categorical_mode"].append(c)

    # ---------- Outliers handling (IQR clipping on numeric X only, safeguarded) ----------
    outlier_rows = []
    clipped_cols = []
    for c in x_cols:
        if not pd.api.types.is_numeric_dtype(out_df[c]):
            continue

        bounds = _iqr_bounds(out_df[c], factor=IQR_FACTOR)
        if bounds is None:
            outlier_rows.append({
                "column": c, "n": int(out_df[c].notna().sum()),
                "q1": None, "q3": None, "iqr": None, "lower": None, "upper": None,
                "outliers": 0, "clipped": False, "outlier_ratio": 0.0
            })
            continue

        q1, q3, iqr, lower, upper = bounds
        x = pd.to_numeric(out_df[c], errors="coerce")
        mask = (x < lower) | (x > upper)
        out_count = int(mask.sum())
        n_num = int(x.notna().sum())
        ratio = float((out_count / max(1, n_num)) if n_num else 0.0)

        # Clip only if enough data and ratio looks reasonable
        do_clip = (n_num >= MIN_NUMERIC_N_FOR_OUTLIERS) and (ratio <= MAX_OUTLIER_RATIO_TO_CLIP) and (out_count > 0)
        if do_clip:
            out_df[c] = x.clip(lower, upper)
            clipped_cols.append(c)

        outlier_rows.append({
            "column": c,
            "n": n_num,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper),
            "outliers": out_count,
            "clipped": bool(do_clip),
            "outlier_ratio": float(ratio),
        })

    # ---------- Save processed dataset ----------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    # ---------- EDA reports ----------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Missing report (after imputation, still useful for Y + leftover)
    miss = pd.DataFrame({
        "column": out_df.columns,
        "missing": [int(out_df[c].isna().sum()) for c in out_df.columns],
        "missing_ratio": [float(out_df[c].isna().mean()) for c in out_df.columns],
        "dtype": [str(out_df[c].dtype) for c in out_df.columns],
        "n_unique": [int(out_df[c].nunique(dropna=True)) for c in out_df.columns],
    }).sort_values("missing_ratio", ascending=False)
    miss.to_csv(EDA_MISSING, index=False)

    outliers_df = (
        pd.DataFrame(outlier_rows).sort_values("outliers", ascending=False)
        if outlier_rows else
        pd.DataFrame(columns=["column", "n", "q1", "q3", "iqr", "lower", "upper", "outliers", "clipped", "outlier_ratio"])
    )
    outliers_df.to_csv(EDA_OUTLIERS, index=False)

    task = (getattr(cfg, "task", None) or "classification").lower()
    y_ser = out_df[y_col]

    if task == "classification":
        y_counts = y_ser.astype(str).value_counts(dropna=False).to_dict()
        y_info = {"type": "classification", "classes": {str(k): int(v) for k, v in y_counts.items()}}
    else:
        y_num = pd.to_numeric(y_ser, errors="coerce")
        y_info = {
            "type": "regression",
            "describe": {k: (None if pd.isna(v) else float(v)) for k, v in y_num.describe().to_dict().items()},
        }

    summary = {
        "config_used": asdict(cfg),
        "raw_csv": str(raw_path),
        "sep": cfg.sep,
        "saved_dataset": str(OUT_PATH),
        "shape": {"rows": int(out_df.shape[0]), "cols": int(out_df.shape[1])},
        "target_col": y_col,
        "id_col": id_col,
        "text_col": text_col,
        "x_cols": list(x_cols),
        "drop_cols": list(drop_set),
        "task": task,
        "data_cleaning": {
            "dropped_duplicates": int(n_dups),
            "dropped_missing_y": int(n_drop_y),
            "cast": cast_info,
            "imputation": impute_info,
            "outliers_clipped_cols": clipped_cols,
            "outlier_rules": {
                "method": "iqr",
                "factor": IQR_FACTOR,
                "min_n": MIN_NUMERIC_N_FOR_OUTLIERS,
                "max_ratio_to_clip": MAX_OUTLIER_RATIO_TO_CLIP,
            },
        },
        "target_info": y_info,
        "notes": [
            "Pré-traitement fait ici: sélection X/Y, NA, conversions, imputation, outliers (clipping).",
            "Encoding/Scaling se fait proprement dans le Pipeline du train.py (OneHot/Scaler/Imputer) pour éviter la fuite de données.",
        ],
    }
    EDA_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _safe_print(f"[OK] Saved: {OUT_PATH} | rows: {len(out_df)}")
    _safe_print(f"[OK] EDA: {EDA_SUMMARY} + {EDA_MISSING} + {EDA_OUTLIERS}")


if __name__ == "__main__":
    main()