# -*- coding: utf-8 -*-
"""src.reco.build_item_index

Build a simple TF-IDF content index for recommendations.

This stage is **optional**:
- It runs only if the dataset has an ID column and at least one text column.
- Works well for the SAKIGA fashion dataset, but can also work for other datasets.

Outputs:
- models/reco_item_index.pkl (joblib dict payload)
- reports/reco_index_meta.json (info for the UI)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.dataset_config import load_dataset_config

MODEL_PATH = Path("models") / "reco_item_index.pkl"
META_PATH = Path("reports") / "reco_index_meta.json"

def _build_text(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].astype(str).fillna(""))
    if not parts:
        return pd.Series([""]*len(df))
    txt = parts[0]
    for p in parts[1:]:
        txt = txt + " " + p
    return txt

def main() -> None:
    cfg = load_dataset_config()
    raw_path = Path(cfg.raw_csv)
    if not raw_path.exists():
        print("⚠️ Reco index skipped: raw CSV missing.")
        return

    df = pd.read_csv(raw_path, sep=cfg.sep)
    id_col = cfg.id_col if cfg.id_col in df.columns else None
    if not id_col:
        print("⚠️ Reco index skipped: no id_col configured/found.")
        return

    # Choose text columns
    text_cols = []
    if cfg.text_col and cfg.text_col in df.columns:
        text_cols = [cfg.text_col]
    else:
        # fallback: all object columns except target
        text_cols = [c for c in df.columns if df[c].dtype == object and c != cfg.target_col][:3]

    if not text_cols:
        print("⚠️ Reco index skipped: no text columns found.")
        return

    items = df[[id_col] + [c for c in text_cols if c in df.columns]].copy()
    items[id_col] = items[id_col].astype(int, errors="ignore")

    corpus = _build_text(items, text_cols)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words=None
    )
    X = vectorizer.fit_transform(corpus)

    payload = {
        "id_col": id_col,
        "vectorizer": vectorizer,
        "item_matrix": X,
        "items": items,
        "text_cols": text_cols,
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(payload, MODEL_PATH, compress=3)

    meta = {
        "id_col": id_col,
        "text_cols": text_cols,
        "rows": int(len(items)),
        "sample_ids": items[id_col].dropna().head(10).tolist(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Reco index built: {MODEL_PATH}")
    print(f"✅ Meta: {META_PATH}")

if __name__ == "__main__":
    main()
