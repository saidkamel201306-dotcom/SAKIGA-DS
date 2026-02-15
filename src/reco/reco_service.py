# -*- coding: utf-8 -*-
"""Recommendation utilities (content-based).

We recommend items similar to the user's recently viewed items.

Algorithm:
- user profile vector = mean of TF-IDF vectors for viewed items
- compute cosine similarity between user profile and all items
- return top-K not yet seen

This is simple, explainable, and easy to demo.

Compatibility:
Supports multiple index payload formats:
A) legacy payload keys: id_col, vectorizer, item_matrix, items (DataFrame)
B) new payload keys: id_col, vectorizer, X/item_matrix, items may be absent (fallback to data/raw/styles.csv)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecoIndex:
    id_col: str
    vectorizer: Any
    item_matrix: Any  # sparse matrix (csr)
    items_df: pd.DataFrame


def _load_items_df_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    for key in ("items", "items_df", "df"):
        if key in payload and payload[key] is not None:
            df = payload[key]
            if isinstance(df, dict):
                df = pd.DataFrame(df)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    styles_path = Path("data") / "raw" / "styles.csv"
    if styles_path.exists():
        # auto-detect sep
        try:
            return pd.read_csv(styles_path, sep=";")
        except Exception:
            return pd.read_csv(styles_path)

    raise FileNotFoundError("No items dataframe in payload and data/raw/styles.csv not found.")


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("id_col", "id")

    if "vectorizer" not in payload and "tfidf" in payload:
        payload["vectorizer"] = payload["tfidf"]

    if "item_matrix" not in payload:
        if "X" in payload:
            payload["item_matrix"] = payload["X"]

    if "items" not in payload:
        payload["items"] = _load_items_df_from_payload(payload)
    else:
        if not isinstance(payload["items"], pd.DataFrame):
            payload["items"] = pd.DataFrame(payload["items"])

    id_col = payload["id_col"]
    if id_col not in payload["items"].columns:
        raise KeyError(f"id_col='{id_col}' not found in items_df columns={list(payload['items'].columns)}")

    return payload


def load_index(path: str) -> RecoIndex:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Index payload must be dict, got {type(payload)}")

    payload = _normalize_payload(payload)
    return RecoIndex(
        id_col=payload["id_col"],
        vectorizer=payload.get("vectorizer"),
        item_matrix=payload["item_matrix"],
        items_df=payload["items"],
    )


def recommend_from_viewed(index: RecoIndex, viewed_item_ids: List[int], k: int = 10) -> List[Dict[str, Any]]:
    items = index.items_df
    id_col = index.id_col
    mat = index.item_matrix

    if k <= 0:
        return []

    # Cold start
    if not viewed_item_ids:
        subset = items.head(k).copy()
        subset["score"] = None
        return subset.to_dict(orient="records")

    # id -> row (assumes matrix order matches items_df order)
    ids_list = items[id_col].astype(int, errors="ignore")
    try:
        ids_list_int = ids_list.astype(int).tolist()
    except Exception:
        ids_list_int = [int(x) for x in ids_list.tolist()]

    id_to_row = {int(v): i for i, v in enumerate(ids_list_int)}

    viewed_set = set(int(x) for x in viewed_item_ids)
    rows = [id_to_row[i] for i in viewed_set if i in id_to_row]

    if not rows:
        subset = items.head(k).copy()
        subset["score"] = None
        return subset.to_dict(orient="records")

    user_vec = mat[rows].mean(axis=0)

    # sklearn may return np.matrix; convert safely
    user_vec = np.asarray(user_vec)

    sims = cosine_similarity(user_vec, mat).ravel()

    for r in rows:
        sims[r] = -1.0

    top_idx = np.argsort(-sims)[:k]
    recs = items.iloc[top_idx].copy()
    recs["score"] = sims[top_idx]
    return recs.to_dict(orient="records")
