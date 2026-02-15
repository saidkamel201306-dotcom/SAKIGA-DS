# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class FeatureSuggestResult:
    suggested: List[str]
    report: pd.DataFrame
    ignored: List[str]          # ✅ compat
    task_used: str = ""         # "classification" | "regression"
    note: str = ""              # message UI optionnel


def _infer_task_from_y(y: pd.Series, fallback: str = "classification") -> str:
    """Heuristique simple: si Y est numérique avec beaucoup de valeurs uniques → régression."""
    y_non_null = y.dropna()
    if len(y_non_null) == 0:
        return fallback

    if pd.api.types.is_numeric_dtype(y_non_null):
        nunique = int(y_non_null.nunique(dropna=True))
        # petites cibles numériques peuvent être classification (0/1/2…)
        if nunique <= 20:
            return fallback
        return "regression"

    return "classification"


def suggest_best_x_fast(
    df: pd.DataFrame,
    y_col: str,
    candidate_x: List[str],
    task: str = "classification",
    top_k: int = 10,
    sample_max: int = 8000,
    random_state: int = 42,
    high_cardinality_ratio: float = 0.95,
) -> FeatureSuggestResult:
    """
    Suggest a global set of X features (model-agnostic) using Mutual Information.

    ✅ Includes anti-leakage / anti-ID heuristics:
      - ignores suspicious columns by name (url/link/filename/id/sku/path/...)
      - ignores "almost unique" columns (high cardinality) which act as IDs

    ✅ Robust on any dataset:
      - if task=classification but Y looks continuous -> auto switch to regression MI (no crash)
    """
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' not found in df columns")

    # keep only existing candidate columns (and not y)
    base_cols = [c for c in candidate_x if c in df.columns and c != y_col]
    if not base_cols:
        raise ValueError("No candidate X columns found")

    # Build working frame
    work = df[base_cols + [y_col]].copy()
    work = work.dropna(subset=[y_col])
    if len(work) == 0:
        raise ValueError("No rows left after dropping missing Y")

    # --- Leakage guard (BEFORE sampling) ---
    suspicious_tokens = (
        "link", "url", "image", "img", "filename", "file", "path",
        "sku", "uuid", "guid", "hash", "token"
    )

    ignored: List[str] = []
    safe_cols: List[str] = []
    n_rows = max(1, len(work))

    fallback_note = ""

    for c in base_cols:
        name = str(c).lower().strip()
        s = work[c]

        # 1) name-based heuristics (OK pour tous les types)
        if any(tok in name for tok in suspicious_tokens) or name == "id" or name.endswith("_id"):
            ignored.append(c)
            continue

        nunique = s.nunique(dropna=True)
        ratio = nunique / n_rows

        # 2) high-cardinality : à appliquer surtout sur colonnes "texte/catégorie"
        is_text_like = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        )

        if is_text_like and ratio >= float(high_cardinality_ratio):
            ignored.append(c)
            continue

        # 3) ID numérique “classique” (entier quasi unique + séquentiel)
        if pd.api.types.is_integer_dtype(s) and ratio >= 0.98:
            s2 = s.dropna()
            if len(s2) > 2 and s2.is_monotonic_increasing:
                ignored.append(c)
                continue

        safe_cols.append(c)

    # ✅ Fallback : si tout a été ignoré, on ne crash pas → on garde tout
    if not safe_cols:
        safe_cols = base_cols[:]  # on garde tout
        fallback_note = (
            "⚠️ Toutes les colonnes X candidates ont été filtrées par les heuristiques anti-leakage. "
            "Fallback: on garde toutes les colonnes candidates. "
            "Si besoin, choisis X manuellement."
        )
    # Optional sampling for speed (after leakage guard)
    if len(work) > sample_max:
        work = work.sample(n=sample_max, random_state=random_state)

    y = work[y_col]
    X = work[safe_cols]

    # --- auto task check (prevents 'Unknown label type: continuous') ---
    task_in = (task or "classification").lower()
    task_used = _infer_task_from_y(y, fallback=task_in)
    note = ""
    if task_in == "classification" and task_used == "regression":
        note = (
            "Y ressemble à une cible continue (régression). "
            "La suggestion a utilisé Mutual Information (régression) pour éviter une erreur."
        )
    elif task_in == "regression" and task_used == "classification":
        note = (
            "Y ressemble à une cible discrète (classification). "
            "La suggestion a utilisé Mutual Information (classification)."
        )

    if fallback_note:
        note = (note + "\n" if note else "") + fallback_note

    # Identify categorical vs numeric
    num_cols = [c for c in safe_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in safe_cols if c not in num_cols]

    # Fill numeric NaNs with median
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # One-hot encode categoricals
    if cat_cols:
        X_enc = pd.get_dummies(
            X,
            columns=cat_cols,
            dummy_na=True,
            prefix_sep="__",
        )
    else:
        X_enc = X.copy()

    X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Prepare y and compute MI
    if task_used == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y_enc, _ = pd.factorize(y.astype(str))
        else:
            y_enc = y.values
        mi = mutual_info_classif(X_enc, y_enc, random_state=random_state)
    else:
        y_num = pd.to_numeric(y, errors="coerce")
        y_num = y_num.fillna(y_num.median() if y_num.notna().any() else 0)
        mi = mutual_info_regression(X_enc, y_num.values, random_state=random_state)

    mi = np.asarray(mi, dtype=float)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

    enc_cols = list(X_enc.columns)
    per_enc = pd.DataFrame({"encoded_feature": enc_cols, "mi_score": mi})

    def base_feature(name: str) -> str:
        return name.split("__", 1)[0] if "__" in name else name

    per_enc["feature"] = per_enc["encoded_feature"].apply(base_feature)

    agg = (
        per_enc.groupby("feature", as_index=False)
        .agg(mi_score=("mi_score", "sum"), n_encoded=("encoded_feature", "count"))
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    top_k = max(1, min(int(top_k), len(agg)))
    suggested = agg.head(top_k)["feature"].tolist()

    return FeatureSuggestResult(
        suggested=suggested,
        report=agg,
        ignored=ignored,     # ✅ compat
        task_used=task_used,
        note=note,
    )
