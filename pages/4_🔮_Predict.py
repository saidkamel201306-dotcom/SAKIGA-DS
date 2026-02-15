# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.ui.utils.ui import inject_css, card


# ============================================================
# ✅ COMPATIBILITÉ PICKLE / JOBLIB (anciens modèles)
# Problème: le modèle picklé référence join_text_cols dans "main"
# Streamlit charge les pages comme __main__ => attribut introuvable.
# Solution: définir join_text_cols ici + l'exposer sur le module "main".
# ============================================================
def join_text_cols(X, cols=None, sep=" ", **kwargs):
    """
    Join multiple text/categorical columns into a single string per row.
    Compatible with sklearn FunctionTransformer / pipelines.

    - X peut être un DataFrame, une Series, ou un array-like.
    - cols peut être une liste de colonnes (DataFrame) ou d'indices (array).
    """
    # DataFrame
    if isinstance(X, pd.DataFrame):
        df = X
        if cols:
            df = df[cols]
        df = df.fillna("").astype(str)
        return df.agg(lambda r: sep.join([v for v in r.values if v is not None]), axis=1).to_numpy()

    # Series
    if isinstance(X, pd.Series):
        s = X.fillna("").astype(str)
        return s.to_numpy()

    # array-like
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.astype(str)

    if cols:
        arr = arr[:, cols]

    out = []
    for row in arr:
        out.append(sep.join([str(v) if v is not None else "" for v in row]))
    return np.array(out)


# Exposer la fonction sous le module "main" (important pour joblib/pickle)
try:
    sys.modules.get("main", sys.modules[__name__]).join_text_cols = join_text_cols
except Exception:
    pass


def _load_train_meta() -> dict:
    p = Path("reports") / "train_meta.json"
    if not p.exists():
        raise FileNotFoundError("reports/train_meta.json introuvable. Lance d'abord Train & Evaluate.")
    meta = json.loads(p.read_text(encoding="utf-8"))

    # Normalize label_mapping keys (JSON turns int keys into strings)
    lm = meta.get("label_mapping")
    if isinstance(lm, dict):
        lm2 = {}
        for k, v in lm.items():
            try:
                ik = int(k)
            except Exception:
                ik = k
            lm2[ik] = v
        meta["label_mapping"] = lm2

    return meta


def _load_model():
    # ✅ BONUS AIRFLOW: si un modèle "déployé" existe, on l'utilise.
    best = Path("models") / "best_model.pkl"
    deployed = Path("models") / "deployed_model.pkl"
    p = deployed if deployed.exists() else best

    if not p.exists():
        raise FileNotFoundError(
            "Aucun modèle trouvé. Lance d'abord Train & Evaluate (ou Airflow).\n"
            "Attendu: models/best_model.pkl (ou models/deployed_model.pkl)."
        )

    try:
        return joblib.load(p), p
    except Exception as e:
        # Message plus clair en démo si un autre attribut manque
        raise RuntimeError(
            f"Impossible de charger le modèle: {p}\n"
            f"Erreur: {e}\n\n"
            "➡️ Cause fréquente: le modèle a été sauvegardé avec une fonction personnalisée\n"
            "(ex: join_text_cols) non disponible au chargement.\n"
            "✅ Correctif: ré-entraîner le modèle après avoir déplacé les fonctions perso dans un module stable (src/...)."
        )


def _safe_read_csv(file, sep: str):
    try:
        return pd.read_csv(file, sep=sep)
    except Exception:
        for s in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(file, sep=s)
            except Exception:
                continue
        raise


def _has_proba(model) -> bool:
    return hasattr(model, "predict_proba")


def _predict_with_threshold(model, X: pd.DataFrame, threshold: float, label_mapping: dict | None):
    """
    - If binary classification with predict_proba: apply threshold on proba[:,1]
      and return (pred, proba_matrix)
    - If multi-class with predict_proba: return (pred, proba_matrix) using argmax
    - Else: fallback to model.predict and return (pred, None)
    """
    if _has_proba(model):
        proba = model.predict_proba(X)
        # Binary
        if hasattr(proba, "shape") and proba.ndim == 2 and proba.shape[1] == 2:
            p1 = proba[:, 1]
            pred01 = (p1 >= threshold).astype(int)
            if label_mapping:
                pred = [label_mapping.get(int(i), str(i)) for i in pred01]
            else:
                pred = pred01
            return pred, proba

        # Multi-class
        if hasattr(proba, "shape") and proba.ndim == 2 and proba.shape[1] >= 3:
            idx = np.argmax(proba, axis=1)
            classes = getattr(model, "classes_", list(range(proba.shape[1])))
            raw_pred = [classes[i] for i in idx]
            if label_mapping:
                pred = [label_mapping.get(int(i), str(i)) for i in raw_pred]
            else:
                pred = raw_pred
            return pred, proba

    # fallback
    pred = model.predict(X)
    if label_mapping:
        pred = [label_mapping.get(int(i), str(i)) for i in pred]
    return pred, None


def _format_class_label(raw, label_mapping: dict | None) -> str:
    # raw can be int/str
    try:
        if label_mapping is not None:
            k = int(raw)
            return str(label_mapping.get(k, label_mapping.get(str(k), raw)))
    except Exception:
        pass
    return str(raw)


def _proba_table(model, proba_row: np.ndarray, label_mapping: dict | None) -> pd.DataFrame:
    classes = getattr(model, "classes_", list(range(len(proba_row))))
    rows = []
    for i, p in enumerate(proba_row):
        rows.append({
            "classe": _format_class_label(classes[i], label_mapping),
            "proba": float(p),
        })
    dfp = pd.DataFrame(rows).sort_values("proba", ascending=False).reset_index(drop=True)
    return dfp


st.set_page_config(page_title="SAKIGA DS - Predict", layout="wide")
inject_css()

st.title("🔮 Predict — saisie / fichier → prédiction")

card(
    "Objectif",
    "Appliquer le **meilleur modèle** entraîné : soit sur un **nouveau CSV**, soit via une **saisie manuelle** (1 ligne).",
    "Important : la prédiction utilise uniquement les colonnes **X** enregistrées pendant l'entraînement."
)

try:
    meta = _load_train_meta()
    model, model_path = _load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

task = meta.get("task", "classification")
x_cols = meta.get("x_cols", [])
id_col = meta.get("id_col", None)
target_col = meta.get("target_col", None)
best_name = meta.get("best_model", "N/A")
label_mapping = meta.get("label_mapping", None)

left, right = st.columns([0.62, 0.38], gap="large")

with right:
    st.subheader("Modèle chargé")
    st.write(f"- Type: **{task}**")
    st.write(f"- Best model: **{best_name}**")
    st.write(f"- X attendus: **{len(x_cols)}** colonnes")
    st.caption(f"Fichier modèle utilisé : `{model_path}`")  # ✅ visible en démo

    if target_col:
        st.caption(f"Y (train) : `{target_col}`")
    if id_col:
        st.caption(f"ID (train) : `{id_col}`")
    if label_mapping:
        st.caption(f"Label mapping : {label_mapping}")

    with st.expander("ℹ️ Rappel (cours) : proba / seuil / AUC", expanded=False):
        st.markdown(
            """
- Certains modèles donnent une **probabilité** `P(classe=1)`.
- Par défaut, on classe souvent avec un **seuil 0.5**.
- Si tu changes le seuil :
  - seuil ↓ : plus de **positifs** (moins de FN) mais plus de FP
  - seuil ↑ : moins de FP mais plus de FN
- **AUC (ROC-AUC)** mesure la qualité **sur tous les seuils** (plus grand = mieux).
"""
        )

    with st.expander("Voir les X attendus", expanded=False):
        st.code(", ".join(map(str, x_cols)) if x_cols else "(vide)")

mode = st.radio("Mode", ["📄 Nouveau CSV", "🧍 Saisie manuelle (1 ligne)"], horizontal=True)

# Threshold UI (only useful for binary classification with proba)
threshold = 0.5
show_threshold = (task == "classification") and _has_proba(model)
if show_threshold:
    threshold = st.slider("Seuil de décision (si proba dispo)", 0.0, 1.0, 0.5, 0.01)

# ---------- MODE 1: CSV ----------
if mode == "📄 Nouveau CSV":
    with left:
        st.subheader("1) Charger un CSV à prédire")
        uploaded = st.file_uploader("CSV (nouveau fichier)", type=["csv"], key="predict_csv")
        sep = st.selectbox("Séparateur (si besoin)", options=[",", ";", "\t", "|"], index=0, key="predict_sep")

        add_proba = st.checkbox("Ajouter proba (si dispo)", value=True)
        include_inputs = st.checkbox("Inclure les colonnes d'entrée dans l'export", value=True)

        if uploaded is None:
            st.info("Charge un CSV pour générer des prédictions.")
            st.stop()

        try:
            df_in = _safe_read_csv(uploaded, sep)
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            st.stop()

        st.write("Aperçu du fichier:")
        st.dataframe(df_in.head(30), use_container_width=True)

        if not x_cols:
            st.error("Aucune liste X trouvée dans train_meta.json. Relance Train.")
            st.stop()

        missing = [c for c in x_cols if c not in df_in.columns]
        if missing:
            st.error("Ton CSV n'a pas toutes les colonnes X attendues.")
            st.write("Colonnes manquantes:", missing)
            st.stop()

        Xp = df_in[x_cols].copy()

        if st.button("🚀 Lancer la prédiction"):
            pred, proba = _predict_with_threshold(model, Xp, threshold, label_mapping)
            out = pd.DataFrame({"prediction": pred})

            # Probabilities export (if available)
            if add_proba and proba is not None:
                n_classes = int(proba.shape[1]) if hasattr(proba, 'shape') else 0
                classes = getattr(model, 'classes_', list(range(n_classes)))
                if n_classes <= 10:
                    for j in range(n_classes):
                        lbl = _format_class_label(classes[j], label_mapping)
                        out[f"proba_{lbl}"] = proba[:, j]
                else:
                    # Too many classes: keep top-1 only
                    top_idx = np.argmax(proba, axis=1)
                    top_p = np.max(proba, axis=1)
                    top_lbl = [_format_class_label(classes[i], label_mapping) for i in top_idx]
                    out['top1_classe'] = top_lbl
                    out['top1_proba'] = top_p

            if include_inputs:
                out = pd.concat([df_in.reset_index(drop=True), out], axis=1)
            else:
                out = pd.concat([out], axis=1)

            st.success("✅ Prédictions générées.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger predictions.csv", data=csv_bytes, file_name="predictions.csv", mime="text/csv")


# ---------- MODE 2: Manual entry ----------
else:
    with left:
        st.subheader("1) Saisie manuelle (1 ligne)")

        if not x_cols:
            st.error("Aucune liste X trouvée dans train_meta.json. Relance Train.")
            st.stop()

        # load processed dataset to infer categories/defaults
        data_path = Path("data") / "processed" / "dataset.csv"
        ref_df = None
        if data_path.exists():
            try:
                ref_df = pd.read_csv(data_path)
            except Exception:
                ref_df = None

        st.caption("On génère un formulaire à partir des colonnes X de l'entraînement.")

        # Choix du mode de saisie pour les colonnes catégorielles/texte
        cat_mode = st.radio(
            "Colonnes texte/catégorie : mode de saisie",
            ["Liste + Autre (recommandé)", "Saisie libre (texte)"],
            horizontal=True,
            index=0,
            key="cat_mode"
        )

        values = {}
        n_cols = 2
        grid = st.columns(n_cols)
        if cat_mode == "Liste + Autre (recommandé)":
            st.caption("💡 Pour les colonnes texte : tu peux choisir dans la liste, ou sélectionner 'Autre…' pour écrire.")

        for i, col in enumerate(x_cols):
            with grid[i % n_cols]:
                if ref_df is not None and col in ref_df.columns:
                    s = ref_df[col]
                    if pd.api.types.is_numeric_dtype(s):
                        med = float(pd.to_numeric(s, errors="coerce").median() if s.notna().any() else 0.0)
                        values[col] = st.number_input(f"{col}", value=med)
                    else:
                        # Texte/Catégorie : liste courte si dispo, sinon saisie libre
                        vc = s.astype(str).replace("nan", "").value_counts().head(30)
                        opts = [o for o in vc.index.tolist() if o != ""]

                        if cat_mode == "Saisie libre (texte)" or not opts:
                            values[col] = st.text_input(f"{col}", value="")
                        else:
                            # Mode liste + possibilité d'écrire une valeur non présente
                            options = opts + ["✍️ Autre…"]
                            choice = st.selectbox(f"{col}", options=options, index=0)
                            if choice == "✍️ Autre…":
                                values[col] = st.text_input(f"{col} (autre)", value="")
                            else:
                                values[col] = choice
                else:
                    # fallback: text input
                    values[col] = st.text_input(f"{col}", value="")

        X_one = pd.DataFrame([values], columns=x_cols)

        st.write("Aperçu de la ligne:")
        st.dataframe(X_one, use_container_width=True)

        if st.button("🚀 Prédire (1 ligne)"):
            pred, proba = _predict_with_threshold(model, X_one, threshold, label_mapping)
            st.success("✅ Prédiction faite.")

            # --- Clear result ---
            pred_val = pred[0] if isinstance(pred, (list, np.ndarray, pd.Series)) else pred
            st.markdown(f"### 🎯 Classe prédite : **{pred_val}**")

            if proba is not None:
                dfp = _proba_table(model, proba[0], label_mapping)
                st.write("Probabilités par classe :")
                st.dataframe(dfp, use_container_width=True, hide_index=True)

                # Binary explanation with threshold
                if hasattr(proba, 'shape') and proba.shape[1] == 2:
                    p0 = float(proba[0, 0])
                    p1 = float(proba[0, 1])
                    st.info(f"Seuil = **{threshold:.2f}** → P(classe 1) = **{p1:.3f}** et P(classe 0) = **{p0:.3f}**")
                    decision = "classe 1" if p1 >= threshold else "classe 0"
                    st.caption(f"Décision : comme P(classe 1) {'≥' if p1 >= threshold else '<'} seuil, la prédiction finale est **{decision}**.")
            else:
                st.caption("(Le modèle ne fournit pas de probabilités : affichage en label uniquement.)")
