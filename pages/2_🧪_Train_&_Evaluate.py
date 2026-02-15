# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import traceback

from src.ui.utils.ui import inject_css, card
from src.config.dataset_config import load_dataset_config
from src.config.model_config import (
    load_model_config, save_model_config,
    CLASSIFICATION_MODELS, REGRESSION_MODELS
)

st.set_page_config(page_title="SAKIGA DS - Train & Evaluate", layout="wide")
inject_css()

# --- Page polish (safe)
st.markdown(
    """
<style>
.skg-steps{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  margin: 10px 0 12px 0;
}
.skg-step{
  border: 1px solid rgba(49,51,63,0.14);
  border-radius: 18px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.80);
  box-shadow: 0 10px 26px rgba(0,0,0,0.05);
}
.skg-step .t{ font-weight: 800; margin-bottom: 6px; }
.skg-step .d{ opacity: 0.80; font-size: .92rem; line-height: 1.3; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧪 Train & Evaluate")

card(
    "Ici on lance la pipeline (sans afficher les résultats)",
    "• Prepare Data : génère `data/processed/dataset.csv` + EDA (si activée)\n"
    "• Train : split train/val/test + GridSearchCV + comparaison (max 3 modèles)\n"
    "• Evaluate : plots + metrics + exports\n\n"
    "➡️ Tous les **résultats** (dashboard, plots, metrics, ZIP/PDF, historique) sont dans **📊 Résultats**.",
)

st.markdown(
    """
<div class="skg-steps">
  <div class="skg-step"><div class="t">1) Prepare</div><div class="d">Standardise dataset + EDA (optionnel)</div></div>
  <div class="skg-step"><div class="t">2) Train</div><div class="d">Compare 1 à 3 modèles + meilleur modèle</div></div>
  <div class="skg-step"><div class="t">3) Evaluate</div><div class="d">Génère plots + metrics + artefacts</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Paths
# ----------------------------
REPORTS_DIR = Path("reports")
TRAIN_META = REPORTS_DIR / "train_meta.json"
METRICS_JSON = REPORTS_DIR / "metrics.json"
PLOTS_DIR = REPORTS_DIR / "plots"

EDA_SUMMARY = REPORTS_DIR / "eda_summary.json"
EDA_MISSING = REPORTS_DIR / "eda_missing.csv"
EDA_OUTLIERS = REPORTS_DIR / "eda_outliers.csv"

DATASET_STD = Path("data") / "processed" / "dataset.csv"
CFG_PATH = Path("data") / "config" / "dataset_config.json"


# ----------------------------
# Helpers
# ----------------------------
def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime if p.exists() else None
    except Exception:
        return None


def _run_step(step_title: str, fn, success_hint: str = "") -> bool:
    """
    UX meilleur au clic:
    - status + progress
    - message persisté (session_state)
    """
    box = st.container()
    info = box.empty()
    bar = box.progress(0)

    try:
        info.markdown(f"⏳ **{step_title}** — démarrage…")
        bar.progress(10)

        with st.spinner(step_title + "…"):
            bar.progress(35)
            fn()
            bar.progress(100)

        st.session_state["__last_action"] = {
            "status": "ok",
            "title": step_title,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hint": success_hint,
        }
        info.markdown(f"✅ **{step_title}** — terminé.")
        try:
            st.toast(f"✅ {step_title} terminé")
        except Exception:
            pass
        if success_hint:
            box.success(success_hint)
        return True

    except Exception as e:
        st.session_state["__last_action"] = {
            "status": "error",
            "title": step_title,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hint": str(e),
        }
        info.markdown(f"❌ **{step_title}** — erreur.")
        box.error(str(e))
        with box.expander("Voir le traceback (debug)"):
            st.code(traceback.format_exc())
        return False


# ----------------------------
# Persistent feedback (top)
# ----------------------------
last = st.session_state.get("__last_action")
if last:
    if last.get("status") == "ok":
        st.success(f"{last.get('title')} — OK ({last.get('ts')})")
        if last.get("hint"):
            st.caption(last.get("hint"))
    else:
        st.error(f"{last.get('title')} — ERREUR ({last.get('ts')})")
        if last.get("hint"):
            st.caption(last.get("hint"))


# ----------------------------
# Current config (dataset + models)
# ----------------------------
ds_cfg = load_dataset_config()
task_type = (getattr(ds_cfg, "task", None) or getattr(ds_cfg, "task_type", None) or "classification").lower()

mcfg = load_model_config()
mcfg.task_type = task_type

with st.expander("🔎 Configuration courante (dataset)", expanded=False):
    st.write(f"- Dataset: `{getattr(ds_cfg, 'raw_csv', '')}` (sep=`{getattr(ds_cfg, 'sep', ',')}`)")
    st.write(f"- Task: **{task_type}**")
    st.write(f"- Y: `{getattr(ds_cfg, 'target_col', '')}`")
    fc = getattr(ds_cfg, "feature_cols", []) or []
    st.write(f"- X: {'AUTO' if len(fc)==0 else f'{len(fc)} colonnes'}")
    st.write(f"- drop_cols: {getattr(ds_cfg, 'drop_cols', []) or []}")
    st.write(f"- text_col: `{getattr(ds_cfg, 'text_col', None)}` | id_col: `{getattr(ds_cfg, 'id_col', None)}`")

if not getattr(ds_cfg, "target_col", ""):
    st.warning("⚠️ Tu n’as pas encore choisi **Y**. Va sur la page **📁 Data** et sélectionne ta cible.")

# ✅ Sécurité : si Y pas choisi, on désactive Prepare (évite KeyError dans prepare_data)
disabled_prepare = not bool(getattr(ds_cfg, "target_col", ""))
if disabled_prepare:
    st.info("👈 Choisis d’abord la cible **Y** dans la page **📁 Data** avant de lancer **Prepare**.")


# ----------------------------
# Model selection
# ----------------------------
st.subheader("Choix des modèles (1 à 3)")

if task_type == "regression":
    options = REGRESSION_MODELS
    default = [m for m in (mcfg.selected_models or []) if m in options] or ["ridge", "random_forest", "xgboost"]
else:
    options = CLASSIFICATION_MODELS
    default = [m for m in (mcfg.selected_models or []) if m in options] or ["logreg", "random_forest", "linear_svc"]

pretty = {
    # Classification
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "linear_svc": "Linear SVC",
    "svc": "SVC (RBF)",
    "sgd": "SGDClassifier",
    "naive_bayes": "Naive Bayes (GaussianNB)",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "gradient_boosting": "Gradient Boosting",
    "bagging": "Bagging (Tree)",
    "mlp": "Neural Net (MLP)",
    "xgboost": "XGBoost",

    # Regression
    "ridge": "Ridge",
    "lasso": "Lasso",
    "linear_regression": "Linear Regression",
    "poly_ridge": "Polynomial Regression (Ridge)",
    "knn_reg": "KNN Regressor",
    "adaboost": "AdaBoost Regressor",
    "lightgbm": "LightGBM Regressor",
}

selected = st.multiselect(
    "Modèles à tester (max 3)",
    options=options,
    default=default[:3],
    format_func=lambda x: pretty.get(x, x),
)

too_many = len(selected) > 3
if too_many:
    st.error("Max 3 modèles. Retire-en un.")

st.subheader("Réglages (GridSearch + Cross-Validation)")
cA, cB, cC = st.columns([1, 1, 2])
with cA:
    mcfg.cv_folds = st.number_input(
        "CV folds",
        min_value=2, max_value=10,
        value=int(getattr(mcfg, "cv_folds", 3) or 3),
        step=1
    )
with cB:
    mcfg.n_jobs = st.number_input(
        "n_jobs",
        min_value=-1, max_value=32,
        value=int(getattr(mcfg, "n_jobs", -1)),
        step=1
    )
with cC:
    with st.expander("ℹ️ Rappel : Split vs CV (cours)", expanded=False):
        st.markdown(
            """
- Split **train/val/test** : test = mesure finale.
- GridSearchCV : CV sur train pour choisir hyperparamètres.
- Résultat : meilleur modèle choisi sur val, puis métriques finales sur test.
"""
        )

c_save, c_hint = st.columns([1, 2])
with c_save:
    if st.button("💾 Sauvegarder sélection modèles", disabled=too_many or len(selected) == 0):
        mcfg.selected_models = selected[:3]
        save_model_config(mcfg)
        st.success("OK: config modèles sauvegardée.")
with c_hint:
    st.caption("Conseil : sauvegarde → Prepare → Train → Evaluate → (voir Résultats).")


# ----------------------------
# EDA section (optionnelle)
# ----------------------------
st.divider()
st.subheader("EDA (optionnel, rapide)")

c_eda1, c_eda2 = st.columns([1, 2])
with c_eda1:
    if st.button("📌 Générer EDA + Prepare Data", key="btn_eda_prepare", disabled=disabled_prepare):
        from src.pipeline.prepare_data import main as prepare_main

        _run_step(
            "Prepare Data + EDA",
            prepare_main,
            success_hint="EDA + dataset standardisé générés. Va sur 📊 Résultats pour visualiser.",
        )
with c_eda2:
    if EDA_SUMMARY.exists():
        eda = _read_json(EDA_SUMMARY)
        if eda:
            with st.expander("Voir résumé EDA", expanded=False):
                st.write(f"- Saved: `{eda.get('saved_dataset')}`")
                st.write(f"- Shape: {eda.get('shape')}")
                st.write(f"- Task: **{eda.get('task')}**")
                st.write(f"- Y: `{eda.get('target_col')}` | X: {len(eda.get('x_cols', []))} colonnes")
                st.caption("Fichiers: reports/eda_summary.json, reports/eda_missing.csv, reports/eda_outliers.csv")
        else:
            st.info("EDA trouvée mais lecture impossible.")
    else:
        st.caption("Pas encore de EDA. Clique sur “Générer EDA + Prepare Data” une fois.")

if EDA_MISSING.exists():
    with st.expander("Missing values (top)", expanded=False):
        try:
            miss = pd.read_csv(EDA_MISSING).head(15)
            st.dataframe(miss, use_container_width=True)
        except Exception:
            st.caption("Impossible de lire reports/eda_missing.csv")

if EDA_OUTLIERS.exists():
    with st.expander("Outliers (IQR) (top)", expanded=False):
        try:
            outl = pd.read_csv(EDA_OUTLIERS).head(15)
            st.dataframe(outl, use_container_width=True)
        except Exception:
            st.caption("Impossible de lire reports/eda_outliers.csv")


# ----------------------------
# Pipeline buttons (Prepare / Train / Evaluate)
# ----------------------------
st.divider()
st.subheader("Lancer la pipeline")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("1) Prepare Data", key="btn_prepare_only", disabled=disabled_prepare):
        from src.pipeline.prepare_data import main as prepare_main

        _run_step(
            "Prepare Data",
            prepare_main,
            success_hint="Dataset prêt. Prochaine étape: Train (ou voir EDA dans 📊 Résultats).",
        )
with c2:
    disabled_train = disabled_prepare or too_many or len(selected) == 0
    if st.button("2) Train", disabled=disabled_train, key="btn_train"):
        mcfg.selected_models = selected[:3]
        save_model_config(mcfg)

        from src.pipeline.train import main as train_main

        ok = _run_step(
            "Train (GridSearchCV + comparaison)",
            train_main,
            success_hint="Train terminé. Prochaine étape: Evaluate, puis dashboard dans 📊 Résultats.",
        )
with c3:
    disabled_eval = not TRAIN_META.exists()
    if st.button("3) Evaluate + Plots", disabled=disabled_eval, key="btn_eval"):
        from src.pipeline.evaluate import main as eval_main

        ok = _run_step(
            "Evaluate (plots + metrics)",
            eval_main,
            success_hint="Evaluate terminé. Ouvre maintenant 📊 Résultats pour voir tout.",
        )
card(
    "👉 Où voir les résultats ?",
    "Tout est centralisé dans **📊 Résultats** :\n"
    "• Dashboard (KPIs, comparaison modèles)\n"
    "• Plots (confusion, ROC, PR, …)\n"
    "• metrics.json + predictions_test.csv\n"
    "• Téléchargements ZIP/PDF\n"
    "• Historique des runs",
)


# ----------------------------
# Quick status (anti confusion)
# ----------------------------
st.divider()
st.subheader("État des artefacts (anti-confusion)")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("dataset.csv", "OK" if DATASET_STD.exists() else "Absent")
with k2:
    st.metric("train_meta.json", "OK" if TRAIN_META.exists() else "Absent")
with k3:
    st.metric("metrics.json", "OK" if METRICS_JSON.exists() else "Absent")
with k4:
    plots_count = len(list(PLOTS_DIR.glob("*.png"))) + len(list(PLOTS_DIR.glob("*.jpg"))) + len(list(PLOTS_DIR.glob("*.jpeg"))) if PLOTS_DIR.exists() else 0
    st.metric("plots", plots_count)

with st.expander("Détails (timestamps)", expanded=False):
    st.write(f"- dataset_config.json : {_fmt_ts(_mtime(CFG_PATH))}")
    st.write(f"- data/processed/dataset.csv : {_fmt_ts(_mtime(DATASET_STD))}")
    st.write(f"- reports/train_meta.json : {_fmt_ts(_mtime(TRAIN_META))}")
    st.write(f"- reports/metrics.json : {_fmt_ts(_mtime(METRICS_JSON))}")
    st.write(f"- reports/plots/ : {_fmt_ts(_mtime(PLOTS_DIR))}")
