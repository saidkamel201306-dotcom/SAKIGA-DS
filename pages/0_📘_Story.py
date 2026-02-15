# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.utils.ui import inject_css, card
from src.config.dataset_config import load_dataset_config


st.set_page_config(page_title="SAKIGA DS - Story", layout="wide")
inject_css()

# ---- Page-specific CSS (clean, minimal, pro)
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

/* Hero */
.skg-hero{
  border-radius: 18px;
  padding: 18px 18px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(135deg, rgba(0,0,0,0.015), rgba(0,0,0,0.03));
}
.skg-hero h1{ margin:0; line-height: 1.05; }
.skg-hero p{ margin:0.40rem 0 0 0; opacity: 0.86; }

/* KPI grid */
.skg-kpis{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 12px;
}
.skg-kpi{
  border-radius: 14px;
  padding: 12px 12px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(0,0,0,0.02);
}
.skg-kpi .label{ font-size: 0.82rem; opacity: 0.75; margin-bottom: 4px; }
.skg-kpi .value{ font-size: 1.10rem; font-weight: 750; line-height: 1.15; }

/* Pills */
.skg-pill-row { margin-top: 10px; display:flex; flex-wrap:wrap; gap:8px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Load config + dataset snapshot
# =========================
cfg = load_dataset_config()

raw_csv = getattr(cfg, "raw_csv", "") or ""
train_csv = getattr(cfg, "raw_train_csv", "") or getattr(cfg, "train_csv", "") or ""
test_csv = getattr(cfg, "raw_test_csv", "") or getattr(cfg, "test_csv", "") or ""

raw_path = Path(raw_csv) if raw_csv else None
train_path = Path(train_csv) if train_csv else None
test_path = Path(test_csv) if test_csv else None

if train_path and test_path:
    data_mode = "Train/Test"
    main_path = train_path
else:
    data_mode = "CSV unique"
    main_path = raw_path

target_col = getattr(cfg, "target_col", "") or ""
sep = getattr(cfg, "sep", ",") or ","
task = getattr(cfg, "task", "") or "classification"

rows = None
cols = None
file_name = None
read_error = None

if main_path and main_path.exists():
    file_name = main_path.name
    try:
        # Snapshot léger (évite de bloquer pour gros CSV)
        df_head = pd.read_csv(main_path, sep=sep, on_bad_lines="skip", nrows=5000)
        rows = "≥ 5 000" if len(df_head) == 5000 else f"{len(df_head):,}".replace(",", " ")
        cols = str(int(df_head.shape[1]))
    except Exception as e:
        read_error = str(e)
else:
    file_name = None

# =========================
# HERO
# =========================
st.markdown(
    f"""
<div class="skg-hero">
  <h1>📘 Story — Problème, cible, valeur</h1>
  <p>
    Application ML prête à l’emploi (classification / régression), réutilisable sur plusieurs datasets
    (CSV unique ou Train/Test), avec pipeline reproductible et historique des runs.
  </p>
  <div class="skg-pill-row">
    <span class="sakiga-pill">✅ Dataset-agnostic</span>
    <span class="sakiga-pill">✅ Auto ou manuel X</span>
    <span class="sakiga-pill">✅ 3 modèles comparés</span>
    <span class="sakiga-pill">✅ Meilleur modèle sauvegardé</span>
    <span class="sakiga-pill">✅ Runs traçables</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# =========================
# Top layout
# =========================
left, right = st.columns([0.62, 0.38], gap="large")

with left:
    card(
        "Contexte (d’où vient l’idée)",
        "Au départ, le travail se faisait via plusieurs notebooks (souvent un par dataset).\n"
        "Ensuite, un projet e-commerce (dataset Kaggle) a mis en évidence un besoin :\n"
        "avoir une solution unique et réutilisable, capable de s’adapter à différents datasets.\n\n"
        "Objectif : une application ML unifiée, utilisable en classification ou régression, en mode CSV unique ou Train/Test.",
    )

    card(
        "Objectif / Cible / Valeur",
        "• Objectif : prédire une variable cible **Y** à partir de variables explicatives **X**.\n"
        "• Cible (Y) : sélectionnée dans la page **Data**.\n"
        "• Valeur : pipeline complet et reproductible (prétraitement → split → entraînement), "
        "comparaison de modèles, sélection du meilleur et sauvegarde des runs.\n\n"
        "Résultat : une application claire, exploitable et multi datasets & models.",
    )

with right:
    st.subheader("Dataset actif")
    st.caption("Aperçu basé sur la configuration définie dans la page Data.")

    mode_val = data_mode
    file_val = file_name or "—"
    rows_val = rows or "—"
    cols_val = cols or "—"
    y_val = target_col if target_col else "—"
    task_val = task

    st.markdown(
        f"""
<div class="skg-kpis">
  <div class="skg-kpi"><div class="label">Mode données</div><div class="value">{mode_val}</div></div>
  <div class="skg-kpi"><div class="label">Type</div><div class="value">{task_val}</div></div>
  <div class="skg-kpi"><div class="label">Cible (Y)</div><div class="value">{y_val}</div></div>
  <div class="skg-kpi"><div class="label">Taille (colonnes / lignes)</div><div class="value">{cols_val} / {rows_val}</div></div>
  <div class="skg-kpi"><div class="label">Fichier</div><div class="value">{file_val}</div></div>
  <div class="skg-kpi"><div class="label">Config</div><div class="value">dataset_config.json</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if read_error:
        st.info("Lecture partielle impossible : vérifier séparateur et encodage.")
        with st.expander("Détail"):
            st.code(read_error)

    if not file_name:
        st.info("Aucun dataset chargé : aller dans **Data** pour uploader un CSV.")

st.divider()

# =========================
# Requirements section
# =========================
st.subheader("Ce que le projet démontre (exigences couvertes)")

c1, c2 = st.columns(2, gap="large")

with c1:
    card(
        "Bloc Data + Préparation",
        "• Définition du problème et de la cible Y\n"
        "• Analyse exploratoire (EDA)\n"
        "• Prétraitement (missing, encodage, scaling, etc.)\n"
        "• Split train / validation / test",
    )

with c2:
    card(
        "Bloc Modélisation + Application",
        "• Comparaison de 3 modèles\n"
        "• Choix du meilleur modèle via métriques\n"
        "• Interface de prédiction (saisie ou fichier)\n"
        "• Historique & traçabilité des runs",
    )

st.divider()

# =========================
# Bonus MLOps / automation
# =========================
st.subheader("Bonus — Automatisation & MLOps (version projet)")

b1, b2 = st.columns(2, gap="large")

with b1:
    card(
        "Pipeline reproductible",
        "• Exécution structurée des étapes (Prepare → Train → Evaluate)\n"
        "• Paramètres centralisés (dataset, X, Y, split, modèles)\n"
        "• Résultats cohérents à configuration égale",
    )

with b2:
    card(
        "Traçabilité & historique",
        "• Sauvegarde de chaque run (config, métriques, artefacts)\n"
        "• Comparaison rapide entre runs\n"
        "• Export des livrables (modèle, rapports, plots)",
    )

st.divider()

# =========================
# How it works
# =========================
st.subheader("Parcours d’utilisation (scénario)")

s1, s2, s3 = st.columns(3, gap="large")

with s1:
    card(
        "1) Data",
        "Charger un dataset, sélectionner la cible Y, définir les variables X (Auto ou Manuel).",
    )

with s2:
    card(
        "2) EDA + Prepare",
        "Contrôler la qualité des données, appliquer le prétraitement et préparer le dataset pour l’entraînement.",
    )

with s3:
    card(
        "3) Train & Evaluate",
        "Comparer les modèles, analyser les métriques, sélectionner le meilleur et sauvegarder le run.",
    )

card(
    "Livrables produits",
    "• Meilleur modèle sauvegardé\n"
    "• Dossier de run : configuration, métriques, artefacts (plots), modèle\n"
    "• Interface de prédiction sur de nouvelles données",
)

st.caption("Cette page résume l’objectif, la valeur et l’automatisation du projet")
