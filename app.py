# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st

from src.ui.utils.ui import inject_css, card
from src.ui.utils.runner import run_pipeline_all

st.set_page_config(page_title="SAKIGA — Data Science Pro App", layout="wide")
inject_css()

st.title("SAKIGA — Data Science Pro App")
st.caption("Application DS complète : data → modèles → évaluation → (optionnel) MLOps → export/DB.")

colA, colB, colC = st.columns(3)
with colA:
    card("Story & contexte", "Une page claire pour expliquer le problème, la cible et l'intérêt du ML.", "Va sur 📘 Story.")
with colB:
    card("Pipeline générique", "On peut charger **n'importe quel CSV**, choisir **Y** et **X**, puis comparer 3 modèles.", "Va sur 📁 Data.")
with colC:
    card("Runs & MLOps léger", "Chaque évaluation génère un snapshot dans `runs/<run_id>` + artefacts (plots/metrics).", "Va sur 🧰 Runs.")

st.divider()

st.subheader("Workflow rapide ")
st.write("Conseil d'utilisation : **Data → Train & Evaluate → Résultats**. On peut aussi lancer un run complet ici.")

left, right = st.columns([0.45, 0.55])
with left:
    if st.button("🚀 Lancer run complet (Prepare → Train → Evaluate)"):
        prog = st.progress(0)
        run_pipeline_all(progress=prog)
        st.success("Run terminé. Va sur 'Résultats'.")

    st.caption("Astuce performance: si on veut un test fluide, on commence par un dataset pas trop gros ou on limite les X.")

with right:
    st.markdown("### Expérimentation rapide de l'appli :")
    st.markdown(
        "- Story (problème, cible, valeur)\n"
        "- Upload CSV + choix X/Y\n"
        "- Comparaison 3 modèles + meilleur\n"
        "- Plots + dossier `runs/<run_id>`\n"
        "- MLOps + Airflow + N8N"
    )
