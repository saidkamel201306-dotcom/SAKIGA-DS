# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st

from src.pipeline.prepare_data import main as prepare_data_main
from src.pipeline.train import main as train_main
from src.pipeline.evaluate import main as evaluate_main

def run_pipeline_all(progress=None):
    """Run the generic pipeline (dataset-agnostic): Prepare → Train → Evaluate."""
    steps = [
        ("Préparer les données", prepare_data_main),
        ("Entraîner (GridSearch)", train_main),
        ("Évaluer (metrics + plots)", evaluate_main),
    ]
    for i, (label, fn) in enumerate(steps, start=1):
        st.info(f"⏳ {i}/{len(steps)} — {label}...")
        fn()
        if progress is not None:
            progress.progress(i/len(steps))
    st.success("✅ Pipeline complet terminé.")
