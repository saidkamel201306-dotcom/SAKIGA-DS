# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Optional, Dict

import pandas as pd
import streamlit as st

from src.ui.utils.ui import inject_css, card

st.set_page_config(page_title="SAKIGA DS - Runs", layout="wide")
inject_css()

st.title("🧾 Runs — historique (bonus)")
st.caption("Lisible, pro : KPIs + plots + export ZIP.")

RUNS_DIR = Path("runs")

# --- Streamlit compatibility helpers (older/newer versions) ---
def _st_image_auto(path: str, caption: str | None = None):
    """Display image with a parameter compatible across Streamlit versions."""
    try:
        # Newer Streamlit (may be deprecated but still works)
        st.image(path, caption=caption, use_container_width=True)
        return
    except TypeError:
        pass
    try:
        # Older Streamlit
        st.image(path, caption=caption, use_column_width=True)
        return
    except TypeError:
        pass
    # Fallback
    st.image(path, caption=caption)


def _st_dataframe_auto(df, **kwargs):
    """Display dataframe full width across Streamlit versions."""
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        # Very old versions may not support use_container_width
        st.dataframe(df, **kwargs)


def _read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _find(p1: Path, p2: Path) -> Optional[Path]:
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def _zip_dir_bytes(root_dir: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in root_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(root_dir)))
    bio.seek(0)
    return bio.read()


def _kpi_from_metrics(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    test = (metrics or {}).get("test_default") or {}
    if not test and isinstance(metrics, dict):
        test = metrics

    if task == "regression":
        return {
            "primary": test.get("r2", None),
            "r2": test.get("r2", None),
            "mae": test.get("mae", None),
            "rmse": test.get("rmse", None),
        }
    return {
        "primary": test.get("f1_macro", None),
        "accuracy": test.get("accuracy", None),
        "f1_macro": test.get("f1_macro", None),
        "roc_auc": (metrics or {}).get("roc_auc", test.get("roc_auc", None)),
    }


card(
    "Comment ça marche",
    "- Après **Evaluate**, un dossier est créé dans `runs/<run_id>/`.\n"
    "- Contenu : `models/`, `reports/metrics.json`, `reports/train_meta.json`, `reports/plots/`.\n"
    "- Ici : choix du run + KPIs + plots + ZIP export.",
    "Si tu ne vois rien : fais **Train**, puis **Evaluate**."
     "Mlflow permet aussi d'avoir un historique dans la page MLOps."
)

if not RUNS_DIR.exists():
    st.info("Aucun dossier `runs/` trouvé pour le moment.")
    st.stop()

runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
if not runs:
    st.info("Aucun run enregistré. Lance Train puis Evaluate.")
    st.stop()

# -------- Scoreboard (top runs)
rows = []
for d in runs[:50]:
    metrics_path = _find(d / "reports" / "metrics.json", d / "metrics.json")
    train_meta_path = _find(d / "reports" / "train_meta.json", d / "train_meta.json")

    metrics = _read_json(metrics_path) if metrics_path else None
    train_meta = _read_json(train_meta_path) if train_meta_path else None

    task = str((metrics or {}).get("task") or (train_meta or {}).get("task") or "classification").lower()
    kpi = _kpi_from_metrics(task, metrics or {})
    rows.append({
        "run_id": d.name,
        "task": task,
        "primary": kpi.get("primary"),
        "accuracy": kpi.get("accuracy"),
        "f1_macro": kpi.get("f1_macro"),
        "roc_auc": kpi.get("roc_auc"),
        "r2": kpi.get("r2"),
        "mae": kpi.get("mae"),
    })

df_score = pd.DataFrame(rows)
with st.expander("🏁 Scoreboard (meilleurs runs)", expanded=False):
    _st_dataframe_auto(df_score, height=260)

st.divider()

# -------- Viewer
run_names = [d.name for d in runs]
choice = st.selectbox("Choisir un run", run_names, index=0)
run_dir = RUNS_DIR / choice

meta_root = _read_json(run_dir / "meta.json") or {}
metrics_path = _find(run_dir / "reports" / "metrics.json", run_dir / "metrics.json")
train_meta_path = _find(run_dir / "reports" / "train_meta.json", run_dir / "train_meta.json")

plots_dir = run_dir / "reports" / "plots"
if not plots_dir.exists():
    plots_dir = run_dir / "plots"

metrics = _read_json(metrics_path) if metrics_path else None
train_meta = _read_json(train_meta_path) if train_meta_path else None

task = str((metrics or {}).get("task") or (train_meta or {}).get("task") or "classification").lower()

st.subheader(f"Run: {choice}")

colA, colB, colC = st.columns([0.45, 0.30, 0.25], gap="large")
with colA:
    with st.expander("Meta (root)", expanded=False):
        st.json(meta_root, expanded=False)
with colB:
    st.write("**KPIs**")
    k = _kpi_from_metrics(task, metrics or {})
    if task == "regression":
        st.metric("R² (test)", f"{k.get('r2', '-')}")
        st.metric("MAE (test)", f"{k.get('mae', '-')}")
    else:
        st.metric("F1-macro (test)", f"{k.get('f1_macro', '-')}")
        st.metric("Accuracy (test)", f"{k.get('accuracy', '-')}")
with colC:
    st.write("**Export**")
    if st.button("Préparer ZIP (run sélectionné)"):
        st.session_state["_runs_zip"] = _zip_dir_bytes(run_dir)
        st.session_state["_runs_zip_name"] = f"{choice}.zip"
        st.success("ZIP prêt ✅")

    if "_runs_zip" in st.session_state and st.session_state.get("_runs_zip_name") == f"{choice}.zip":
        st.download_button(
            "⬇️ Télécharger ZIP",
            data=st.session_state["_runs_zip"],
            file_name=st.session_state["_runs_zip_name"],
            mime="application/zip",
        )

st.markdown("---")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("metrics.json")
    if metrics:
        st.json(metrics, expanded=False)
    else:
        st.caption("metrics.json introuvable pour ce run.")
with col2:
    st.subheader("train_meta.json")
    if train_meta:
        st.json(train_meta, expanded=False)
    else:
        st.caption("train_meta.json introuvable pour ce run.")

st.subheader("Plots")
if plots_dir.exists():
    imgs = sorted([p for p in plots_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if imgs:
        cols = st.columns(2)
        for i, p in enumerate(imgs):
            with cols[i % 2]:
                _st_image_auto(str(p), caption=p.name)
    else:
        st.caption("Aucun plot trouvé.")
else:
    st.caption("Dossier plots introuvable.")
