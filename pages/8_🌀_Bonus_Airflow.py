# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import streamlit as st

from src.ui.utils.ui import inject_css, card
from src.config.dataset_config import load_dataset_config

st.set_page_config(page_title="SAKIGA DS — Bonus Airflow", layout="wide")
inject_css()

st.title("🌀 Bonus — Airflow (orchestration)")
card(
    "Objectif : ",
    "Mettre en place un **DAG** qui automatise : préparation → entraînement → évaluation → **déploiement conditionnel**.",
    "Ici on le fait via **Docker** (le plus simple sous Windows).",
)

ROOT = Path.cwd()
COMPOSE = ROOT / "bonus_stack" / "docker-compose.bonus.yml"
DAG_FILE = ROOT / "bonus_stack" / "airflow" / "dags" / "sakiga_ds_pipeline.py"

cfg = load_dataset_config()
raw = Path(cfg.raw_csv)

c1, c2, c3 = st.columns(3)
c1.metric("Dataset", raw.name if raw.exists() else "—")
c2.metric("Cible Y", cfg.target_col or "—")
c3.metric("Task", cfg.task)

def run_cmd(cmd: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, shell=False)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return int(p.returncode), out.strip()
    except Exception as e:
        return 1, f"Erreur exécution: {e}"

def docker_compose_cmd(args: list[str]) -> Tuple[int, str]:
    # try modern: docker compose
    if shutil.which("docker"):
        code, out = run_cmd(["docker", "compose", "-f", str(COMPOSE), *args], cwd=COMPOSE.parent)
        if code == 0 or "unknown command" not in out.lower():
            return code, out
    # fallback: docker-compose
    if shutil.which("docker-compose"):
        return run_cmd(["docker-compose", "-f", str(COMPOSE), *args], cwd=COMPOSE.parent)
    return 1, "Docker/Docker Compose non détecté. Installe Docker Desktop."

docker_ok = shutil.which("docker") is not None or shutil.which("docker-compose") is not None
st.write("Statut Docker :", "✅ OK" if docker_ok else "❌ manquant")

with st.expander("📄 Voir le DAG (sakiga_ds_pipeline.py)", expanded=False):
    if DAG_FILE.exists():
        st.code(DAG_FILE.read_text(encoding="utf-8", errors="ignore"), language="python")
    else:
        st.error("DAG introuvable. Vérifie bonus_stack/airflow/dags/")

st.divider()
st.subheader("Contrôles (depuis Streamlit)")

colA, colB, colC, colD = st.columns(4)

with colA:
    if st.button("▶️ Start Airflow", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["up", "-d", "airflow"])
        st.code(out or f"exit={code}")

with colB:
    if st.button("⏹️ Stop Airflow", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["stop", "airflow"])
        st.code(out or f"exit={code}")

with colC:
    if st.button("🔁 Trigger DAG run", disabled=not docker_ok, use_container_width=True):
        # Trigger from inside container
        code, out = docker_compose_cmd(["exec", "-T", "airflow", "airflow", "dags", "trigger", "sakiga_ds_pipeline"])
        st.code(out or f"exit={code}")

with colD:
    try:
        st.link_button("🌐 Ouvrir UI Airflow", "http://localhost:8080/home")
    except Exception:
        st.markdown("[🌐 Ouvrir UI Airflow](http://localhost:8080/home)")

st.divider()
st.subheader("État & logs")

colL, colR = st.columns([0.5, 0.5], gap="large")
with colL:
    if st.button("📌 Status (docker compose ps)", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["ps"])
        st.code(out or f"exit={code}")

with colR:
    if st.button("🧾 Logs Airflow (tail 80)", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["logs", "--tail", "80", "airflow"])
        st.code(out or f"exit={code}")

st.caption(
    "🔐 Airflow Standalone crée un user admin en local. Les identifiants sont affichés dans les logs "
    "(bouton 'Logs Airflow')."
)

st.markdown("### Comment tester cette partie : ")
st.markdown(
    "1) Upload CSV + choix Y dans **📁 Data**\n"
    "2) Ici: **Start Airflow** → ouvrir l'UI\n"
    "3) Dans Airflow: DAG `sakiga_ds_pipeline` → **Trigger** (ou bouton ici)\n"
    "4) Résultats : ça génère `reports/metrics.json`, `reports/plots/` et (si OK) `models/deployed_model.pkl`"
)
