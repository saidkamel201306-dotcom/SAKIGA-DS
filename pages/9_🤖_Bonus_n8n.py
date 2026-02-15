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

st.set_page_config(page_title="SAKIGA DS — Bonus n8n", layout="wide")
inject_css()

st.title("🤖 Bonus — n8n (automatisation)")
card(
    "Objectif : ",
    "Déclencher automatiquement des workflows MLOps (webhooks) + notifications après entraînement/déploiement.",
    "Ici: un workflow n8n prêt à importer pour **déclencher le DAG Airflow** via Webhook.",
)

ROOT = Path.cwd()
COMPOSE = ROOT / "bonus_stack" / "docker-compose.bonus.yml"
WF_FILE = ROOT / "bonus_stack" / "n8n" / "workflow_trigger_airflow.json"

cfg = load_dataset_config()
raw = Path(cfg.raw_csv)

c1, c2 = st.columns(2)
c1.metric("Dataset", raw.name if raw.exists() else "—")
c2.metric("Cible Y", cfg.target_col or "—")

def run_cmd(cmd: list[str], cwd: Path | None = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, shell=False)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return int(p.returncode), out.strip()
    except Exception as e:
        return 1, f"Erreur exécution: {e}"

def docker_compose_cmd(args: list[str]) -> Tuple[int, str]:
    if shutil.which("docker"):
        code, out = run_cmd(["docker", "compose", "-f", str(COMPOSE), *args], cwd=COMPOSE.parent)
        if code == 0 or "unknown command" not in out.lower():
            return code, out
    if shutil.which("docker-compose"):
        return run_cmd(["docker-compose", "-f", str(COMPOSE), *args], cwd=COMPOSE.parent)
    return 1, "Docker/Docker Compose non détecté. Installe Docker Desktop."

docker_ok = shutil.which("docker") is not None or shutil.which("docker-compose") is not None
st.write("Statut Docker :", "✅ OK" if docker_ok else "❌ manquant")

st.divider()
st.subheader("Contrôles n8n (Docker)")

colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("▶️ Start n8n", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["up", "-d", "n8n"])
        st.code(out or f"exit={code}")

with colB:
    if st.button("⏹️ Stop n8n", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["stop", "n8n"])
        st.code(out or f"exit={code}")

with colC:
    if st.button("📌 Status", disabled=not docker_ok, use_container_width=True):
        code, out = docker_compose_cmd(["ps"])
        st.code(out or f"exit={code}")

with colD:
    try:
        st.link_button("🌐 Ouvrir n8n UI", "http://localhost:5678")
    except Exception:
        st.markdown("[🌐 Ouvrir n8n UI](http://localhost:5678)")

st.divider()
st.subheader("Workflow prêt à importer")

if WF_FILE.exists():
    st.success(f"✅ Fichier prêt : {WF_FILE}")
    wf_bytes = WF_FILE.read_bytes()
    st.download_button(
        "⬇️ Télécharger le workflow (JSON)",
        data=wf_bytes,
        file_name="sakiga_n8n_trigger_airflow.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Aperçu du JSON", expanded=False):
        try:
            st.json(json.loads(wf_bytes.decode("utf-8")))
        except Exception:
            st.code(wf_bytes.decode("utf-8", errors="ignore"))
else:
    st.error("Workflow introuvable: bonus_stack/n8n/workflow_trigger_airflow.json")

st.markdown("### Guide d'utilisation : ")
st.markdown(
    "1) Start n8n → ouvrir l'UI\n"
    "2) **Import** le JSON téléchargé (menu n8n → Import)\n"
    "3) Crée un credential **HTTP Basic Auth** (nom: `Airflow Basic Auth`) avec le user/pass Airflow\n"
    "4) Active le workflow\n"
    "5) Appelle le webhook : `POST http://localhost:5678/webhook/sakiga/trigger-train`\n"
    "➡️ Résultat attendu: n8n déclenche le DAG Airflow → ton pipeline tourne → metrics/plots générés."
)

st.caption("💡 Bonus démo: ajoute un node 'Email' / 'Discord' dans n8n pour notifier après le déclenchement.")
