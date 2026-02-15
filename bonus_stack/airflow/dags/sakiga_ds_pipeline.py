# -*- coding: utf-8 -*-
"""SAKIGA DS — Airflow DAG (bonus)

Objectif (cahier des charges):
- Automatiser: préparation → entraînement → évaluation → déploiement conditionnel
- Scheduling + dépendances
- Compatible avec n'importe quel dataset uploadé dans l'app Streamlit

Important:
- Le pipeline lit la config: data/config/dataset_config.json
- Le dataset uploadé est toujours copié ici: data/raw/user_dataset.csv
- Les tâches utilisent un venv local dans /opt/project/.airflow_venv pour éviter de casser Airflow
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_DIR = "/opt/project"
VENV_DIR = "/opt/project/.airflow_venv"
REQ_FILE = "/opt/project/requirements.txt"

DEFAULT_ARGS = {
    "owner": "sakiga",
    "retries": 0,
}

# Thresholds (demo)
DEFAULT_F1_THRESHOLD = float(os.getenv("SAKIGA_F1_THRESHOLD", "0.75"))
DEFAULT_R2_THRESHOLD = float(os.getenv("SAKIGA_R2_THRESHOLD", "0.60"))

def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def choose_deploy_branch(**_context) -> str:
    train_meta = _read_json(os.path.join(PROJECT_DIR, "reports", "train_meta.json"))
    metrics = _read_json(os.path.join(PROJECT_DIR, "reports", "metrics.json"))

    task = (train_meta.get("task") or "").lower()
    if task not in ("classification", "regression"):
        # fallback: guess from metrics keys
        task = "regression" if "r2" in metrics else "classification"

    if task == "classification":
        score = metrics.get("f1_macro") or metrics.get("accuracy") or 0.0
        ok = float(score) >= DEFAULT_F1_THRESHOLD
    else:
        score = metrics.get("r2") or 0.0
        ok = float(score) >= DEFAULT_R2_THRESHOLD

    return "deploy_model" if ok else "skip_deploy"

def deploy_model(**_context) -> None:
    src = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
    dst = os.path.join(PROJECT_DIR, "models", "deployed_model.pkl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)

    status = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "deployed_model": "models/deployed_model.pkl" if os.path.exists(dst) else None,
    }
    out = os.path.join(PROJECT_DIR, "reports", "deploy_status.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

with DAG(
    dag_id="sakiga_ds_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["sakiga", "mlops", "bonus"],
) as dag:

    start = EmptyOperator(task_id="start")

    setup_and_prepare = BashOperator(
        task_id="prepare_data",
        cwd=PROJECT_DIR,
        bash_command=r"""
set -e
echo "📦 [SAKIGA] Prepare data"
python3 -m venv "{venv}" || true

REQ_HASH=$(python3 - << 'PY'
import hashlib, pathlib
p = pathlib.Path("{req}")
h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
print(h)
PY
)

if [ ! -f "{venv}/.req_hash" ] || [ "$(cat {venv}/.req_hash)" != "$REQ_HASH" ]; then
  echo "🔧 Installing requirements into venv (first time / updated requirements)..."
  "{venv}/bin/pip" install --upgrade pip setuptools wheel >/dev/null
  "{venv}/bin/pip" install -r "{req}"
  echo "$REQ_HASH" > "{venv}/.req_hash"
else
  echo "✅ venv already ready"
fi

"{venv}/bin/python" -m src.pipeline.prepare_data
""".format(venv=VENV_DIR, req=REQ_FILE),
    )

    train = BashOperator(
        task_id="train_model",
        cwd=PROJECT_DIR,
        bash_command=r"""
set -e
echo "🤖 [SAKIGA] Train model"
"{venv}/bin/python" -m src.pipeline.train
""".format(venv=VENV_DIR),
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        cwd=PROJECT_DIR,
        bash_command=r"""
set -e
echo "📊 [SAKIGA] Evaluate model"
"{venv}/bin/python" -m src.pipeline.evaluate
""".format(venv=VENV_DIR),
    )

    branch = BranchPythonOperator(
        task_id="deploy_condition",
        python_callable=choose_deploy_branch,
    )

    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    skip = EmptyOperator(task_id="skip_deploy")

    end = EmptyOperator(task_id="end")

    start >> setup_and_prepare >> train >> evaluate >> branch
    branch >> deploy >> end
    branch >> skip >> end
