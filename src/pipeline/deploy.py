# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


def _project_root() -> Path:
    """
    On prend le dossier racine du projet à partir de l'emplacement du fichier.
    deploy.py = <root>/src/pipeline/deploy.py
    parents[2] => <root>
    """
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _project_root()

    models_dir = root / "models"
    reports_dir = root / "reports"

    best_model = models_dir / "best_model.pkl"
    deployed_model = models_dir / "deployed_model.pkl"
    status_file = reports_dir / "deploy_status.json"

    # Sécurité: créer les dossiers s'ils n'existent pas
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not best_model.exists():
        raise FileNotFoundError(
            f"Modèle source introuvable: {best_model}\n"
            "➡️ Lance d'abord Train & Evaluate (ou le DAG Airflow jusqu'à l'évaluation)."
        )

    # "Déploiement" simple et clair : on copie le best_model vers deployed_model
    shutil.copy2(best_model, deployed_model)

    # Petit statut utile pour ta démo/prof
    status = {
        "status": "deployed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": str(best_model),
        "destination": str(deployed_model),
        "destination_exists": deployed_model.exists(),
        "destination_size_bytes": deployed_model.stat().st_size if deployed_model.exists() else None,
    }

    status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ DEPLOY OK")
    print(f" - source      : {best_model}")
    print(f" - destination : {deployed_model}")
    print(f" - status json : {status_file}")


if __name__ == "__main__":
    main()
