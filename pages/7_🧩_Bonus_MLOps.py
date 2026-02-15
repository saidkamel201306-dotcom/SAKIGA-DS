# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st

from src.ui.utils.ui import inject_css, card
from src.config.dataset_config import load_dataset_config

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="SAKIGA DS — Bonus MLOps", layout="wide")
inject_css()

ROOT = Path.cwd()
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

st.title("🧩 Bonus — MLOps (DVC + MLflow + Evidently)")
card(
    "Objectif bonus :",
    "Le projet fait **plus que entraîner un modèle** : "
    "versionnement (DVC), traçabilité (MLflow) et monitoring (Evidently).",
    "Tout fonctionne avec le dataset uploadé via l'app (compat : `data/raw/user_dataset.csv`).",
)

# ----------------------------
# Helpers
# ----------------------------
def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int = 900,
) -> Tuple[int, str]:
    """
    Exécute une commande et retourne (exit_code, stdout+stderr).
    - encoding UTF-8 + errors replace => pas de crash Windows
    - timeout pour éviter les freezes infinis
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            shell=False,
            env=env,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return int(p.returncode), out.strip()
    except subprocess.TimeoutExpired:
        return 124, "Timeout : la commande a pris trop de temps (124)."
    except Exception as e:
        return 1, f"Erreur exécution: {e}"


def show_cmd(title: str, code: int, out: str):
    st.markdown(f"**{title}**")
    st.code(f"exit={code}\n\n{out}".strip())


def link_button(label: str, url: str):
    try:
        st.link_button(label, url)
    except Exception:
        st.markdown(f"[{label}]({url})")


def read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def safe_read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)


def summarize_eda() -> dict:
    """
    Résumé humain pour éviter le feedback 'pourri'.
    On lit ce que ton pipeline EDA écrit déjà :
      - reports/eda_summary.json
      - reports/eda_missing.csv
      - reports/eda_outliers.csv
    """
    summary = {
        "rows": None,
        "cols": None,
        "missing_top": [],
        "missing_cols_count": None,
        "outliers_rows": None,
        "files": [],
    }

    eda_summary = read_json(REPORTS_DIR / "eda_summary.json")
    if eda_summary:
        # on essaye plusieurs clés au cas où
        summary["rows"] = eda_summary.get("rows") or eda_summary.get("n_rows") or eda_summary.get("row_count")
        summary["cols"] = eda_summary.get("cols") or eda_summary.get("n_cols") or eda_summary.get("col_count")
        summary["files"].append("reports/eda_summary.json")

    miss_csv = REPORTS_DIR / "eda_missing.csv"
    if miss_csv.exists():
        try:
            dfm = safe_read_csv(miss_csv)
            # colonnes attendues : column + missing_pct (sinon on s’adapte)
            if "missing_pct" in dfm.columns:
                dfm2 = dfm[dfm["missing_pct"] > 0].sort_values("missing_pct", ascending=False)
                summary["missing_cols_count"] = int(len(dfm2))
                if len(dfm2) > 0:
                    top = dfm2.head(5)[["column", "missing_pct"]].values.tolist()
                    summary["missing_top"] = [(str(c), float(p)) for c, p in top]
            elif "missing_count" in dfm.columns:
                dfm2 = dfm[dfm["missing_count"] > 0].sort_values("missing_count", ascending=False)
                summary["missing_cols_count"] = int(len(dfm2))
                if len(dfm2) > 0:
                    top = dfm2.head(5)[["column", "missing_count"]].values.tolist()
                    summary["missing_top"] = [(str(c), float(p)) for c, p in top]
            summary["files"].append("reports/eda_missing.csv")
        except Exception:
            pass

    out_csv = REPORTS_DIR / "eda_outliers.csv"
    if out_csv.exists():
        try:
            dfo = safe_read_csv(out_csv)
            summary["outliers_rows"] = int(len(dfo))
            summary["files"].append("reports/eda_outliers.csv")
        except Exception:
            pass

    return summary


def read_evidently_pointer() -> tuple[Path, Path]:
    """
    Si ton module evidently écrit reports/evidently_pointer.json :
    on s'en sert. Sinon fallback sur noms par défaut.
    """
    pointer = REPORTS_DIR / "evidently_pointer.json"
    if pointer.exists():
        try:
            p = json.loads(pointer.read_text(encoding="utf-8"))
            html_path = Path(p.get("html", "reports/evidently_report.html"))
            json_path = Path(p.get("json", "reports/evidently_report.json"))
            return html_path, json_path
        except Exception:
            pass
    return Path("reports/evidently_report.html"), Path("reports/evidently_report.json")


# ----------------------------
# Dataset context (dataset-agnostic)
# ----------------------------
cfg = load_dataset_config()
raw = Path(cfg.raw_csv)
compat = Path("data/raw/user_dataset.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset (config)", raw.name if raw.exists() else "—")
c2.metric("Cible Y", cfg.target_col or "—")
c3.metric("Task", cfg.task)
c4.metric("Compat dataset", "OK" if compat.exists() else "absent")

with st.expander("Voir dataset_config.json", expanded=False):
    st.code(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), language="json")

st.divider()

tab_dvc, tab_mlflow, tab_evid = st.tabs(["📦 DVC", "🧪 MLflow", "📈 Evidently"])

# ==========================================================
# TAB DVC
# ==========================================================
with tab_dvc:
    st.subheader("📦 DVC — versionner données + rejouer le pipeline")
    st.write("But :  **reproductibilité** (DAG + repro).")

    dvc_available = shutil.which("dvc") is not None
    dvc_repo = (ROOT / ".dvc").exists()

    st.write("DVC installé :", "✅ Oui" if dvc_available else "❌ Non")
    st.write("Repo DVC :", "✅ Déjà initialisé (.dvc/)" if dvc_repo else "❌ Pas initialisé")

    generic_dvc_yaml = """stages:
  prepare_data:
    cmd: python -m src.pipeline.prepare_data
    deps:
      - data/raw/user_dataset.csv
      - data/config/dataset_config.json
      - src/pipeline/prepare_data.py
    outs:
      - data/processed/dataset.csv

  train_model:
    cmd: python -m src.pipeline.train
    deps:
      - data/processed/dataset.csv
      - data/config/dataset_config.json
      - src/pipeline/train.py
    outs:
      - models/best_model.pkl
      - data/processed/splits
      - reports/train_meta.json

  evaluate:
    cmd: python -m src.pipeline.evaluate
    deps:
      - models/best_model.pkl
      - data/processed/splits
      - reports/train_meta.json
      - src/pipeline/evaluate.py
    outs:
      - reports/metrics.json
      - reports/plots
"""

    colA, colB, colC = st.columns([0.36, 0.32, 0.32], gap="large")

    with colA:
        if st.button("🧱 Écrire dvc.yaml (générique)", use_container_width=True):
            (ROOT / "dvc.yaml").write_text(generic_dvc_yaml, encoding="utf-8")
            st.success("✅ dvc.yaml prêt")

        st.download_button(
            "⬇️ Télécharger dvc.yaml",
            data=generic_dvc_yaml.encode("utf-8"),
            file_name="dvc.yaml",
            mime="text/yaml",
            use_container_width=True,
        )

    with colB:
        show_logs = st.checkbox("Afficher logs techniques (DVC)", value=False)

        if st.button("🔎 Diagnostic DVC", use_container_width=True):
            code, out = run_cmd(["dvc", "--version"], cwd=ROOT)
            if show_logs:
                show_cmd("dvc --version", code, out)
            else:
                st.info(out if out else "Diagnostic terminé.")

        if dvc_repo:
            st.info("DVC est déjà initialisé → donc bouton Init désactivé.")

        if st.button("🧪 Init DVC (no-scm)", disabled=(not dvc_available) or dvc_repo, use_container_width=True):
            code, out = run_cmd(["dvc", "init", "--no-scm", "-v"], cwd=ROOT)
            if code == 0:
                st.success("✅ DVC initialisé")
                if show_logs:
                    show_cmd("dvc init --no-scm -v", code, out)
                st.rerun()
            else:
                st.error("❌ Init DVC a échoué")
                show_cmd("dvc init --no-scm -v", code, out)

        if st.button("➕ dvc add dataset (user_dataset.csv)", disabled=not dvc_available, use_container_width=True):
            if not compat.exists():
                st.error("`data/raw/user_dataset.csv` absent. Upload d’abord le dataset via la page Data.")
            else:
                code, out = run_cmd(["dvc", "add", str(compat)], cwd=ROOT)
                if code == 0:
                    st.success("✅ Dataset versionné par DVC")
                    if show_logs:
                        show_cmd("dvc add ...", code, out)
                else:
                    st.error("❌ dvc add a échoué")
                    show_cmd("dvc add ...", code, out)

        with st.expander("⚠️ Avancé — Réinitialiser DVC (optionnel)", expanded=False):
            st.warning("À utiliser uniquement si on veut repartir de zéro (supprime .dvc + dvc.lock + dvc.yaml).")
            confirm = st.checkbox("Je confirme la réinitialisation DVC", value=False)
            if st.button("🧨 Réinitialiser DVC", disabled=not confirm, use_container_width=True):
                for p in [ROOT / ".dvc", ROOT / "dvc.lock", ROOT / "dvc.yaml", ROOT / ".dvcignore"]:
                    try:
                        if p.is_dir():
                            shutil.rmtree(p)
                        elif p.exists():
                            p.unlink()
                    except Exception as e:
                        st.error(f"Erreur suppression {p}: {e}")
                st.success("✅ Réinitialisation terminée")
                st.rerun()

    with colC:
        show_logs2 = st.checkbox("Afficher logs techniques (repro/dag)", value=False)

        if st.button("🔁 dvc repro", disabled=not dvc_available, use_container_width=True):
            code, out = run_cmd(["dvc", "repro"], cwd=ROOT, timeout=1800)
            if code == 0:
                st.success("✅ Pipeline rejoué / à jour")
                if show_logs2:
                    show_cmd("dvc repro", code, out)
            else:
                st.error("❌ dvc repro a échoué")
                show_cmd("dvc repro", code, out)

        if st.button("🧬 dvc dag", disabled=not dvc_available, use_container_width=True):
            code, out = run_cmd(["dvc", "dag"], cwd=ROOT)
            if code == 0:
                st.success("✅ DAG généré")
                st.code(out)
            else:
                st.error("❌ dvc dag a échoué")
                show_cmd("dvc dag", code, out)

    st.caption("Conseil d'utilisation : **dvc dag** (pipeline) puis **dvc repro** (reproductibilité).")

# ==========================================================
# TAB MLFLOW
# ==========================================================
with tab_mlflow:
    st.subheader("🧪 MLflow — traçabilité des expériences")
    st.write("But : garder un historique des runs (params/metrics/artifacts) et ouvrir une UI dédiée.")

    try:
        import mlflow  # noqa: F401
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    st.write("MLflow installé :", "✅ Oui" if mlflow_ok else "❌ Non (optionnel)")

    col1, col2, col3 = st.columns([0.34, 0.36, 0.30], gap="large")

    with col1:
        if st.button("🚀 Run complet (Prepare → Train → Evaluate)", use_container_width=True):
            prog = st.progress(0)
            try:
                from src.pipeline.prepare_data import main as prepare_data_main
                from src.pipeline.train import main as train_main
                from src.pipeline.evaluate import main as evaluate_main

                prepare_data_main(); prog.progress(1/3)
                train_main(); prog.progress(2/3)
                evaluate_main(); prog.progress(1.0)
                st.success("✅ Run complet terminé")
            except Exception as e:
                st.error(f"❌ Erreur pipeline: {e}")

    with col2:
        if st.button("🧾 Log en MLflow (depuis reports/ + models/)", disabled=not mlflow_ok, use_container_width=True):
            try:
                from src.monitoring.mlflow_logger import log_current_run

                exp = f"SAKIGA-DS::{raw.name}" if raw.exists() else "SAKIGA-DS"
                run_id = log_current_run(experiment_name=exp, run_name="streamlit-log")
                st.success(f"✅ MLflow log OK (run_id={run_id})")
            except Exception as e:
                st.error(f"❌ Impossible de log: {e}")

    with col3:
        port = st.number_input("Port MLflow UI", min_value=5000, max_value=5999, value=5000, step=1)
        if st.button("🧪 Démarrer MLflow UI", disabled=not mlflow_ok, use_container_width=True):
            try:
                cmd = [
                    os.sys.executable,
                    "-m",
                    "mlflow",
                    "ui",
                    "--backend-store-uri",
                    "mlruns",
                    "--port",
                    str(int(port)),
                ]
                subprocess.Popen(cmd, cwd=str(ROOT))
                st.success("✅ MLflow UI démarré")
            except Exception as e:
                st.error(f"❌ Impossible de démarrer MLflow UI: {e}")

        link_button("🔗 Ouvrir MLflow UI", f"http://localhost:{int(port)}")

# ==========================================================
# TAB EVIDENTLY
# ==========================================================
with tab_evid:
    st.subheader("📈 Evidently — monitoring (qualité + drift)")
    st.write(
        "But : générer un **rapport HTML** qui vérifie la **qualité des données** et le **drift** "
        "(si train/test existent)."
    )

    colL, colR = st.columns([0.52, 0.48], gap="large")

    with colL:
        st.markdown("### Génération (sans bruit)")
        hide_logs = st.checkbox("Masquer totalement le bruit technique", value=True)

        if st.button("📊 Générer le rapport Evidently", use_container_width=True):
            env = os.environ.copy()
            # ✅ supprime les RuntimeWarning numpy/scipy (le “bruit”)
            env["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
            env["PYTHONIOENCODING"] = "utf-8"

            code, out = run_cmd([os.sys.executable, "-m", "src.monitoring.evidently_generic"], cwd=ROOT, env=env, timeout=1800)

            if code == 0:
                st.success("✅ Rapport Evidently généré.")
                # Résumé humain basé sur EDA (clair)
                s = summarize_eda()

                st.markdown("### Résumé clair")
                if s["rows"] and s["cols"]:
                    st.info(f"Dataset traité : **{s['rows']} lignes** × **{s['cols']} colonnes**")
                else:
                    st.info("Dataset traité : OK (détails lignes/colonnes non disponibles).")

                if s["missing_cols_count"] is None:
                    st.info("Manquants : non mesuré (pas de fichier EDA missing).")
                elif s["missing_cols_count"] == 0:
                    st.success("Manquants : ✅ aucune colonne avec valeurs manquantes.")
                else:
                    msg = f"Manquants : ⚠️ {s['missing_cols_count']} colonnes concernées."
                    if s["missing_top"]:
                        top_txt = ", ".join([f"{c} ({p:.1f}%)" for c, p in s["missing_top"]])
                        msg += f" Top: {top_txt}"
                    st.warning(msg)

                if s["outliers_rows"] is None:
                    st.info("Outliers : non mesuré (pas de fichier EDA outliers).")
                elif s["outliers_rows"] == 0:
                    st.success("Outliers : ✅ aucun outlier détecté (selon la règle EDA).")
                else:
                    st.warning(f"Outliers : ⚠️ {s['outliers_rows']} lignes potentielles (voir reports/eda_outliers.csv).")

                if s["files"]:
                    st.caption("Fichiers utilisés : " + " | ".join(s["files"]))

                # Logs techniques seulement si demandé
                if not hide_logs and out:
                    with st.expander("Logs techniques (optionnel)", expanded=False):
                        st.code(out)

            else:
                st.error("❌ Evidently a échoué.")
                st.code(out or "(aucun log)")

    with colR:
        st.markdown("### Rapport")
        html_path, json_path = read_evidently_pointer()

        if Path(html_path).exists():
            st.success("✅ Rapport HTML prêt")
            st.download_button(
                "⬇️ Télécharger Evidently HTML",
                data=Path(html_path).read_bytes(),
                file_name=Path(html_path).name,
                mime="text/html",
                use_container_width=True,
            )
            st.caption(f"Chemin : {html_path}")
        else:
            st.info("Pas encore de rapport. Clique sur **Générer le rapport Evidently**.")

        st.markdown("### Objectifs evidently :")
        st.write(
            "• **Data Quality** : vérifie si les données sont propres (manquants, colonnes problématiques).\n"
            "• **Data Drift** : vérifie si les données changent entre train et test/nouvelles données — si drift élevé, on pense à ré-entraîner."
        )
