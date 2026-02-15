# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.ui.utils.ui import inject_css, card

st.set_page_config(page_title="SAKIGA DS - Export & DB", layout="wide")
inject_css()

st.title("🗃️ Export & DB — CSV / artefacts / MySQL (optionnel)")
st.caption("Page bonus propre : exporter des artefacts, générer un CSV manuel compatible avec le dataset, et pousser vers MySQL (si besoin).")

REPORTS = Path("reports")
MODELS = Path("models")
RUNS = Path("runs")
DATASET_PROCESSED = Path("data/processed/dataset.csv")
SPLITS = Path("data/processed/splits")


# ----------------------------
# Helpers
# ----------------------------
def _read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _find_first(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _zip_dir_bytes(root_dir: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in root_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(root_dir)))
    bio.seek(0)
    return bio.read()


def _latest_run_dir() -> Optional[Path]:
    if not RUNS.exists():
        return None
    dirs = [d for d in RUNS.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1]


def _load_train_schema() -> Dict[str, Any]:
    meta = _read_json(REPORTS / "train_meta.json") or {}
    x_cols = meta.get("x_cols") or []
    ft = meta.get("feature_types") or {}
    num = set(ft.get("num") or [])
    cat = set(ft.get("cat") or [])
    text = set(ft.get("text") or [])
    return {"meta": meta, "x_cols": x_cols, "num": num, "cat": cat, "text": text}


def _sample_categories(df: Optional[pd.DataFrame], col: str, k: int = 25) -> List[str]:
    if df is None or col not in df.columns:
        return [""]
    s = df[col].astype(str).replace("nan", "").replace("None", "")
    vc = s.value_counts().head(k)
    opts = [o for o in vc.index.tolist() if o != ""]
    return opts if opts else [""]


def _ensure_session_rows():
    if "manual_rows_generic" not in st.session_state:
        st.session_state.manual_rows_generic = []


def _rows_to_df(rows: List[dict], cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


# ----------------------------
# Header cards
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("train_meta.json", "✅ OK" if (REPORTS / "train_meta.json").exists() else "❌ Absent", "Schéma X + types")
with c2:
    card("best_model.pkl", "✅ OK" if (MODELS / "best_model.pkl").exists() else "❌ Absent", "Modèle entraîné")
with c3:
    card("metrics.json", "✅ OK" if (REPORTS / "metrics.json").exists() else "❌ Absent", "Résultats Evaluate")
with c4:
    card("runs/", "✅ OK" if RUNS.exists() else "❌ Absent", "Historique bonus")

st.divider()

tabs = st.tabs(["📦 Export artefacts", "🧍 Générer CSV manuel", "🛢️ MySQL (optionnel)"])


# ==========================================================
# TAB 1 — Export artefacts
# ==========================================================
with tabs[0]:
    st.subheader("📦 Export des artefacts")

    colA, colB = st.columns([0.55, 0.45], gap="large")

    with colA:
        st.write("**Fichiers “latest” (racine)**")
        files = [
            (MODELS / "best_model.pkl", "best_model.pkl"),
            (REPORTS / "train_meta.json", "train_meta.json"),
            (REPORTS / "metrics.json", "metrics.json"),
            (REPORTS / "predictions_test.csv", "predictions_test.csv"),
            (REPORTS / "confusion_matrix.csv", "confusion_matrix.csv"),
        ]
        for p, label in files:
            if p.exists():
                st.download_button(
                    f"⬇️ Télécharger {label}",
                    data=p.read_bytes(),
                    file_name=label,
                    mime="application/octet-stream" if p.suffix == ".pkl" else "text/plain",
                )
            else:
                st.caption(f"— {label} (absent)")

    with colB:
        st.write("**ZIP du dernier run (recommandé)**")
        last = _latest_run_dir()
        if not last:
            st.info("Aucun run trouvé. Fais Train → Evaluate d'abord.")
        else:
            st.code(last.as_posix(), language="text")
            if st.button("Préparer ZIP (dernier run)"):
                zbytes = _zip_dir_bytes(last)
                st.session_state["_export_last_run_zip"] = zbytes
                st.session_state["_export_last_run_name"] = f"{last.name}.zip"
                st.success("✅ ZIP prêt.")

            if "_export_last_run_zip" in st.session_state:
                st.download_button(
                    "⬇️ Télécharger ZIP (dernier run)",
                    data=st.session_state["_export_last_run_zip"],
                    file_name=st.session_state.get("_export_last_run_name", "run.zip"),
                    mime="application/zip",
                )

    st.caption("Astuce : le ZIP run est le plus “pro” (modèle + rapports + plots + meta).")


# ==========================================================
# TAB 2 — Manual CSV generator (generic)
# ==========================================================
with tabs[1]:
    st.subheader("🧍 Générer un CSV manuel (générique, basé sur tes X)")

    schema = _load_train_schema()
    x_cols: List[str] = schema["x_cols"]
    num = schema["num"]
    cat = schema["cat"]
    text = schema["text"]

    _ensure_session_rows()

    ref_df = None
    if DATASET_PROCESSED.exists():
        try:
            ref_df = pd.read_csv(DATASET_PROCESSED)
        except Exception:
            ref_df = None

    if not x_cols:
        st.warning("Je ne trouve pas `x_cols` (reports/train_meta.json). Fais d'abord Train.")
        st.stop()

    card(
        "Principe",
        "- Le formulaire est généré automatiquement depuis `x_cols` (train_meta).\n"
        "- Les types (num/cat/text) sont repris depuis `feature_types`.\n"
        "- Tu peux ajouter des lignes, exporter CSV, et éventuellement pousser en DB.",
        "C’est réutilisable sur n’importe quel dataset (du moment que tu as entraîné un modèle)."
    )

    with st.expander("Ajouter une ligne (1 observation)", expanded=True):
        cols = st.columns(2, gap="large")
        values: Dict[str, Any] = {}

        for i, col in enumerate(x_cols):
            with cols[i % 2]:
                # TEXT
                if col in text:
                    default = ""
                    if ref_df is not None and col in ref_df.columns and ref_df[col].notna().any():
                        default = str(ref_df[col].dropna().astype(str).head(1).values[0])
                    values[col] = st.text_area(col, value=default, height=90)
                    continue

                # NUM
                if col in num:
                    med = 0.0
                    if ref_df is not None and col in ref_df.columns:
                        s = pd.to_numeric(ref_df[col], errors="coerce")
                        if s.notna().any():
                            med = float(s.median())
                    values[col] = st.number_input(col, value=float(med))
                    continue

                # CAT / fallback
                opts = _sample_categories(ref_df, col, k=25)
                # si aucune option intéressante => input texte
                if opts == [""]:
                    values[col] = st.text_input(col, value="")
                else:
                    values[col] = st.selectbox(col, options=opts, index=0)

        b1, b2, b3 = st.columns([0.25, 0.25, 0.5])
        with b1:
            if st.button("➕ Ajouter la ligne"):
                st.session_state.manual_rows_generic.append(values)
                st.success("✅ Ligne ajoutée.")
        with b2:
            if st.button("↩️ Supprimer la dernière"):
                if st.session_state.manual_rows_generic:
                    st.session_state.manual_rows_generic.pop()
                    st.info("Dernière ligne supprimée.")
        with b3:
            if st.button("🧹 Vider toutes les lignes"):
                st.session_state.manual_rows_generic = []
                st.warning("Toutes les lignes ont été supprimées.")

    st.markdown("---")
    st.subheader("Table (lignes générées)")

    dfm = _rows_to_df(st.session_state.manual_rows_generic, x_cols)
    st.dataframe(dfm, use_container_width=True, height=340)

    colX, colY = st.columns([0.6, 0.4], gap="large")
    with colX:
        st.write("**Importer un CSV (append)**")
        up = st.file_uploader("Importer un CSV (doit contenir les X)", type=["csv"], key="manual_import_csv")
        if up is not None:
            try:
                dfi = pd.read_csv(up)
                missing = [c for c in x_cols if c not in dfi.columns]
                if missing:
                    st.error("CSV importé invalide : colonnes X manquantes.")
                    st.write(missing)
                else:
                    rows = dfi[x_cols].to_dict(orient="records")
                    st.session_state.manual_rows_generic.extend(rows)
                    st.success(f"✅ {len(rows)} lignes ajoutées.")
            except Exception as e:
                st.error(f"Erreur import CSV : {e}")

    with colY:
        st.write("**Exporter**")
        export_path = Path("data/processed/manual_rows.csv")
        if st.button("💾 Sauver dans data/processed/manual_rows.csv"):
            export_path.parent.mkdir(parents=True, exist_ok=True)
            dfm.to_csv(export_path, index=False)
            st.success(f"✅ Sauvé : {export_path.as_posix()}")

        csv_bytes = dfm.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger manual_rows.csv", data=csv_bytes, file_name="manual_rows.csv", mime="text/csv")


# ==========================================================
# TAB 3 — MySQL push (optional)
# ==========================================================
with tabs[2]:
    st.subheader("🛢️ MySQL (optionnel) — Push table")

    st.caption("Optionnel : si on veut une connexion DB.")
    _ensure_session_rows()
    schema = _load_train_schema()
    x_cols = schema["x_cols"]

    dfm = _rows_to_df(st.session_state.manual_rows_generic, x_cols)
    if dfm.empty:
        st.info("Ajoute au moins une ligne dans l’onglet 'Générer CSV manuel' pour activer le push.")
        st.stop()

    db_url = st.text_input("DB_URL (SQLAlchemy)", value=os.getenv("DB_URL", ""))
    table = st.text_input("Table cible", value=os.getenv("DB_TABLE", "dataset_rows"))
    if_exists = st.selectbox("Mode", ["append", "replace"], index=0)

    st.caption("Exemple MySQL: mysql+pymysql://user:pass@localhost:3306/ma_db")

    if st.button("🚀 Push MySQL"):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except Exception:
            st.error("SQLAlchemy non installé. Fais `pip install sqlalchemy pymysql` si besoin de l'utiliser.")
            st.stop()

        if not db_url.strip():
            st.error("DB_URL vide.")
            st.stop()

        try:
            engine = create_engine(db_url)
            dfm.to_sql(table, con=engine, if_exists=if_exists, index=False)
            st.success(f"✅ {len(dfm)} lignes envoyées vers `{table}` ({if_exists}).")
        except Exception as e:
            st.error(f"Erreur MySQL : {e}")
