# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.utils.ui import inject_css, card, st_image_safe, make_pdf_report_bytes

st.set_page_config(page_title="SAKIGA DS - Résultats", layout="wide")
inject_css()

st.title("📊 Résultats — dashboard, plots & historique")

# ----------------------------
# Paths (latest)
# ----------------------------
REPORTS_DIR = Path("reports")
RUNS_DIR = Path("runs")

META_LATEST = REPORTS_DIR / "train_meta.json"
METRICS_LATEST = REPORTS_DIR / "metrics.json"
PLOTS_LATEST = REPORTS_DIR / "plots"

CFG_PATH = Path("data") / "config" / "dataset_config.json"
DATASET_STD = Path("data") / "processed" / "dataset.csv"


# ----------------------------
# Helpers
# ----------------------------
def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime if p.exists() else None
    except Exception:
        return None


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _as_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    imgs: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        imgs.extend(folder.glob(ext))
    return sorted(imgs, key=lambda p: p.name)


def _zip_dir_bytes(root_dir: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in root_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(root_dir)))
    bio.seek(0)
    return bio.read()


def _zip_fallback_bytes(base_reports: Path) -> bytes:
    """
    Zip reports/ + models/best_model.pkl (si présent).
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if base_reports.exists():
            for p in base_reports.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
        m = Path("models") / "best_model.pkl"
        if m.exists():
            z.write(m, arcname=str(m))
    bio.seek(0)
    return bio.read()


def _normalize_comparison(comp: list) -> pd.DataFrame:
    if not comp:
        return pd.DataFrame()
    df = pd.DataFrame(comp)

    if "val_primary_metric" in df.columns and "primary_metric" not in df.columns:
        df["primary_metric"] = df["val_primary_metric"]

    if "val_metrics" in df.columns:
        try:
            m = pd.json_normalize(df["val_metrics"])
            for c in m.columns:
                if c not in df.columns:
                    df[c] = m[c]
        except Exception:
            pass

    return df


def _label_pairs(label_mapping: dict) -> list[tuple[str, str]]:
    if not isinstance(label_mapping, dict) or not label_mapping:
        return []
    k0 = next(iter(label_mapping.keys()))
    pairs = []
    if isinstance(k0, int) or (isinstance(k0, str) and str(k0).isdigit()):
        for code, lab in label_mapping.items():
            pairs.append((str(lab), str(code)))
    else:
        for lab, code in label_mapping.items():
            pairs.append((str(lab), str(code)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _render_train_summary(meta: dict, title: str):
    if not meta:
        st.info("Aucun résultat trouvé.")
        return

    task = meta.get("task", "classification")
    scoring = meta.get("scoring", "f1_macro" if task == "classification" else "r2")

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Best model", meta.get("best_model", "-"))
    with c2:
        v = meta.get("best_val_primary_metric")
        st.metric(f"Val ({scoring})", f"{_as_float(v):.4f}" if v is not None else "-")
    with c3:
        t = meta.get("best_test_primary_metric")
        st.metric(f"Test ({scoring})", f"{_as_float(t):.4f}" if t is not None else "-")
    with c4:
        split = meta.get("split", {}) or {}
        st.metric("Rows (train/val/test)", f"{split.get('train_rows','-')}/{split.get('val_rows','-')}/{split.get('test_rows','-')}")

    split_warn = (meta.get("split") or {}).get("split_warning")
    cv_warn = meta.get("cv_warning")
    if split_warn:
        st.warning(f"⚠️ Split: {split_warn}")
    if cv_warn:
        st.warning(f"⚠️ CV: {cv_warn}")

    with st.expander("✅ Justification + meilleurs hyperparamètres", expanded=False):
        st.write(meta.get("justification", ""))
        st.code(json.dumps(meta.get("best_params", {}), indent=2, ensure_ascii=False), language="json")

    if task == "classification" and meta.get("label_mapping"):
        pairs = _label_pairs(meta.get("label_mapping") or {})
        if pairs:
            pretty = ", ".join([f"{lab} → {code}" for lab, code in pairs])
            st.caption(f"Encodage des classes (Y) : {pretty}")


def _render_comparison(meta: dict):
    comp = meta.get("comparison") or []
    df = _normalize_comparison(comp)
    if df.empty:
        st.info("Aucune comparaison disponible.")
        return

    st.markdown("### 📊 Comparaison des modèles (validation)")
    ordered = [c for c in ["model", "primary_metric", "f1_macro", "accuracy", "roc_auc", "r2", "mae", "rmse", "cv_best_score"] if c in df.columns]
    rest = [c for c in df.columns if c not in set(ordered) and c not in {"val_metrics"}]
    if "primary_metric" in df.columns:
        df = df.sort_values("primary_metric", ascending=False, na_position="last")
    st.dataframe(df[ordered + rest], use_container_width=True)


def _render_plots(plot_dir: Path):
    imgs = _list_images(plot_dir)
    st.markdown("### 🖼️ Plots (Evaluate)")
    if not imgs:
        st.caption(f"Aucun plot trouvé dans `{plot_dir}`")
        return
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        with cols[i % 2]:
            st_image_safe(str(p), caption=p.name)


def _render_metrics_and_files(base_reports: Path):
    metrics_path = base_reports / "metrics.json"
    pred_path = base_reports / "predictions_test.csv"
    conf_path = base_reports / "confusion_matrix.csv"

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.expander("📌 metrics.json", expanded=True):
            if metrics_path.exists():
                st.json(_read_json(metrics_path) or {"error": "lecture impossible"})
            else:
                st.caption(f"Aucun metrics.json dans `{base_reports}`")

    with c2:
        with st.expander("🧾 predictions_test.csv (aperçu)", expanded=True):
            if pred_path.exists():
                try:
                    dfp = pd.read_csv(pred_path)
                    st.dataframe(dfp.head(30), use_container_width=True)
                    st.caption(f"Lignes: {len(dfp)} | Colonnes: {len(dfp.columns)}")
                except Exception as e:
                    st.caption(f"Impossible de lire predictions_test.csv: {e}")
            else:
                st.caption(f"Aucun predictions_test.csv dans `{base_reports}`")

    if conf_path.exists():
        with st.expander("🧩 confusion_matrix.csv (aperçu)", expanded=False):
            try:
                dfc = pd.read_csv(conf_path)
                st.dataframe(dfc, use_container_width=True)
            except Exception as e:
                st.caption(f"Impossible de lire confusion_matrix.csv: {e}")


def _list_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "reports" / "train_meta.json"
        meta = _read_json(meta_path)
        if not meta:
            continue
        split = meta.get("split") or {}
        runs.append({
            "run_id": meta.get("run_id", d.name),
            "task": meta.get("task"),
            "target": meta.get("target_col"),
            "best_model": meta.get("best_model"),
            "val_primary": meta.get("best_val_primary_metric"),
            "test_primary": meta.get("best_test_primary_metric"),
            "train_rows": split.get("train_rows"),
            "val_rows": split.get("val_rows"),
            "test_rows": split.get("test_rows"),
            "run_dir": str(d),
        })
    return runs


def _warnings_for_latest():
    meta_ts = _mtime(META_LATEST)
    metrics_ts = _mtime(METRICS_LATEST)
    cfg_ts = _mtime(CFG_PATH)
    ds_ts = _mtime(DATASET_STD)

    warn_msgs = []

    if cfg_ts is not None and meta_ts is not None and cfg_ts > meta_ts:
        warn_msgs.append(
            "⚠️ La config dataset a été modifiée après le dernier train.\n"
            f"- config : {_fmt_ts(cfg_ts)}\n"
            f"- train  : {_fmt_ts(meta_ts)}\n"
            "➡️ Relance Train (puis Evaluate)."
        )

    if ds_ts is not None and meta_ts is not None and ds_ts > meta_ts:
        warn_msgs.append(
            "⚠️ Le dataset standardisé (data/processed/dataset.csv) est plus récent que le dernier train.\n"
            f"- dataset : {_fmt_ts(ds_ts)}\n"
            f"- train   : {_fmt_ts(meta_ts)}\n"
            "➡️ Relance Train (puis Evaluate)."
        )

    if meta_ts is not None and metrics_ts is not None and meta_ts > metrics_ts:
        warn_msgs.append(
            "⚠️ Le train est plus récent que l’evaluate.\n"
            f"- train    : {_fmt_ts(meta_ts)}\n"
            f"- evaluate : {_fmt_ts(metrics_ts)}\n"
            "➡️ Relance Evaluate."
        )

    if not METRICS_LATEST.exists():
        warn_msgs.append("ℹ️ Aucune métrique/plots d’évaluation détectés. Lance Evaluate après le train.")

    if warn_msgs:
        st.warning("\n\n".join(warn_msgs))

    with st.expander("🔎 Contexte dataset (anti-confusion)", expanded=False):
        if CFG_PATH.exists():
            cfg = _read_json(CFG_PATH) or {}
            st.write(f"Fichier brut : {cfg.get('raw_csv', '-')}")
            st.write(f"Séparateur : {cfg.get('sep', '-')}")
            st.write(f"Type : {cfg.get('task', '-')}")
            st.write(f"Y (cible) : {cfg.get('target_col', '-')}")
            st.write(f"Nb X : {len(cfg.get('feature_cols') or [])}")
            st.write(f"Dernière modif config : {_fmt_ts(_mtime(CFG_PATH))}")
        else:
            st.info("Config dataset introuvable.")

        st.write(f"Dernier train_meta : {_fmt_ts(_mtime(META_LATEST))}")
        st.write(f"Dernier metrics.json : {_fmt_ts(_mtime(METRICS_LATEST))}")
        st.write(f"Dernier dataset.csv : {_fmt_ts(_mtime(DATASET_STD))}")


def _downloads_block(run_dir: Path | None, base_reports: Path, key_prefix: str):
    st.markdown("### ⬇️ Exports (ZIP + PDF)")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**ZIP (artefacts complets)**")
        if run_dir and run_dir.exists():
            if st.button("Préparer ZIP", key=f"{key_prefix}_zip_prepare"):
                with st.spinner("Création du ZIP…"):
                    st.session_state[f"{key_prefix}_zip_bytes"] = _zip_dir_bytes(run_dir)
                    st.session_state[f"{key_prefix}_zip_name"] = f"{run_dir.name}.zip"

            if f"{key_prefix}_zip_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Télécharger ZIP",
                    data=st.session_state[f"{key_prefix}_zip_bytes"],
                    file_name=st.session_state.get(f"{key_prefix}_zip_name", "run.zip"),
                    mime="application/zip",
                    key=f"{key_prefix}_zip_dl",
                )
                st.caption(f"Source: `{run_dir}`")
        else:
            st.caption("run_dir introuvable → ZIP fallback (reports + model).")
            if st.button("Préparer ZIP (fallback)", key=f"{key_prefix}_zip_fallback_prepare"):
                with st.spinner("Création du ZIP fallback…"):
                    st.session_state[f"{key_prefix}_zip_fallback_bytes"] = _zip_fallback_bytes(base_reports)

            if f"{key_prefix}_zip_fallback_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Télécharger ZIP (fallback)",
                    data=st.session_state[f"{key_prefix}_zip_fallback_bytes"],
                    file_name="sakiga_latest_artifacts.zip",
                    mime="application/zip",
                    key=f"{key_prefix}_zip_fallback_dl",
                )

    with c2:
        st.markdown("**PDF (rapport pro)**")
        if st.button("Préparer PDF", key=f"{key_prefix}_pdf_prepare"):
            try:
                with st.spinner("Génération du PDF…"):
                    pdf_data, pdf_name = make_pdf_report_bytes(run_dir=str(run_dir) if run_dir else None)
                    st.session_state[f"{key_prefix}_pdf_bytes"] = pdf_data
                    st.session_state[f"{key_prefix}_pdf_name"] = pdf_name
            except Exception as e:
                st.error(f"PDF impossible: {e}")

        if f"{key_prefix}_pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️ Télécharger PDF",
                data=st.session_state[f"{key_prefix}_pdf_bytes"],
                file_name=st.session_state.get(f"{key_prefix}_pdf_name", "report.pdf"),
                mime="application/pdf",
                key=f"{key_prefix}_pdf_dl",
            )


# ----------------------------
# Header KPI
# ----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("train_meta.json", "OK" if META_LATEST.exists() else "Absent")
with col2:
    st.metric("metrics.json", "OK" if METRICS_LATEST.exists() else "Absent")
with col3:
    st.metric("plots", len(_list_images(PLOTS_LATEST)))
with col4:
    st.metric("runs/", len([d for d in RUNS_DIR.iterdir() if d.is_dir()]) if RUNS_DIR.exists() else 0)

card(
    "Ce que on doit voir ici",
    "- **Dernier run** : KPIs, comparaison, plots, fichiers (metrics/predictions/confusion)\n"
    "- **Exports** : ZIP + PDF\n"
    "- **Historique** : comparer plusieurs runs",
    "Si on change le dataset ou X/Y : on relance **Prepare → Train → Evaluate**.",
)

view = st.radio(
    "Vue",
    ["✅ Dernier run (latest)", "🕒 Historique des runs"],
    horizontal=True,
    key="results_view",
)

# =========================================================
# Vue 1 — Latest
# =========================================================
if view.startswith("✅"):

    if not META_LATEST.exists():
        st.warning("Aucun résultat d'entraînement trouvé. Lance un run depuis 🧪 Train & Evaluate.")
        st.stop()

    meta_latest = _read_json(META_LATEST) or {}
    _warnings_for_latest()

    # Resolve run_dir if available
    run_dir = None
    art = meta_latest.get("artifacts") or {}
    if art.get("run_dir"):
        p = Path(art["run_dir"])
        if p.exists():
            run_dir = p

    _render_train_summary(meta_latest, title="Dernier entraînement (latest)")
    st.divider()

    _render_comparison(meta_latest)
    st.divider()

    _render_plots(PLOTS_LATEST)
    st.divider()

    _render_metrics_and_files(REPORTS_DIR)
    st.divider()

    _downloads_block(run_dir=run_dir, base_reports=REPORTS_DIR, key_prefix="latest")

# =========================================================
# Vue 2 — History
# =========================================================
else:
    runs = _list_runs()
    if not runs:
        st.info("Aucun run trouvé dans `runs/`. Ils apparaissent après un Train.")
        st.stop()

    df_runs = pd.DataFrame(runs)
    st.dataframe(
        df_runs[["run_id", "task", "target", "best_model", "val_primary", "test_primary", "train_rows", "val_rows", "test_rows"]],
        use_container_width=True,
    )

    run_ids = df_runs["run_id"].tolist()
    chosen = st.selectbox("Choisir un run", options=run_ids, index=0, key="run_select")

    chosen_row = df_runs[df_runs["run_id"] == chosen].iloc[0]
    chosen_dir = Path(chosen_row["run_dir"])
    chosen_reports = chosen_dir / "reports"
    chosen_meta = _read_json(chosen_reports / "train_meta.json") or {}

    st.divider()
    _render_train_summary(chosen_meta, title=f"Run sélectionné : {chosen}")
    st.caption(f"Dossier run : `{chosen_dir}`")

    st.divider()
    _render_comparison(chosen_meta)

    st.divider()
    _render_plots(chosen_reports / "plots")

    st.divider()
    _render_metrics_and_files(chosen_reports)

    st.divider()
    _downloads_block(run_dir=chosen_dir, base_reports=chosen_reports, key_prefix="sel")
