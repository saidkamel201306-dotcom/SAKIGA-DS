# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.ui.utils.ui import inject_css, card
from src.config.dataset_config import load_dataset_config, save_dataset_config
from src.ui.utils.feature_suggest import suggest_best_x_fast

st.set_page_config(page_title="SAKIGA DS - Data", layout="wide")
inject_css()

st.title("📁 Données — charger un CSV et configurer X / Y")

card(
    "Objectif",
    "Charger un dataset (CSV), choisir la cible Y, sélectionner les variables X, puis lancer la pipeline sur ces choix.",
    "Astuce : vérifie le séparateur (`,` ou `;`)."
)

cfg = load_dataset_config()


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = base.replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "dataset.csv"


def _clear_suggestion_state():
    for k in ["__suggested_x", "__suggest_report", "__suggest_note", "__suggest_task_used"]:
        st.session_state.pop(k, None)


def _reset_x_state():
    st.session_state.pop("feature_cols", None)
    _clear_suggestion_state()


st.subheader("1) Charger un CSV")
colA, colB = st.columns([0.62, 0.38], gap="large")

with colA:
    uploaded = st.file_uploader("Dataset CSV", type=["csv"], key="uploader_csv")
    sep = st.selectbox(
        "Séparateur",
        options=[",", ";", "\t", "|"],
        index=1 if cfg.sep == ";" else 0,
        key="sep_select",
    )

    if uploaded is not None:
        raw_dir = Path("data") / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        fname = _safe_filename(getattr(uploaded, "name", "dataset.csv"))
        raw_path = raw_dir / fname
        raw_path.write_bytes(uploaded.getvalue())

        # compat copy (si d’autres scripts l’utilisent)
        compat_path = raw_dir / "user_dataset.csv"
        compat_path.write_bytes(uploaded.getvalue())

        cfg.raw_csv = str(raw_path)
        cfg.sep = "\t" if sep == "\t" else sep

        # Important: quand on change de dataset, on ne garde pas un Y/X vieux
        cfg.target_col = ""
        cfg.feature_cols = []
        cfg.drop_cols = cfg.drop_cols or []
        cfg.id_col = None
        cfg.text_col = None

        save_dataset_config(cfg)
        _reset_x_state()

        st.success(f"✅ Sauvegardé: {raw_path.name}")
        st.caption("Copie compatibilité : `data/raw/user_dataset.csv` (écrasée à chaque upload).")
    else:
        st.caption(f"Dataset actuel: `{cfg.raw_csv}` (sep=`{cfg.sep}`)")

with colB:
    st.markdown("**Type de problème**")
    task = st.radio(
        "Mode",
        ["classification", "regression"],
        index=0 if cfg.task == "classification" else 1,
        key="task_radio",
    )
    cfg.task = task
    save_dataset_config(cfg)
    st.caption("Classification → F1/AUC | Régression → R²/MAE")

st.divider()

st.subheader("2) Prévisualiser & choisir Y / X")
raw_path = Path(cfg.raw_csv)
if not raw_path.exists():
    st.warning("⚠️ Aucun CSV trouvé. Charge un fichier ci-dessus ou mets un CSV dans `data/raw/` puis mets `raw_csv` dans la config.")
    st.stop()

try:
    df = pd.read_csv(raw_path, sep=cfg.sep)
except Exception as e:
    st.error(f"Impossible de lire le CSV: {e}")
    st.stop()

# --- detect dataset change (path + columns signature) to avoid weird widget resets ---
dataset_signature = f"{raw_path.resolve()}|{cfg.sep}|{len(df)}|{','.join(map(str, df.columns))}"
if st.session_state.get("__dataset_signature") != dataset_signature:
    st.session_state["__dataset_signature"] = dataset_signature
    _reset_x_state()
    st.session_state.pop("drop_cols", None)
    st.session_state.pop("__last_target_col", None)

st.write("Aperçu:")
st.dataframe(df.head(30), use_container_width=True)

cols = list(df.columns)
c1, c2 = st.columns([0.45, 0.55], gap="large")

# --- Normalize config fields for current dataset (NE PAS forcer Y par défaut) ---
if cfg.target_col and cfg.target_col not in cols:
    cfg.target_col = ""
if getattr(cfg, "id_col", None) not in cols:
    cfg.id_col = None
if getattr(cfg, "text_col", None) not in cols:
    cfg.text_col = None
cfg.drop_cols = [c for c in (cfg.drop_cols or []) if c in cols]
save_dataset_config(cfg)

with c1:
    # ✅ Y obligatoire avec placeholder
    target_options = ["(choisir...)"] + cols
    target_idx = (1 + cols.index(cfg.target_col)) if (cfg.target_col in cols) else 0

    target_choice = st.selectbox(
        "Choisir la cible Y (obligatoire)",
        options=target_options,
        index=target_idx,
        key="target_col_select",
    )

    # Si Y change, purge X + suggestion
    prev_target = st.session_state.get("__last_target_col")
    if prev_target is None:
        st.session_state["__last_target_col"] = target_choice
    elif prev_target != target_choice:
        st.session_state["__last_target_col"] = target_choice
        _reset_x_state()

    cfg.target_col = "" if target_choice == "(choisir...)" else target_choice

    id_col = st.selectbox(
        "Colonne ID (optionnel)",
        options=["(none)"] + cols,
        index=(1 + cols.index(cfg.id_col)) if cfg.id_col in cols else 0,
        key="id_col_select",
    )
    cfg.id_col = None if id_col == "(none)" else id_col

    text_col = st.selectbox(
        "Colonne texte principale (optionnel)",
        options=["(none)"] + cols,
        index=(1 + cols.index(cfg.text_col)) if cfg.text_col in cols else 0,
        key="text_col_select",
    )
    cfg.text_col = None if text_col == "(none)" else text_col

    save_dataset_config(cfg)

# ✅ Si Y pas choisi, on bloque proprement (évite crash plus bas)
if not cfg.target_col:
    st.info("👆 Choisis d’abord la **cible Y** (obligatoire). Ensuite tu pourras configurer X et lancer la pipeline.")
    st.stop()

with c2:
    default_feats = [c for c in cols if c not in {cfg.target_col} and c != cfg.id_col]

    # Drop cols (stable key)
    if "drop_cols" not in st.session_state:
        st.session_state["drop_cols"] = [c for c in (cfg.drop_cols or []) if c in cols]

    # nettoyage drop_cols (sécurité)
    st.session_state["drop_cols"] = [
        c for c in (st.session_state.get("drop_cols") or [])
        if c in cols and c != cfg.target_col
    ]

    drop_cols = st.multiselect(
        "Colonnes à ignorer (optionnel)",
        options=[c for c in cols if c not in {cfg.target_col}],
        default=st.session_state["drop_cols"],
        key="drop_cols",
    )
    cfg.drop_cols = [c for c in drop_cols if c in cols]
    ignored_set = set(cfg.drop_cols or [])

    # ✅ Mode X : Auto vs Manuel (clé de l’app générique)
    x_mode = st.radio(
        "Mode des variables X",
        ["Auto (recommandé)", "Manuel"],
        horizontal=True,
        key="x_mode_radio",
    )

    # reset si changement de mode
    prev_mode = st.session_state.get("__x_mode")
    if prev_mode is None:
        st.session_state["__x_mode"] = x_mode
    elif prev_mode != x_mode:
        st.session_state["__x_mode"] = x_mode
        _reset_x_state()

    candidate_x = [c for c in default_feats if c not in ignored_set]

    if x_mode.startswith("Auto"):
        cfg.feature_cols = []  # => auto dans le reste de l’app
        save_dataset_config(cfg)
        st.success(f"✅ Mode Auto : {len(candidate_x)} colonnes candidates (hors Y / ID / ignorées).")
        st.caption("La pipeline décidera automatiquement des transformations (num/cat/text) selon tes scripts.")
    else:
        # Manuel : on permet la sélection + suggestion
        if "feature_cols" not in st.session_state:
            if getattr(cfg, "feature_cols", None):
                st.session_state["feature_cols"] = [c for c in cfg.feature_cols if c in default_feats]
            else:
                st.session_state["feature_cols"] = candidate_x[: min(8, len(candidate_x))]

        # ✅ Stabilisation du multiselect X (manuel)
        # - ne pas utiliser `default` quand on utilise `key`
        # - ne pas réécrire feature_cols à chaque rerun (sinon suppression parfois "bloquée")
        cand_sig = "|".join(candidate_x)
        if st.session_state.get("__cand_sig_x") != cand_sig:
            st.session_state["__cand_sig_x"] = cand_sig
            st.session_state["feature_cols"] = [
                c for c in (st.session_state.get("feature_cols") or [])
                if c in candidate_x
            ]

        feature_cols = st.multiselect(
            "Choisir les variables X (manuel)",
            options=candidate_x,
            key="feature_cols",
        )
        cfg.feature_cols = [c for c in feature_cols if c in candidate_x]
        save_dataset_config(cfg)

        with st.expander("ℹ️ Aide : comment choisir X ?", expanded=False):
            st.markdown(
                """
- **Rapide** : utilise la suggestion Mutual Information (score global, model-agnostic).
- **Plus fin** : SFS / forward / backward (mais plus coûteux).
"""
            )

        k_max = max(3, min(20, len(candidate_x))) if len(candidate_x) > 0 else 3
        top_k = 3 if k_max <= 3 else st.slider(
            "Top K (suggestion rapide)",
            min_value=3,
            max_value=k_max,
            value=min(10, k_max),
            step=1,
            key="topk_slider",
        )

        if st.button("✨ Suggérer des X (rapide)", key="btn_suggest_x"):
            try:
                if not candidate_x:
                    raise ValueError("Impossible de suggérer des X : aucune colonne candidate.")
                top_k_eff = min(int(top_k), len(candidate_x))

                res = suggest_best_x_fast(
                    df=df,
                    y_col=cfg.target_col,
                    candidate_x=candidate_x,
                    task=cfg.task,
                    top_k=top_k_eff,
                )

                st.session_state["__suggested_x"] = [c for c in res.suggested if c in candidate_x]
                st.session_state["__suggest_report"] = res.report
                st.session_state["__suggest_note"] = getattr(res, "note", "")
                st.session_state["__suggest_task_used"] = getattr(res, "task_used", "")

                st.success(f"✅ Suggestions (Top {top_k_eff}) : {', '.join(st.session_state['__suggested_x'])}")
                if getattr(res, "note", ""):
                    st.info(res.note)
            except Exception as e:
                st.error(f"Impossible de suggérer des X : {e}")

        if "__suggested_x" in st.session_state:
            st.caption("Classement global (Mutual Information) :")
            st.dataframe(st.session_state["__suggest_report"].head(30), use_container_width=True)

            c_apply, c_clear = st.columns([0.62, 0.38], gap="large")
            with c_apply:
                if st.button("✅ Appliquer ces X", key="btn_apply_suggest"):
                    st.session_state["feature_cols"] = st.session_state["__suggested_x"]
                    cfg.feature_cols = st.session_state["feature_cols"]
                    save_dataset_config(cfg)
                    st.success("OK — X mis à jour.")
                    st.rerun()
            with c_clear:
                if st.button("🧹 Effacer suggestion", key="btn_clear_suggest"):
                    _clear_suggestion_state()
                    st.rerun()

# Warning si Y continue en classification
y_series = df[cfg.target_col]
if cfg.task == "classification" and pd.api.types.is_numeric_dtype(y_series) and y_series.nunique(dropna=True) > 20:
    st.warning(
        "⚠️ Ta cible Y ressemble à une variable continue. "
        "Pense à passer en **régression** dans le panneau de droite."
    )

save_dataset_config(cfg)

# Résumé pro (AUTO vs MANUEL)
x_is_auto = (cfg.feature_cols is None) or (len(cfg.feature_cols) == 0)
x_label = "AUTO" if x_is_auto else f"{len(cfg.feature_cols)} colonnes"

st.info(
    f"✅ Config enregistrée dans `data/config/dataset_config.json`\n\n"
    f"- Dataset = `{Path(cfg.raw_csv).name}`\n"
    f"- Y = `{cfg.target_col}`\n"
    f"- X = {x_label}\n"
    f"- task = `{cfg.task}`"
)

st.divider()

st.subheader("3) Vérifications rapides")
c3, c4, c5 = st.columns(3)
with c3:
    st.metric("Lignes", f"{len(df):,}".replace(",", " "))
with c4:
    st.metric("Colonnes", len(df.columns))
with c5:
    st.metric("Valeurs manquantes (Y)", int(df[cfg.target_col].isna().sum()) if cfg.target_col else 0)

# ✅ Bouton désactivé si Y pas choisi (sécurité)
btn_disabled = not bool(cfg.target_col)


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _run_prepare_data_ui() -> bool:
    """
    UX meilleur au clic:
    - progress + spinner
    - résumé des fichiers générés (persistant)
    """
    box = st.container()
    msg = box.empty()
    bar = box.progress(0)

    try:
        msg.markdown("⏳ **Prepare Data** — démarrage…")
        bar.progress(10)

        from src.pipeline.prepare_data import main as prepare_main

        with st.spinner("Préparation + standardisation en cours…"):
            bar.progress(35)
            prepare_main()
            bar.progress(100)

        # Résumé des outputs
        outputs = [
            Path("data/processed/dataset.csv"),
            Path("reports/eda_summary.json"),
            Path("reports/eda_missing.csv"),
            Path("reports/eda_outliers.csv"),
        ]

        lines = []
        for p in outputs:
            if p.exists():
                try:
                    ts = _fmt_ts(p.stat().st_mtime)
                    sz = p.stat().st_size / 1024
                    lines.append(f"✅ `{p}` — {sz:.1f} KB — {ts}")
                except Exception:
                    lines.append(f"✅ `{p}`")
            else:
                lines.append(f"— `{p}`")

        st.session_state["__prepare_summary"] = "\n".join(lines)
        msg.markdown("✅ **Prepare Data** — terminé.")
        return True

    except Exception as e:
        msg.markdown("❌ **Prepare Data** — erreur.")
        box.error(str(e))
        return False


if st.button(
    "📦 Sauvegarder une copie standardisée (data/processed/dataset.csv)",
    key="btn_prepare",
    disabled=btn_disabled,
):
    ok = _run_prepare_data_ui()
    if ok:
        st.success("OK. Prochaine étape : **🧪 Train & Evaluate** (Train puis Evaluate).")
        st.info("👉 Les visualisations + exports + historique sont dans **📊 Résultats**.")
        try:
            st.toast("✅ Prepare Data terminé")
        except Exception:
            pass

# Résumé persistant (après rerun)
if st.session_state.get("__prepare_summary"):
    card(
        "Fichiers générés (Prepare)",
        st.session_state["__prepare_summary"],
        "Astuce : si tu changes Y/X, relance Prepare → Train → Evaluate.",
    )
