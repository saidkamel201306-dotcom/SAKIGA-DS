# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import re
from html import escape as _escape
from pathlib import Path
from typing import Optional

import streamlit as st


# ----------------------------
# UI: CSS + Cards
# ----------------------------
def inject_css():
    """
    CSS global "pro" (safe).
    - Améliore cards / metrics / buttons / tabs / expanders / sidebar / tables
    - Ne change pas tes APIs Python -> pas de casse
    """
    st.markdown(
        """
<style>
/* =========================================================
   SAKIGA DS — Pro UI (global, safe)
   ========================================================= */

/* Layout spacing */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px; }

/* Variables (soft + neutral) */
:root {
  --sak-bg: rgba(255,255,255,0.92);
  --sak-border: rgba(49,51,63,0.14);
  --sak-border-2: rgba(49,51,63,0.10);
  --sak-shadow: 0 10px 30px rgba(0,0,0,0.06);
  --sak-radius: 18px;
  --sak-radius-sm: 12px;
  --sak-pad: 14px 16px;
  --sak-muted: rgba(49,51,63,0.70);
}

/* Titles typography */
h1, h2, h3 { letter-spacing: -0.2px; }
small, .sakiga-muted { color: var(--sak-muted); }

/* Horizontal rules */
hr { margin: 0.8rem 0; border: none; border-top: 1px solid var(--sak-border-2); }

/* ---------------------------------------------------------
   Cards
--------------------------------------------------------- */
.sakiga-card {
  border: 1px solid var(--sak-border);
  border-radius: var(--sak-radius);
  padding: var(--sak-pad);
  background: var(--sak-bg);
  box-shadow: var(--sak-shadow);
}
.sakiga-title {
  font-weight: 780;
  font-size: 1.06rem;
  margin-bottom: .35rem;
}
.sakiga-muted {
  font-size: 0.96rem;
  line-height: 1.45;
}

/* Listes dans cards */
.sakiga-ul { margin: .35rem 0 0 1.1rem; padding: 0; }
.sakiga-ul li { margin: .15rem 0; }

/* Inline code dans cards */
.sakiga-card code{
  background: rgba(49,51,63,0.08);
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 0.85em;
}

/* ---------------------------------------------------------
   Buttons (Streamlit)
--------------------------------------------------------- */
.stButton button, .stDownloadButton button {
  border-radius: 14px !important;
  padding: 0.55rem 0.9rem !important;
  border: 1px solid var(--sak-border) !important;
  box-shadow: 0 8px 18px rgba(0,0,0,0.05) !important;
}
.stButton button:hover, .stDownloadButton button:hover {
  transform: translateY(-1px);
}

/* ---------------------------------------------------------
   Inputs
--------------------------------------------------------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
  border-radius: 14px !important;
}

/* ---------------------------------------------------------
   Metrics (KPI boxes)
--------------------------------------------------------- */
div[data-testid="stMetric"]{
  border: 1px solid var(--sak-border);
  border-radius: 16px;
  padding: 12px 14px;
  background: var(--sak-bg);
  box-shadow: 0 10px 24px rgba(0,0,0,0.05);
}
div[data-testid="stMetricLabel"] p { color: var(--sak-muted); font-weight: 650; }
div[data-testid="stMetricValue"] { font-weight: 820; }

/* ---------------------------------------------------------
   Tabs
--------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"]{
  gap: 8px;
}
.stTabs [data-baseweb="tab"]{
  border-radius: 14px;
  border: 1px solid var(--sak-border);
  background: rgba(255,255,255,0.75);
  padding: 8px 14px;
}
.stTabs [aria-selected="true"]{
  box-shadow: 0 10px 22px rgba(0,0,0,0.06);
}

/* ---------------------------------------------------------
   Expanders
--------------------------------------------------------- */
details {
  border: 1px solid var(--sak-border);
  border-radius: 16px;
  padding: 8px 10px;
  background: rgba(255,255,255,0.70);
}
details > summary { cursor: pointer; font-weight: 700; color: rgba(49,51,63,0.85); }

/* ---------------------------------------------------------
   Dataframe / tables
--------------------------------------------------------- */
div[data-testid="stDataFrame"]{
  border: 1px solid var(--sak-border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 26px rgba(0,0,0,0.05);
}

/* ---------------------------------------------------------
   Sidebar polish
--------------------------------------------------------- */
section[data-testid="stSidebar"]{
  border-right: 1px solid var(--sak-border-2);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.2rem;
}

/* ---------------------------------------------------------
   Alerts
--------------------------------------------------------- */
div[data-testid="stAlert"]{
  border-radius: 16px !important;
}

/* Small utility pills */
.sakiga-pill{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--sak-border);
  background: rgba(255,255,255,0.75);
  font-size: 0.85rem;
  color: rgba(49,51,63,0.75);
}

/* Reduce top empty gap sometimes present */
header { visibility: hidden; height: 0px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _format_card_text(text: Optional[str]) -> str:
    """
    Formate un texte pour l'afficher proprement dans une card:
    - préserve tes <br/> existants
    - supporte retours à la ligne
    - supporte listes (•, - , 1) ...)
    - convertit **gras** et `code`
    """
    if not text:
        return ""

    raw = str(text)

    # 1) Préserver <br> / <br/> (compat avec ton code existant)
    br_token = "___SAKIGA_BR___"
    raw = raw.replace("<br/>", br_token).replace("<br />", br_token).replace("<br>", br_token)

    # 2) On échappe le reste pour éviter que du HTML cassant s'injecte
    t = _escape(raw)

    # 3) Restaure les BR
    t = t.replace(br_token, "\n")

    # 4) Markdown minimal
    # **bold**
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    # `code`
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)

    # 5) Listes
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and all(
        ln.startswith("•") or ln.startswith("- ") or re.match(r"^\d+\)", ln)
        for ln in lines
    ):
        items = []
        for ln in lines:
            ln = re.sub(r"^(•|\-|\d+\))\s*", "", ln)
            items.append(f"<li>{ln}</li>")
        return "<ul class='sakiga-ul'>" + "".join(items) + "</ul>"

    # 6) Texte normal: garder retours à la ligne
    return t.replace("\n", "<br/>")


def card(title: str, body: str, footer: Optional[str] = None):
    title_html = _escape(str(title))
    body_html = _format_card_text(body)

    html = (
        f"<div class='sakiga-card'>"
        f"<div class='sakiga-title'>{title_html}</div>"
        f"<div class='sakiga-muted'>{body_html}</div>"
    )
    if footer:
        footer_html = _format_card_text(footer)
        html += f"<hr/><div class='sakiga-muted'>{footer_html}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_title(title: str):
    """Compat helper utilisé par plusieurs pages."""
    st.markdown(f"### {title}")


def pill(text: str):
    """Petit badge discret (optionnel)."""
    st.markdown(f"<span class='sakiga-pill'>{_escape(str(text))}</span>", unsafe_allow_html=True)


# ----------------------------
# Utils: JSON / runs / files
# ----------------------------
def read_json(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def copy_tree(src: str, dst: str):
    """
    Copie récursive simple (sans dépendances).
    Gardée pour compat avec tes pages existantes.
    """
    srcp = Path(src)
    dstp = Path(dst)
    dstp.mkdir(parents=True, exist_ok=True)

    for item in srcp.rglob("*"):
        rel = item.relative_to(srcp)
        target = dstp / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())


def new_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run_artifacts(run_id: str, note: str = ""):
    """
    Snapshot de reports/, models/, data/processed/ dans runs/<run_id>/
    """
    base = Path("runs") / run_id
    base.mkdir(parents=True, exist_ok=True)

    meta = {"run_id": run_id, "note": note, "timestamp": datetime.datetime.now().isoformat()}
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for folder in ["reports", "models", "data/processed"]:
        fp = Path(folder)
        if fp.exists():
            copy_tree(str(fp), str(base / folder))

    return str(base)


# ----------------------------
# Compat Streamlit: image + guards
# ----------------------------
def st_image_safe(img_path: str, caption: Optional[str] = None):
    """
    Certaines versions Streamlit n'acceptent pas use_container_width.
    On gère les deux.
    """
    try:
        st.image(img_path, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img_path, caption=caption, use_column_width=True)


def require_files(paths: list[str], message: str, footer: Optional[str] = None):
    """
    Garde-fou: si un fichier clé manque, on explique et on stop la page (au lieu de planter).
    """
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        card(
            "⚠️ Page indisponible (artefacts manquants)",
            message,
            "Manquants: " + ", ".join([f"`{m}`" for m in missing]) + (f"<br/><br/>{footer}" if footer else ""),
        )
        st.stop()


# ----------------------------
# PDF report (SOUTENANCE) - intégré ici (zéro nouveau fichier)
# ----------------------------
def make_pdf_report_bytes(run_dir: Optional[str] = None, title: str = "SAKIGA DS — Rapport"):
    """
    Génère un PDF PRO à partir de:
    - runs/<run_id>/reports/ si run_dir fourni
    - sinon reports/ (latest)

    Retourne: (pdf_bytes, file_name)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception as e:
        raise RuntimeError("ReportLab manquant. Installe-le: pip install reportlab") from e

    import io

    def _rjson(p: Path):
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def _csv_head_table(p: Path, n: int = 12):
        try:
            import pandas as pd
            if not p.exists():
                return None
            df = pd.read_csv(p).head(n)
            return [list(map(str, df.columns))] + df.astype(str).values.tolist()
        except Exception:
            return None

    def _table(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    def _img(path: Path, max_w_cm: float = 17.5, max_h_cm: float = 12.0):
        if not path.exists():
            return None
        try:
            im = Image(str(path))
            max_w = max_w_cm * cm
            max_h = max_h_cm * cm
            w, h = im.imageWidth, im.imageHeight
            if w <= 0 or h <= 0:
                return None
            scale = min(max_w / w, max_h / h, 1.0)
            im.drawWidth = w * scale
            im.drawHeight = h * scale
            return im
        except Exception:
            return None

    if run_dir:
        run_path = Path(run_dir)
        base_reports = run_path / "reports"
        base_plots = base_reports / "plots"
        run_id = run_path.name
    else:
        base_reports = Path("reports")
        base_plots = base_reports / "plots"
        meta_tmp = _rjson(base_reports / "train_meta.json") or {}
        run_id = meta_tmp.get("run_id", "latest")

    train_meta = _rjson(base_reports / "train_meta.json") or {}
    metrics = _rjson(base_reports / "metrics.json") or {}

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    p = ParagraphStyle("p", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#374151"))

    pdf_bytes = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_bytes,
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=title,
    )

    story = []
    story.append(Paragraph(title, h1))
    story.append(Paragraph(f"<b>Run:</b> {run_id}", small))
    story.append(Paragraph(f"<b>Source:</b> {str(base_reports)}", small))
    story.append(Spacer(1, 10))

    task = train_meta.get("task", metrics.get("task", ""))
    y_col = train_meta.get("target_col", "")
    best_model = train_meta.get("best_model", "")
    x_cols = train_meta.get("x_cols", []) or []
    ft = train_meta.get("feature_types", {}) or {}

    story.append(Paragraph("1) Contexte & configuration", h2))
    story.append(Paragraph(
        "<br/>".join([
            f"<b>Task:</b> {task}",
            f"<b>Target (Y):</b> {y_col}",
            f"<b>Best model:</b> {best_model}",
            f"<b>Nb X:</b> {len(x_cols)}",
            f"<b>Types:</b> num={len(ft.get('num', []))} | cat={len(ft.get('cat', []))} | text={len(ft.get('text', []))}",
        ]),
        p
    ))
    story.append(Spacer(1, 8))

    split = train_meta.get("split", {}) or {}
    if split:
        story.append(Paragraph("Split train / val / test", h2))
        tdata = [
            ["train_rows", "val_rows", "test_rows", "warning"],
            [str(split.get("train_rows","")), str(split.get("val_rows","")), str(split.get("test_rows","")),
             (split.get("split_warning") or "")[:140]],
        ]
        story.append(_table(tdata, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 6.5*cm]))
        story.append(Spacer(1, 8))

    comp = train_meta.get("comparison") or []
    if comp:
        story.append(Paragraph("2) Comparaison des modèles", h2))
        rows = [["model", "val_primary", "cv_best_score", "error"]]
        for r in comp[:10]:
            rows.append([
                str(r.get("model","")),
                str(r.get("val_primary_metric","")),
                str(r.get("cv_best_score","")),
                (str(r.get("error",""))[:80] if r.get("error") else "")
            ])
        story.append(_table(rows, col_widths=[4*cm, 3.2*cm, 3.2*cm, 7.2*cm]))
        story.append(Spacer(1, 8))

    pred = _csv_head_table(base_reports / "predictions_test.csv", n=10)
    if pred:
        story.append(Paragraph("3) Aperçu predictions_test.csv", h2))
        story.append(_table(pred))
        story.append(Spacer(1, 8))

    plot_files = []
    if base_plots.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            plot_files.extend(base_plots.glob(ext))
    plot_files = sorted(plot_files, key=lambda x: x.name)

    if plot_files:
        story.append(PageBreak())
        story.append(Paragraph("4) Plots", h2))
        for pf in plot_files[:12]:
            story.append(Paragraph(f"<b>{pf.name}</b>", small))
            im = _img(pf)
            if im:
                story.append(im)
                story.append(Spacer(1, 10))

    doc.build(story)
    pdf_bytes.seek(0)
    return pdf_bytes.read(), f"report_{run_id}.pdf"
