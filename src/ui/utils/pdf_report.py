# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _safe_read_csv_head(p: Path, n: int = 15) -> Optional[List[List[str]]]:
    try:
        import pandas as pd
        if not p.exists():
            return None
        df = pd.read_csv(p).head(n)
        # convert to list of lists (strings)
        return [list(map(str, df.columns))] + df.astype(str).values.tolist()
    except Exception:
        return None


def _table(data: List[List[str]], col_widths=None) -> Table:
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


def _img(path: Path, max_w_cm: float = 17.5, max_h_cm: float = 12.0) -> Optional[Image]:
    if not path.exists():
        return None
    try:
        img = Image(str(path))
        # scale within bounds
        max_w = max_w_cm * cm
        max_h = max_h_cm * cm
        w, h = img.imageWidth, img.imageHeight
        if w <= 0 or h <= 0:
            return None
        scale = min(max_w / w, max_h / h, 1.0)
        img.drawWidth = w * scale
        img.drawHeight = h * scale
        return img
    except Exception:
        return None


def build_run_pdf_report(
    run_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
    title: str = "SAKIGA DS — Rapport",
) -> Path:
    """
    Generate a PDF report either:
    - for a given run_dir (runs/run_...),
    - or fallback to latest reports/ folder.

    Returns path to the created PDF.
    """
    run_dir = Path(run_dir) if run_dir else None

    # Resolve base folders (run preferred)
    if run_dir and run_dir.exists():
        base_reports = run_dir / "reports"
        base_plots = base_reports / "plots"
        train_meta_path = base_reports / "train_meta.json"
        metrics_path = base_reports / "metrics.json"
        pred_path = base_reports / "predictions_test.csv"
        cm_path = base_reports / "confusion_matrix.csv"
        run_id = run_dir.name
    else:
        base_reports = Path("reports")
        base_plots = base_reports / "plots"
        train_meta_path = base_reports / "train_meta.json"
        metrics_path = base_reports / "metrics.json"
        pred_path = base_reports / "predictions_test.csv"
        cm_path = base_reports / "confusion_matrix.csv"
        meta_tmp = _read_json(train_meta_path) or {}
        run_id = meta_tmp.get("run_id", "latest")

    if out_path is None:
        # put PDF inside the same reports folder (run or latest)
        out_path = base_reports / f"report_{run_id}.pdf"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    p = ParagraphStyle("p", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#374151"))

    doc = SimpleDocTemplate(
        str(out_path),
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
    story.append(Paragraph(f"<b>Base reports:</b> {str(base_reports)}", small))
    story.append(Spacer(1, 10))

    # ---- Load artifacts
    train_meta = _read_json(train_meta_path) or {}
    metrics = _read_json(metrics_path) or {}

    # ---- Section: Dataset / Config
    story.append(Paragraph("1) Contexte & configuration", h2))
    task = train_meta.get("task", metrics.get("task", ""))
    y_col = train_meta.get("target_col", "")
    best_model = train_meta.get("best_model", "")
    x_cols = train_meta.get("x_cols", []) or []
    ft = train_meta.get("feature_types", {}) or {}

    lines = [
        f"<b>Task:</b> {task}",
        f"<b>Target (Y):</b> {y_col}",
        f"<b>Best model:</b> {best_model}",
        f"<b>Nb X:</b> {len(x_cols)}",
        f"<b>Types:</b> num={len(ft.get('num', []))} | cat={len(ft.get('cat', []))} | text={len(ft.get('text', []))}",
    ]
    story.append(Paragraph("<br/>".join(lines), p))
    story.append(Spacer(1, 8))

    split = train_meta.get("split", {}) or {}
    if split:
        story.append(Paragraph("Split train / val / test", h2))
        tdata = [
            ["train_rows", "val_rows", "test_rows", "warnings"],
            [str(split.get("train_rows", "")), str(split.get("val_rows", "")), str(split.get("test_rows", "")),
             (split.get("split_warning") or "")[:120]],
        ]
        story.append(_table(tdata, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 6.5*cm]))
        story.append(Spacer(1, 8))

    # ---- Section: EDA (if exists)
    eda_summary = _read_json(base_reports / "eda_summary.json")
    if eda_summary:
        story.append(Paragraph("2) EDA & pré-traitement", h2))
        shp = eda_summary.get("shape", {})
        story.append(Paragraph(
            f"<b>Shape:</b> rows={shp.get('rows','')} cols={shp.get('cols','')}<br/>"
            f"<b>Y:</b> {eda_summary.get('target_col','')}<br/>"
            f"<b>Nb X:</b> {len(eda_summary.get('x_cols',[]) or [])}",
            p
        ))
        story.append(Spacer(1, 6))

        miss_table = _safe_read_csv_head(base_reports / "eda_missing.csv", n=12)
        if miss_table:
            story.append(Paragraph("Missing values (top)", small))
            story.append(_table(miss_table))
            story.append(Spacer(1, 6))

        outl_table = _safe_read_csv_head(base_reports / "eda_outliers.csv", n=12)
        if outl_table:
            story.append(Paragraph("Outliers IQR (top)", small))
            story.append(_table(outl_table))
            story.append(Spacer(1, 6))

    # ---- Section: model comparison
    comp = train_meta.get("comparison") or []
    if comp:
        story.append(Paragraph("3) Comparaison des modèles", h2))
        rows = [["model", "val_primary", "cv_best_score", "error"]]
        for r in comp[:10]:
            rows.append([
                str(r.get("model", "")),
                str(r.get("val_primary_metric", "")),
                str(r.get("cv_best_score", "")),
                (str(r.get("error", ""))[:80] if r.get("error") else ""),
            ])
        story.append(_table(rows, col_widths=[4*cm, 3.2*cm, 3.2*cm, 7.2*cm]))
        story.append(Spacer(1, 8))
        justif = train_meta.get("justification")
        if justif:
            story.append(Paragraph("<b>Justification :</b> " + str(justif), p))
            story.append(Spacer(1, 6))

    # ---- Section: results / metrics
    story.append(Paragraph("4) Résultats (test) & exports", h2))

    # metrics quick table
    metric_rows = [["key", "value"]]
    for k in ["accuracy", "f1_macro", "roc_auc", "r2", "mae", "rmse"]:
        if k in metrics:
            metric_rows.append([k, str(metrics.get(k))])
    if len(metric_rows) > 1:
        story.append(_table(metric_rows, col_widths=[5*cm, 12*cm]))
        story.append(Spacer(1, 8))

    # predictions preview
    pred_table = _safe_read_csv_head(pred_path, n=12)
    if pred_table:
        story.append(Paragraph("Aperçu predictions_test.csv", small))
        story.append(_table(pred_table))
        story.append(Spacer(1, 8))

    # confusion matrix table preview (if exists)
    cm_preview = _safe_read_csv_head(cm_path, n=12)
    if cm_preview:
        story.append(Paragraph("Confusion matrix (CSV)", small))
        story.append(_table(cm_preview))
        story.append(Spacer(1, 8))

    # ---- Section: plots gallery
    plot_files = []
    if base_plots.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            plot_files.extend(base_plots.glob(ext))
    plot_files = sorted(plot_files, key=lambda x: x.name)

    if plot_files:
        story.append(PageBreak())
        story.append(Paragraph("5) Plots", h2))
        for pf in plot_files[:12]:  # limit to avoid huge PDF
            story.append(Paragraph(f"<b>{pf.name}</b>", small))
            im = _img(pf)
            if im:
                story.append(im)
                story.append(Spacer(1, 10))

    doc.build(story)
    return out_path
