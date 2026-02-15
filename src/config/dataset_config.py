# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

DEFAULT_CONFIG_PATH = Path("data") / "config" / "dataset_config.json"

# -------------------------
# Helpers
# -------------------------
def _norm_path_str(p: str) -> str:
    """
    Normalize paths so they work both on Windows and Linux containers:
    - convert backslashes to forward slashes
    - strip spaces
    """
    if p is None:
        return ""
    return str(p).strip().replace("\\", "/")

def _first_existing(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def _smart_defaults() -> Dict[str, Any]:
    """
    Smart defaults when config doesn't exist yet.

    Priority:
    1) user_dataset.csv (generic upload mode)
    2) styles.csv (compat with your old demo)
    3) dataset.csv (generic fallback)
    """
    user_csv = "data/raw/user_dataset.csv"
    styles = "data/raw/styles.csv"
    generic = "data/raw/dataset.csv"

    chosen = _first_existing([user_csv, styles, generic]) or generic

    # Keep minimal safe defaults: user will pick Y in UI if not known
    defaults = {
        "raw_csv": chosen,
        "sep": ",",
        "target_col": "",          # empty => user chooses in UI
        "task": "classification",  # stable default
        "id_col": None,
        "text_col": None,
        "feature_cols": [],
        "drop_cols": [],
    }

    # If styles.csv exists, keep compatibility but still not mandatory
    if chosen == styles:
        defaults.update(
            {
                "sep": ";",
                "target_col": "gender",
                "task": "classification",
                "id_col": "id",
                "text_col": "productDisplayName",
            }
        )

    return defaults


@dataclass
class DatasetConfig:
    # Dataset location
    raw_csv: str = "data/raw/user_dataset.csv"
    sep: str = ","

    # ML target / features
    target_col: str = ""                 # empty => not set
    feature_cols: Optional[List[str]] = None  # None/[] => auto
    task: str = "classification"         # classification | regression

    # Optional helpers
    id_col: Optional[str] = None
    text_col: Optional[str] = None
    drop_cols: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)

        # Keep backward compatibility with your existing pipeline:
        # feature_cols == [] means "auto"
        if d["feature_cols"] is None:
            d["feature_cols"] = []
        if d["drop_cols"] is None:
            d["drop_cols"] = []

        # Normalize empty strings to keep JSON clean
        if d.get("target_col") is None:
            d["target_col"] = ""
        if d.get("id_col") == "":
            d["id_col"] = None
        if d.get("text_col") == "":
            d["text_col"] = None

        # IMPORTANT: normalize path for cross-platform
        d["raw_csv"] = _norm_path_str(d.get("raw_csv", ""))

        return d


def ensure_config_dir() -> Path:
    p = DEFAULT_CONFIG_PATH.parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset_config(path: Path = DEFAULT_CONFIG_PATH) -> DatasetConfig:
    """
    Loads dataset_config.json.
    If not found, creates one with smart defaults.

    Cross-platform behavior:
    - converts Windows path separators to Linux-friendly
    - if configured CSV doesn't exist, falls back to user_dataset.csv (generic mode)
    """
    if not path.exists():
        ensure_config_dir()
        defaults = _smart_defaults()
        cfg = DatasetConfig(
            raw_csv=_norm_path_str(defaults["raw_csv"]),
            sep=defaults["sep"],
            target_col=defaults["target_col"],
            feature_cols=defaults.get("feature_cols") or [],
            task=defaults.get("task", "classification"),
            id_col=defaults.get("id_col"),
            text_col=defaults.get("text_col"),
            drop_cols=defaults.get("drop_cols") or [],
        )
        save_dataset_config(cfg, path)
        return cfg

    data = json.loads(path.read_text(encoding="utf-8"))

    raw_csv = _norm_path_str(data.get("raw_csv") or "data/raw/user_dataset.csv")
    sep = data.get("sep") or ","
    target_col = data.get("target_col") or ""

    # If relative and doesn't exist => fallback to a dataset that exists (generic mode)
    raw_path = Path(raw_csv)
    if not raw_path.is_absolute() and not raw_path.exists():
        fallback = _first_existing(
            [
                "data/raw/user_dataset.csv",
                "data/raw/dataset.csv",
                "data/raw/styles.csv",
            ]
        )
        if fallback:
            raw_csv = fallback

    cfg = DatasetConfig(
        raw_csv=raw_csv,
        sep=sep,
        target_col=target_col,
        feature_cols=data.get("feature_cols") or [],
        task=data.get("task", "classification"),
        id_col=data.get("id_col") if data.get("id_col") not in ["", "null"] else None,
        text_col=data.get("text_col") if data.get("text_col") not in ["", "null"] else None,
        drop_cols=data.get("drop_cols") or [],
    )

    # Auto-heal: if we changed the path normalization/fallback, persist it
    healed_raw = _norm_path_str(cfg.raw_csv)
    if data.get("raw_csv") != healed_raw:
        cfg.raw_csv = healed_raw
        save_dataset_config(cfg, path)

    return cfg


def save_dataset_config(cfg: DatasetConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    ensure_config_dir()
    path.write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
