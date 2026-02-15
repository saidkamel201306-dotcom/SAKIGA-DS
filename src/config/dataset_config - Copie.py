# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

DEFAULT_CONFIG_PATH = Path("data") / "config" / "dataset_config.json"

# -------------------------
# Helpers (safe defaults)
# -------------------------
def _smart_defaults() -> Dict[str, Any]:
    """
    If there is no config yet, we pick sane defaults:
    - If styles.csv exists => keep your old project working out-of-the-box
    - Else => point to a generic dataset path and force user to pick Y later
    """
    styles = Path("data") / "raw" / "styles.csv"
    generic = Path("data") / "raw" / "dataset.csv"

    if styles.exists():
        return {
            "raw_csv": str(styles),
            "sep": ";",
            "target_col": "gender",
            "task": "classification",
            "id_col": "id",
            "text_col": "productDisplayName",
            "feature_cols": [],
            "drop_cols": [],
        }

    # Generic mode
    return {
        "raw_csv": str(generic),
        "sep": ",",
        "target_col": "",          # empty => user must choose in UI
        "task": "classification",  # keep stable default (can be made "auto" later in UI)
        "id_col": None,
        "text_col": None,
        "feature_cols": [],        # empty list => auto (compat with existing code)
        "drop_cols": [],
    }


@dataclass
class DatasetConfig:
    # Dataset location
    raw_csv: str = "data/raw/dataset.csv"
    sep: str = ","

    # ML target / features
    target_col: str = ""                 # empty => not set
    feature_cols: List[str] = None       # None/[] => auto (we save [] for compatibility)
    task: str = "classification"         # classification | regression

    # Optional helpers (can be None)
    id_col: Optional[str] = None
    text_col: Optional[str] = None       # optional single text column
    drop_cols: List[str] = None

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

        return d


def ensure_config_dir() -> Path:
    p = DEFAULT_CONFIG_PATH.parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset_config(path: Path = DEFAULT_CONFIG_PATH) -> DatasetConfig:
    """
    Loads dataset_config.json.
    If not found, creates one with smart defaults (styles.csv if present, else generic).
    """
    if not path.exists():
        ensure_config_dir()
        defaults = _smart_defaults()
        cfg = DatasetConfig(
            raw_csv=defaults["raw_csv"],
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

    # Migration-friendly reading (old configs still work)
    raw_csv = data.get("raw_csv") or "data/raw/dataset.csv"
    sep = data.get("sep") or ","
    target_col = data.get("target_col") or ""

    return DatasetConfig(
        raw_csv=raw_csv,
        sep=sep,
        target_col=target_col,
        feature_cols=data.get("feature_cols") or [],
        task=data.get("task", "classification"),
        id_col=data.get("id_col") if data.get("id_col") not in ["", "null"] else None,
        text_col=data.get("text_col") if data.get("text_col") not in ["", "null"] else None,
        drop_cols=data.get("drop_cols") or [],
    )


def save_dataset_config(cfg: DatasetConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    ensure_config_dir()
    path.write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
