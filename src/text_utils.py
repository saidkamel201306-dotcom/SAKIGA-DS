# -*- coding: utf-8 -*-
"""
Text utilities used consistently for training + inference (no leakage).

We remove explicit gender tokens from product names to avoid target leakage:
MEN/WOMEN/BOYS/GIRLS/UNISEX/KIDS, etc.
"""
from __future__ import annotations
import re

# tokens to remove (case-insensitive) - keep it simple and defensible for report
_LEAK_TOKENS = [
    r"\bmen\b", r"\bman's\b", r"\bman\b",
    r"\bwomen\b", r"\bwoman's\b", r"\bwoman\b",
    r"\bboys?\b", r"\bgirls?\b",
    r"\bunisex\b",
    r"\bkids?\b", r"\bchild(?:ren)?\b", r"\byouth\b",
]

_LEAK_RE = re.compile("|".join(_LEAK_TOKENS), flags=re.IGNORECASE)

_WS_RE = re.compile(r"\s+")

def clean_leakage(text: str) -> str:
    """Remove obvious target-leakage words and normalize whitespace."""
    if text is None:
        return ""
    s = str(text)
    s = _LEAK_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def build_model_text(productDisplayName: str, subCategory: str, articleType: str) -> str:
    """Build the exact text used by the model (train + inference)."""
    p = clean_leakage(productDisplayName)
    s = clean_leakage(subCategory)
    a = clean_leakage(articleType)
    return f"{p} {s} {a}".strip()
