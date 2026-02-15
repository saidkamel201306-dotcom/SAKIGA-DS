# -*- coding: utf-8 -*-
"""
Quick demo for video:
1) Build reco index
2) Run a local recommendation query (no API)
"""
from __future__ import annotations
import random, joblib
from src.reco.reco_service import load_index, recommend_from_viewed

def main():
    idx = load_index("models/reco_item_index.pkl")
    ids = idx.items_df[idx.id_col].astype(int).tolist()
    viewed = random.sample(ids, 3)
    print("Viewed items:", viewed)
    recs = recommend_from_viewed(idx, viewed_item_ids=viewed, k=10)
    print("Top 5 recommendations:")
    for r in recs[:5]:
        print("-", r.get("productDisplayName"), "|", r.get("subCategory"), "| score=", round(float(r.get("score",0.0)), 4))

if __name__ == "__main__":
    main()
