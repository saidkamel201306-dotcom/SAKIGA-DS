# -*- coding: utf-8 -*-
"""
Offline evaluation for the recommender (Precision@K / Recall@K).

Input (optional):
- data/history/events.csv with columns: user_id,item_id,ts
  where ts is sortable (timestamp or integer).
If file not found, we explain how to create it from your ms-user history.

This script is designed for the report:
- "We measured Precision@K with temporal split per user."

Output:
- reports/reco_eval.json
"""
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np

from src.reco.reco_service import load_index, recommend_from_viewed

INDEX_PATH = os.path.join("models","reco_item_index.pkl")
EVENTS_PATH = os.path.join("data","history","events.csv")
OUT_PATH = os.path.join("reports","reco_eval.json")

def precision_recall_at_k(recommended, ground_truth_set):
    if not recommended:
        return 0.0, 0.0
    rec_set = set(recommended)
    hits = len(rec_set & ground_truth_set)
    prec = hits / len(rec_set)
    rec = hits / max(1, len(ground_truth_set))
    return prec, rec

def main():
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(EVENTS_PATH):
        info = {
            "error": "events.csv_not_found",
            "expected_path": EVENTS_PATH,
            "how_to_create": "Export browsing history events with columns user_id,item_id,ts (ts sortable).",
        }
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("⚠️ events.csv not found. Wrote instructions to", OUT_PATH)
        return

    ev = pd.read_csv(EVENTS_PATH)
    for c in ["user_id","item_id","ts"]:
        if c not in ev.columns:
            raise ValueError(f"Missing column '{c}' in events.csv")

    ev = ev.dropna(subset=["user_id","item_id","ts"]).copy()
    ev["ts"] = pd.to_datetime(ev["ts"], errors="ignore")
    # sort per user
    ev = ev.sort_values(["user_id","ts"])

    index = load_index(INDEX_PATH)

    K = 10
    per_user = []
    for user_id, grp in ev.groupby("user_id"):
        items = grp["item_id"].astype(int).tolist()
        if len(items) < 5:
            continue
        split = int(len(items) * 0.8)
        seen = items[:split]
        future = set(items[split:])  # ground truth

        recs = recommend_from_viewed(index, seen[-20:], k=K)
        rec_ids = [int(r[index.id_col]) for r in recs if index.id_col in r]

        p, r = precision_recall_at_k(rec_ids, future)
        per_user.append({"user_id": str(user_id), "precision@k": p, "recall@k": r, "n_events": len(items)})

    if not per_user:
        out = {"warning":"not_enough_history", "users_evaluated": 0}
    else:
        out = {
            "users_evaluated": len(per_user),
            "avg_precision@k": float(np.mean([u["precision@k"] for u in per_user])),
            "avg_recall@k": float(np.mean([u["recall@k"] for u in per_user])),
            "k": K,
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "per_user": per_user[:200]}, f, ensure_ascii=False, indent=2)

    print("✅ Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
