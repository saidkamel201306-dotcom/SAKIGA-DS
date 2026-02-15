import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

styles = pd.read_csv("data/raw/styles.csv", sep=";")
styles["id"] = styles["id"].astype(int)

# Texte utilisé pour la reco (tu peux l'améliorer après)
text = (
    styles["productDisplayName"].fillna("").astype(str) + " | " +
    styles["subCategory"].fillna("").astype(str) + " | " +
    styles["articleType"].fillna("").astype(str) + " | " +
    styles["baseColour"].fillna("").astype(str)
)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = vectorizer.fit_transform(text)

id_to_idx = {int(pid): int(i) for i, pid in enumerate(styles["id"].tolist())}

index = {
    "id_col": "id",
    "id_to_idx": id_to_idx,
    "vectorizer": vectorizer,
    "X": X,
    "topk_default": 10
}

Path("models").mkdir(exist_ok=True)
with open("models/reco_item_index.pkl", "wb") as f:
    pickle.dump(index, f)

meta = {
    "id_col": "id",
    "n_items": int(styles.shape[0]),
    "sample_ids": styles["id"].head(10).astype(int).tolist(),
    "note": "Reco index TF-IDF (productDisplayName+meta) généré localement."
}
Path("models/reco_index_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("✅ reco_item_index.pkl regenerated")
print("✅ reco_index_meta.json updated")
print("sample_ids:", meta["sample_ids"])
