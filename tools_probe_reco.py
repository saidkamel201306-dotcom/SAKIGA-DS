import time, pickle
import pandas as pd

t0 = time.time()
print("Loading styles...")
styles = pd.read_csv("data/raw/styles.csv", sep=";")
print("styles rows:", len(styles), "cols:", list(styles.columns)[:8])

print("Loading index pickle...")
t1 = time.time()
with open("models/reco_item_index.pkl", "rb") as f:
    idx = pickle.load(f)
print("pickle loaded in", round(time.time()-t1,2), "s")
print("index type:", type(idx))

# Inspect structure safely
if isinstance(idx, dict):
    print("dict keys:", list(idx.keys())[:20])
    # common patterns:
    for k in ["id_col","ids","item_ids","df","items","vectors","embeddings","vectorizer","tfidf","matrix","X"]:
        if k in idx:
            v = idx[k]
            print("found key:", k, "type:", type(v))
else:
    print("repr head:", repr(idx)[:200])

# Check that your IDs exist in styles
ids_to_test = [21379, 53759, 1855]
found = styles[styles["id"].isin(ids_to_test)] if "id" in styles.columns else None
print("IDs found in styles:", 0 if found is None else len(found))
if found is not None and len(found)>0:
    print(found[["id","productDisplayName","subCategory","articleType"]].head(5).to_string(index=False))

print("DONE in", round(time.time()-t0,2), "s")
