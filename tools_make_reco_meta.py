import json
from pathlib import Path
import pandas as pd

Path("models").mkdir(exist_ok=True)

styles_path = Path("data/raw/styles.csv")
sample_ids = []
cols = []

if styles_path.exists():
    s = pd.read_csv(styles_path, sep=";")
    cols = list(s.columns)
    if "id" in s.columns:
        sample_ids = s["id"].dropna().astype(int).head(20).tolist()

meta = {
    "id_col": "id",
    "sample_ids": sample_ids[:10],
    "styles_columns": cols[:20],
    "note": "Meta créé automatiquement (sep=';') pour aider la page Reco (IDs exemples)."
}

Path("models/reco_index_meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("✅ Created models/reco_index_meta.json")
print(meta)
