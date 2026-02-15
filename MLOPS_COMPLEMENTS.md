# Compléments MLOps (version “comme la vidéo”)

Tu as raison : la vidéo montre souvent **DVC / Docker / CI/CD / (parfois MLflow/Evidently/Airflow)**.

Dans ce projet SAKIGA, on a ajouté une **version réaliste mais pas lourde** :

## Inclus dans le pack (concret)
✅ **DVC-like pipeline** (fichier `dvc.yaml`) avec 3 stages :
- `prepare_data` → crée `data/processed/dataset.csv`
- `train` → entraîne + GridSearch → `models/best_model.pkl`
- `evaluate` → métriques test → `reports/metrics.yaml` + `reports/metrics.json`

> Si vous installez DVC, vous pourrez lancer `dvc repro`.
> Sinon, vous pouvez lancer les stages avec Python (voir commandes ci-dessous).

✅ **Docker** : `Dockerfile` pour containeriser l’API FastAPI.

✅ **CI/CD (GitHub Actions)** : `.github/workflows/ci.yml` exécute le pipeline sur chaque push.

## Pas inclus (bonus)
- MLflow tracking
- Evidently monitoring
- Airflow orchestration

➡️ Pourquoi ? Parce que ce sont des *bonus* dans beaucoup de cahiers des charges.
Si ton prof l’exige vraiment, on peut les ajouter après.

## Commandes rapides
Sans DVC :
```bash
python -m src.pipeline.prepare_data
python -m src.pipeline.train
python -m src.pipeline.evaluate
```

Avec DVC (si installé) :
```bash
dvc repro
```

Docker API :
```bash
docker build -t sakiga-api .
docker run -p 8099:8099 sakiga-api
```


## Bonus Reco (v5)
- `python -m src.reco.build_item_index` (build index)
- Endpoint FastAPI `/recommend`
- Offline eval `python -m src.reco.evaluate_reco`
