# Recommandation personnalisée (WAOUH)

## Problème (métier)
L’application s’adapte à l’utilisateur : après plusieurs consultations, on affiche une section **"Pour vous"**.

## Données
- Produits: `styles.csv`
- Historique user: liste d’IDs consultés (ex: `viewedItemIds`), ou export `events.csv`

## Méthode (content-based)
1) Construire un texte produit: (name + subCategory + articleType) avec nettoyage anti-leakage.
2) Vectoriser TF‑IDF → chaque produit devient un vecteur.
3) Profil user = moyenne des vecteurs des produits consultés récemment.
4) Recommandations = top‑K produits les plus similaires (cosine similarity), en excluant déjà vus.

## Déploiement (API)
Endpoint:
- `POST /recommend`  body: `{ userId, viewedItemIds: [..], k }`

Build index:
```bash
python -m src.reco.build_item_index
```

Lancer API:
```bash
uvicorn api.main:app --reload --port 8099
```

## Évaluation (bonus)
Si tu as `data/history/events.csv` (user_id,item_id,ts):
```bash
python -m src.reco.evaluate_reco
```
Sortie: `reports/reco_eval.json` (Precision@K / Recall@K)

## Lien avec le cahier des charges
- Problématique ML + dataset + features
- Modèle (similarité / KNN-like)
- Évaluation (Precision@K)
- Déploiement web (API)
