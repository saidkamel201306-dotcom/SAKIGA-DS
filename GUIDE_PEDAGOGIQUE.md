# Guide pédagogique — SAKIGA Data Science (conforme cahier des charges)

Ce guide explique **ce qu’on fait**, **pourquoi**, et **où ça correspond au cahier des charges du prof**.

---

## 1) Le contexte du projet (métier)
On a une application e‑commerce (faite dans une autre matière).  
Dans l’app, on a besoin d’orienter automatiquement les produits vers le bon univers :

- **MEN** (homme)
- **WOMEN** (femme)

👉 Donc notre **problématique Data Science** est :

> **Prédire le genre (MEN/WOMEN) d’un produit à partir de son texte** (nom + catégorie + type).

Cela permet :
- d’automatiser le “dispatch” (répartition) des produits en base MEN/WOMEN,
- d’améliorer l’organisation du catalogue et l’expérience utilisateur.

---

## 2) Le dataset (Kaggle)
Le dataset est `styles.csv` (fashion products).  
On utilise principalement :
- `productDisplayName` (nom)
- `subCategory` (sous-catégorie)
- `articleType` (type)
- `gender` (la vérité terrain)

✅ **Cahier des charges :** expliquer les données, les variables et la cible.

---

## 3) Attention : le “hic” (anti‑triche / leakage)
Beaucoup de `productDisplayName` contiennent directement “Men/Women”.  
Si on garde ça, le modèle peut “tricher” (target leakage) et avoir une accuracy artificiellement élevée.

✅ Notre solution (propre académiquement) :
- on **supprime** dans le texte les tokens explicites :  
  `men, women, boys, girls, unisex, kids, …`
- même nettoyage en **training** et en **prediction**

➡️ Fichier concerné : `src/text_utils.py`

✅ **Cahier des charges :** montrer une démarche DS correcte (préprocessing sérieux).

---

## 4) Étapes du projet (vue globale)
### Étape A — Préparation & features
On construit une feature texte :
- `text = productDisplayName + subCategory + articleType`  
(après nettoyage anti-leakage)

### Étape B — Split des données
On fait un split **stratifié** :
- Train 70%
- Validation 15%
- Test 15%

✅ **Cahier des charges :** train/validation/test.

### Étape C — Modélisation (au moins 3 algorithmes)
On compare 3 modèles “baseline solides” pour du texte :
1) Logistic Regression
2) Linear SVC
3) Multinomial Naive Bayes

✅ **Cahier des charges :** minimum 3 algorithmes + comparaison.

### Étape D — Pipeline + GridSearchCV
Chaque modèle est dans un `Pipeline` :
- TF‑IDF (vectorisation)
- Classifier

Puis on fait `GridSearchCV` pour optimiser :
- `tfidf__ngram_range`
- `tfidf__max_features`
- paramètres du modèle (`C`, `alpha`, …)

✅ **Cahier des charges :** Pipeline + GridSearchCV.

### Étape E — Évaluation
On choisit le meilleur modèle selon **macro‑F1** sur validation, puis on teste une seule fois sur le **test set**.

Outputs :
- `models/best_model.pkl` (modèle final)
- `reports/metrics.json` (scores + matrice de confusion)

✅ **Cahier des charges :** métriques + justification du meilleur modèle.

---

## 5) Où est le code pour chaque étape ?
### Entraînement complet
👉 `train_gender_model.py`

Commande :
```bash
python train_gender_model.py
```

### Déploiement Web (API)
👉 `api/main.py` (FastAPI)

Commande :
```bash
uvicorn api.main:app --reload --port 8099
```

Test rapide :
- `GET /health`
- `POST /predict`

Exemple payload :
```json
{
  "productDisplayName": "Elegant summer dress",
  "subCategory": "Apparel",
  "articleType": "Dress"
}
```

✅ **Cahier des charges :** déploiement web.

### (Optionnel) Dispatch vers MySQL
👉 `dispatch_products_asamiga_v3_noleak.py`

But : lire `db_asamiga_main.products_main` et insérer automatiquement dans
- `db_asamiga_men.products_men`
- `db_asamiga_women.products_women`

⚠️ Optionnel pour la note DS : utile si on veut montrer l’intégration à l’app.

---

## 6) “Checklist” cahier des charges (preuve de conformité)
- [x] Problématique + cible (classification)
- [x] Dataset + features + nettoyage
- [x] Split train/val/test
- [x] 3 modèles comparés
- [x] Pipeline scikit‑learn
- [x] GridSearchCV
- [x] Évaluation (macro‑F1 + confusion matrix)
- [x] Modèle sauvegardé `.pkl`
- [x] Déploiement web (FastAPI)

---

## 7) Conseils pour la vidéo 5 minutes (structure)
1. Problème & app e‑commerce (20s)
2. Dataset & features (40s)
3. Modèles testés + GridSearch (60s)
4. Résultats (30s)
5. Démo API `/predict` (90s)
6. Conclusion & limites (40s)

---

## 8) Installation rapide (pour les binômes)
Windows PowerShell :
```bash
cd sakiga_ds_deliverable
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_gender_model.py
uvicorn api.main:app --reload --port 8099
```

C’est tout ✅


---

## 9) Bonus WAOUH (v4)
Ce pack ajoute : ROC/AUC, PR curve, threshold tuning, plots et un notebook de leakage audit.
- Lance `python -m src.pipeline.evaluate` puis ouvre `notebook_03_WAOUH.ipynb`.

---

## 10) Le "VRAI WAOUH" (v6) : reco connectée à l'historique réel (ms-user)
Avant, `/recommend` acceptait surtout une liste `viewedItemIds`.
Maintenant, **si `viewedItemIds` est vide**, l'API va chercher l'historique réel :

- ms-user : `GET /api/users/{userId}/history/recent-products?limit=...`
- puis calcule les recommandations via similarité TF-IDF

### Pourquoi ça vous démarque
- Démo vidéo claire : *je consulte 3 produits -> "Pour vous" change*.
- Lien cours/ML : profil utilisateur + cosine similarity (KNN-like).
- Lien cahier des charges : **déploiement web** + **utilisation d'une donnée réelle** (historique).

### Pour activer (si ms-user n'est pas sur 8083)
Définir une variable d'environnement :
- Windows PowerShell:
  `setx USER_API "http://localhost:8083"`
  puis relancer le terminal
- Ou en session:
  `$env:USER_API="http://localhost:8083"`
