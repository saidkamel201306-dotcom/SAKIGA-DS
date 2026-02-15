# Rapport Bonus (WAOUH)

## A) Choix de métriques (plus pro que l’accuracy)
- Accuracy : OK mais peut masquer certaines erreurs.
- Macro‑F1 : robuste même si léger déséquilibre MEN/WOMEN.
- ROC‑AUC / PR : quand le modèle expose des scores (proba/decision_function).

➡️ Les résultats sont dans :
- `reports/metrics.json`
- `reports/metrics.yaml`
- `reports/plots/`

## B) Threshold tuning (démarche “métier”)
Au lieu de prendre la prédiction par défaut, on choisit un **seuil** optimisé sur **validation** pour maximiser **macro‑F1**.
- Seuil choisi : `threshold_tuning.validation_best.threshold`
- Score test après seuil : `threshold_tuning.test_after_threshold.test_f1_macro`

## C) Leakage audit (anti “triche”)
On compare :
- **Naïf** (texte brut, risque de fuite)
- **Nettoyé** (suppression tokens men/women/boys/girls/unisex/kids)

➡️ Notebook : `notebook_03_WAOUH.ipynb` (section Leakage audit)

## D) Conclusion
Ce bonus montre :
- rigueur scientifique (leakage)
- métriques avancées + interprétation
- alignement métier (seuil)
