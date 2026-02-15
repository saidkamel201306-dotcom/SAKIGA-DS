# v10 — Alignement modèles (cours du prof / DataCamp)

Le cahier des charges demande de tester 3 modèles. En v10, l'entraînement compare:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Le résultat (scores + best_params) est sauvegardé dans:
- reports/train_meta.json
et repris dans:
- reports/metrics.json (train_meta)

Important:
- XGBoost nécessite: pip install xgboost==2.1.1
