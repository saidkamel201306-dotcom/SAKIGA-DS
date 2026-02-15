# Pack v9 — 'Beau + Bon + Waouh + Complet' (intégré dans l'app)

Ce pack garde l'app Streamlit pro (v8) et ajoute **2 pages MLOps** intégrées :
- **⚙️ MLOps Cockpit** : params.yaml, DVC (dvc repro), MLflow UI, Evidently report
- **🌐 API Live Tests** : démarrage API + tests /predict et /recommend

## Installation
Base:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optionnel MLOps (recommandé si tu veux ressembler à l'exemple du prof):
```bash
pip install -r requirements_mlops.txt
```
## Démo vidéo (script)
1) Home (Mode Démo Prof)
2) Résultats (KPI + plots)
3) Reco Live Demo (explication historique → Pour vous)
4) ⚙️ MLOps Cockpit (montrer dvc repro / Evidently / MLflow)
5) 🌐 API Live Tests (health + predict + recommend)
