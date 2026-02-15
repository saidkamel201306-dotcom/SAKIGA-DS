# SAKIGA‑DS — Application Data Science (Streamlit) + Bonus MLOps

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

**SAKIGA‑DS** est une application **Streamlit** qui automatise un pipeline Machine Learning **de bout en bout** :  
**Data → EDA/Pré‑traitement → Modélisation (3 modèles) → GridSearchCV → Résultats → Prédiction → Export & Historique**.

> ✅ Conforme aux exigences du cahier des charges : EDA, split train/val/test, comparaison d’au moins 3 modèles, Pipeline + GridSearchCV, déploiement via une web app (Streamlit) et prédiction via saisie.  
> (Les bonus MLOps : DVC, Airflow, n8n sont intégrés en option.)

---

## Sommaire
- [Fonctionnalités](#fonctionnalités)
- [Pages de l'application](#pages-de-lapplication)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Utilisation](#utilisation)
- [Artefacts & sorties générées](#artefacts--sorties-générées)
- [Bonus MLOps (DVC / MLflow / Evidently / Airflow / n8n)](#bonus-mlops-dvc--mlflow--evidently--airflow--n8n)
- [Structure du projet](#structure-du-projet)
- [Conformité cahier des charges](#conformité-cahier-des-charges)
- [Partager sur GitHub](#partager-sur-github)
- [Crédits](#crédits)

---

## Fonctionnalités
- **Dataset‑agnostic** : charge un CSV et configure la cible **Y** + les variables **X** (auto ou manuel).
- **EDA & préparation** : contrôle qualité + prétraitements (missing values, encodage, standardisation…).
- **Modélisation** : sélection de **1 à 3 modèles** (classification ou régression), comparaison et choix du meilleur.
- **Tuning** : **Pipeline scikit‑learn + GridSearchCV** + Cross‑Validation.
- **Résultats** : métriques, tableaux, plots, export PDF/ZIP.
- **Prédiction** : saisie manuelle **ou** prédiction sur un fichier.
- **Historique** : sauvegarde de chaque run (config, métriques, artefacts, modèle).
- **Export & DB** (optionnel) : export livrables + envoi vers une base (si configurée).
- **Bonus MLOps** : DVC / MLflow / Evidently + orchestration Airflow + automatisation n8n (webhooks).

---

## Pages de l'application
L’application est organisée en pages Streamlit :

- **📘 Story** : contexte, objectifs, valeur + parcours d’usage.
- **📁 Data** : upload dataset, choix de la cible **Y**, sélection des **X**.
- **🧪 Train & Evaluate** : EDA/Prepare, choix des modèles (max 3), réglages (CV, GridSearch), lancement Train/Evaluate.
- **📊 Résultats** : dashboard métriques + plots, exports.
- **🔮 Predict** : prédiction sur nouvelles données (formulaire ou fichier).
- **🗃️ Export & DB** : export livrables + DB (si configurée).
- **🧾 Runs** : historique des exécutions et accès aux artefacts.
- **🧩 Bonus MLOps** : cockpit DVC/MLflow/Evidently.
- **🌀 Bonus Airflow** : orchestration DAG (si stack Airflow disponible).
- **🤖 Bonus n8n** : automatisation via webhook (si stack n8n disponible).

---

## Installation
### Prérequis
- **Python 3.10+** (recommandé : 3.11)
- Windows / Linux / macOS

### Installer les dépendances
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Démarrage rapide
```bash
streamlit run app.py
```

Puis ouvre l’URL indiquée dans le terminal (souvent `http://localhost:8501`).

---

## Utilisation
1) **📁 Data**  
   - Upload ton CSV  
   - Choisis la **cible Y**  
   - Définis les **X** (Auto ou Manuel)

2) **🧪 Train & Evaluate**  
   - (Optionnel) **EDA + Prepare**  
   - Sélectionne **1 à 3 modèles**  
   - Ajuste CV / GridSearch si besoin  
   - Lance **Train** puis **Evaluate**

3) **📊 Résultats**  
   - Consulte métriques/plots  
   - Exporte (PDF/ZIP si disponible)

4) **🔮 Predict**  
   - Fais une prédiction par **saisie** ou **fichier**

5) **🧾 Runs / 🗃️ Export & DB**  
   - Retrouve l’historique + export des artefacts  
   - (Optionnel) push DB si configurée

---

## Artefacts & sorties générées
Selon l’exécution, le projet génère typiquement :
- `models/` : meilleur modèle sauvegardé (ex: `best_model.pkl`)
- `reports/` : métriques et rapports (ex: `metrics.json`) + `plots/`
- `runs/` : historique d’exécutions (config + métriques + artefacts + modèle)

> 💡 Astuce : garde un petit exemple de sortie (1 run) pour la démo, mais évite de versionner de gros fichiers.

---

## Bonus MLOps (DVC / MLflow / Evidently / Airflow / n8n)
### DVC (reproductibilité)
Si DVC est configuré :
```bash
dvc repro
```

### MLflow (tracking d’expériences)
```bash
mlflow ui
```

### Evidently (qualité / drift / rapport)
Le rapport se trouve dans `reports/` (HTML/JSON selon config).

### Airflow (orchestration)
Si tu as une stack Airflow Docker, le DAG automatise :
Prepare → Train → Evaluate → (optionnel) déploiement.

### n8n (automatisation / webhook)
Exemple : un webhook n8n déclenche le DAG Airflow.
```bash
curl -X POST http://localhost:5678/webhook/sakiga/trigger-train -H "Content-Type: application/json" -d "{}"
```

---

## Structure du projet
> Peut légèrement varier selon ta version, mais l’idée est :

```
.
├── app.py
├── pages/
│   ├── 0_📘_Story.py
│   ├── 1_📁_Data.py
│   ├── 2_🧪_Train_&_Evaluate.py
│   ├── 3_📊_Résultats.py
│   ├── 4_🔮_Predict.py
│   ├── 5_🗃️_Export_&_DB.py
│   ├── 6_🧾_Runs.py
│   ├── 7_🧩_Bonus_MLOps.py
│   ├── 8_🌀_Bonus_Airflow.py
│   └── 9_🤖_Bonus_n8n.py
├── src/                # pipeline (prepare/train/evaluate), utils, etc.
├── data/               # raw/ processed/ config/
├── models/
├── reports/
├── runs/
├── requirements.txt
└── README.md
```

---

## Conformité cahier des charges
- [x] Problématique + cible (classification ou régression)
- [x] EDA + pré-traitement (missing, encoding, scaling…)
- [x] Split train/validation/test
- [x] **Comparaison d’au moins 3 modèles**
- [x] **Pipeline scikit‑learn + GridSearchCV**
- [x] Déploiement via **Streamlit** (saisie → prédiction)

---

## Crédits
Projet réalisé dans le cadre du module **Data Science (IGA)**.  
Encadrant : **M. Azmi Mohamed**.

