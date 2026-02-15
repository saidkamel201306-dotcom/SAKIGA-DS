# SAKIGA DS App (WAOUH) — Une seule application

Tu as raison : plusieurs scripts + commandes peuvent perdre l'équipe.

👉 Solution : **une seule application Streamlit** qui pilote tout :

- Chargement du CSV
- Préparation / entraînement / évaluation (boutons)
- Visualisation (metrics + plots)
- Recommandation (test rapide)
- Saisie manuelle + export
- (Optionnel) Push MySQL via un bouton

## Lancer l'app
```bash
cd sakiga_ds_deliverable
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes MySQL
- Renseigner `DB_URL` dans l'app (sidebar) ou en variable d'environnement.
Exemple:
`mysql+pymysql://user:password@localhost:3306/BDprojets`

## Lien cahier des charges / cours
- Projet DS complet (data → modèle → éval → déploiement)
- Déploiement web (Streamlit + API FastAPI)
- Bonus MLOps (CI/Docker) déjà dans le pack
- WAOUH démo (reco personnalisée)
