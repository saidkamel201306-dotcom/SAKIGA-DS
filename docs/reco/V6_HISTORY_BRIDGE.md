# v6 — Reco branchée sur l'historique réel (ms-user)

## Objectif
Avoir une recommandation personnalisée **automatique** : on fournit seulement `userId`,
et le service reco récupère les derniers produits consultés depuis **ms-user**.

## Flux (à filmer)
1) L'utilisateur consulte 3 produits dans l'app.
2) Le front envoie 3 appels `POST /history/view` (ms-user).
3) Quand la home recharge, le front appelle `/api/reco/recommend` avec `userId`.
4) Le reco-service appelle ms-user `recent-products`, calcule, renvoie une liste.
5) La section "Pour vous" change.

## Endpoints
- ms-user (déjà dans votre app)
  - `POST /api/users/{userId}/history/view`
  - `GET /api/users/{userId}/history/recent-products?limit=12`

- reco (dans ce pack)
  - `POST /recommend`
    - mode 1: avec `viewedItemIds`
    - mode 2: sans `viewedItemIds` -> fetch auto depuis ms-user

## Configuration
Par défaut, le pack suppose :
- ms-user: http://localhost:8083

Si différent :
- variable d'environnement `USER_API`

## Lien cahier des charges
- Déploiement web + démo fonctionnelle
- Données réelles (historique)
- Modèle explicable (content-based similarity)
