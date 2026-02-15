# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests

from src.text_utils import build_model_text
from src.reco.reco_service import load_index, recommend_from_viewed


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ✅ BONUS AIRFLOW: si un modèle "déployé" existe, on l'utilise.
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
DEPLOYED_MODEL_PATH = os.path.join(MODELS_DIR, "deployed_model.pkl")

RECO_INDEX_PATH = os.path.join(MODELS_DIR, "reco_item_index.pkl")
USER_API = os.getenv("USER_API", "http://localhost:8083")  # ms-user base URL

app = FastAPI(title="SAKIGA Gender Predictor", version="1.0")


class PredictIn(BaseModel):
    productDisplayName: str = ""
    subCategory: str = ""
    articleType: str = ""


class PredictOut(BaseModel):
    prediction: str
    confidence: float | None = None


class RecommendIn(BaseModel):
    userId: str = "demo"
    viewedItemIds: list[int] = []
    k: int = 10


class RecommendOut(BaseModel):
    userId: str
    recommendations: list[dict]


def fetch_recent_viewed_ids(user_id: str, limit: int = 12) -> list[int]:
    """Fetch viewed product IDs from ms-user recent history."""
    try:
        url = f"{USER_API}/api/users/{user_id}/history/recent-products"
        r = requests.get(url, params={"limit": int(limit)}, timeout=3)
        if r.status_code != 200:
            return []
        data = r.json()
        ids = []
        for it in data if isinstance(data, list) else []:
            # ms-user uses "id" as productId in response
            if isinstance(it, dict) and "id" in it:
                try:
                    ids.append(int(it["id"]))
                except Exception:
                    pass
        return ids
    except Exception:
        return []


def resolve_model_path() -> str:
    """Return deployed model path if it exists, else best model path."""
    return DEPLOYED_MODEL_PATH if os.path.exists(DEPLOYED_MODEL_PATH) else BEST_MODEL_PATH


# ✅ Auto-reload: si Airflow déploie un nouveau modèle pendant que l'API tourne,
# l'API le recharge automatiquement au prochain /predict.
_model = None
_model_loaded_path = None

def get_model():
    global _model, _model_loaded_path
    current_path = resolve_model_path()

    if not os.path.exists(current_path):
        raise FileNotFoundError(
            f"Model not found: {current_path}. "
            "Run Train & Evaluate (ou Airflow) d'abord."
        )

    if _model is None or _model_loaded_path != current_path:
        _model = joblib.load(current_path)
        _model_loaded_path = current_path

    return _model, _model_loaded_path


reco_index = None


@app.on_event("startup")
def _startup():
    global reco_index
    # charge le modèle une première fois
    get_model()

    # Reco index is optional (build it with: python -m src.reco.build_item_index)
    try:
        reco_index = load_index(RECO_INDEX_PATH) if os.path.exists(RECO_INDEX_PATH) else None
    except Exception:
        reco_index = None


@app.get("/health")
def health():
    try:
        _, p = get_model()
        ok = True
    except Exception:
        ok = False
        p = None
    return {"status": "ok", "model_loaded": ok, "model_path": p}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    model, _path = get_model()

    text = build_model_text(payload.productDisplayName, payload.subCategory, payload.articleType)
    pred = model.predict([text])[0]
    conf = None

    # If classifier supports predict_proba, return max probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        conf = float(max(proba))

    return PredictOut(prediction=str(pred), confidence=conf)


@app.post("/recommend", response_model=RecommendOut)
def recommend(payload: RecommendIn):
    """
    Recommend products based on browsing history.

    ✅ Two modes (compatible with your existing app):
    1) If `viewedItemIds` is provided -> use it directly.
    2) If empty -> fetch real history from ms-user:
       GET {USER_API}/api/users/{userId}/history/recent-products?limit=...
    """
    if reco_index is None:
        return RecommendOut(userId=payload.userId, recommendations=[])

    viewed = list(payload.viewedItemIds or [])
    if not viewed:
        viewed = fetch_recent_viewed_ids(payload.userId, limit=max(12, int(payload.k) * 2))

    recs = recommend_from_viewed(reco_index, viewed, k=int(payload.k))
    return RecommendOut(userId=payload.userId, recommendations=recs)
