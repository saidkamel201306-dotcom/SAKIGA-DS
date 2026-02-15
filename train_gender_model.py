# -*- coding: utf-8 -*-
"""
Train Gender Classifier (MEN vs WOMEN) from styles.csv, with:
- EDA-friendly prints
- Train / Validation / Test split
- At least 3 algorithms
- Pipeline + GridSearchCV
- Leakage mitigation (remove explicit gender tokens from text)

Outputs:
- models/best_model.pkl
- reports/metrics.json
"""
from __future__ import annotations

import os, json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import joblib

from src.text_utils import build_model_text

DATA_PATH = os.path.join(os.path.dirname(__file__), "styles.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def normalize_gender(g):
    if g in ["Men", "Boys"]:
        return "MEN"
    if g in ["Women", "Girls"]:
        return "WOMEN"
    return None

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["gender_group"] = df["gender"].apply(normalize_gender)
    df = df.dropna(subset=["gender_group"])

    # fill NA for used cols
    for c in ["productDisplayName", "subCategory", "articleType"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Build text with leakage mitigation
    df["text"] = df.apply(lambda r: build_model_text(r["productDisplayName"], r["subCategory"], r["articleType"]), axis=1)

    return df

def evaluate(model, X, y, label: str) -> dict:
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="macro"))
    rep = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=["MEN", "WOMEN"]).tolist()
    return {"label": label, "accuracy": acc, "f1_macro": f1, "report": rep, "confusion_matrix": cm}

def main():
    df = load_data(DATA_PATH)

    # Basic EDA prints
    print("Rows:", len(df))
    print("Target distribution:\n", df["gender_group"].value_counts())

    X = df["text"].astype(str)
    y = df["gender_group"].astype(str)

    # Train/Val/Test: 70/15/15 (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    # 3 algorithms (each inside its own pipeline)
    pipelines = {
        "logreg": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "linearsvc": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LinearSVC())
        ]),
        "mnb": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB())
        ])
    }

    # GridSearch spaces (compact but meaningful)
    param_grids = {
        "logreg": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [5000, 8000, 12000],
            "clf__C": [0.5, 1.0, 2.0],
        },
        "linearsvc": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [5000, 8000, 12000],
            "clf__C": [0.5, 1.0, 2.0],
        },
        "mnb": {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__max_features": [5000, 8000, 12000],
            "clf__alpha": [0.3, 1.0, 2.0],
        }
    }

    # Choose a scoring that is robust for 2-class with imbalance: macro-F1
    best_overall = None
    best_name = None
    best_score = -1.0
    results = {}

    for name, pipe in pipelines.items():
        print("\n==============================")
        print("GridSearch for:", name)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            scoring="f1_macro",
            n_jobs=-1,
            cv=5,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        # Evaluate on validation set
        val_pred = grid.best_estimator_.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred))
        results[name] = {
            "best_params": grid.best_params_,
            "cv_best_score_f1_macro": float(grid.best_score_),
            "val_f1_macro": val_f1,
            "val_accuracy": val_acc,
        }
        print("Best params:", grid.best_params_)
        print("CV best f1_macro:", grid.best_score_)
        print("Val f1_macro:", val_f1, " | Val acc:", val_acc)

        if val_f1 > best_score:
            best_score = val_f1
            best_overall = grid.best_estimator_
            best_name = name

    assert best_overall is not None

    # Final evaluation on TEST set (never used in tuning)
    metrics = {
        "chosen_model": best_name,
        "selection_metric": "val_f1_macro",
        "val_f1_macro": best_score,
        "models_summary": results,
        "validation": evaluate(best_overall, X_val, y_val, "validation"),
        "test": evaluate(best_overall, X_test, y_test, "test"),
    }

    model_path = os.path.join(OUT_DIR, "best_model.pkl")
    joblib.dump(best_overall, model_path)

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved best model to:", model_path)
    print("✅ Saved metrics to:", os.path.join(REPORT_DIR, "metrics.json"))
    print("\nTest accuracy:", metrics["test"]["accuracy"])
    print("Test macro-F1:", metrics["test"]["f1_macro"])

if __name__ == "__main__":
    main()
