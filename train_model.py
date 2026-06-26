"""
train_model.py

Trains and compares 4 models (Logistic Regression, Random Forest,
XGBoost, and a TensorFlow Neural Network) for BOTH supported diseases
(heart, diabetes), evaluates each on Accuracy, Precision, Recall, F1,
and ROC-AUC, and automatically saves the best-performing model per
disease along with its scaler and feature importances.

Usage:
    python train_model.py --disease heart
    python train_model.py --disease diabetes
    python train_model.py --disease all
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import get_processed_data, get_config
from utils.explainability import get_global_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_model")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


def evaluate(name, y_true, preds, probs):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def build_nn(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_disease(disease: str):
    logger.info("=" * 60)
    logger.info("Training models for: %s", disease.upper())
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test, scaler = get_processed_data(disease)
    feature_order = get_config(disease)["feature_order"]

    results = []
    trained_models = {}

    # --- Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    probs = logreg.predict_proba(X_test)[:, 1]
    preds = logreg.predict(X_test)
    results.append(evaluate("logistic_regression", y_test, preds, probs))
    trained_models["logistic_regression"] = logreg

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    preds = rf.predict(X_test)
    results.append(evaluate("random_forest", y_test, preds, probs))
    trained_models["random_forest"] = rf

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42
    )
    xgb_model.fit(X_train, y_train)
    probs = xgb_model.predict_proba(X_test)[:, 1]
    preds = xgb_model.predict(X_test)
    results.append(evaluate("xgboost", y_test, preds, probs))
    trained_models["xgboost"] = xgb_model

    # --- TensorFlow Neural Network ---
    nn = build_nn(X_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    nn.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=16,
           callbacks=[early_stop], verbose=0)
    probs = nn.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    results.append(evaluate("tensorflow_nn", y_test, preds, probs))
    trained_models["tensorflow_nn"] = nn

    # --- Compare and select best model (by ROC-AUC, tie-break on F1) ---
    results_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "f1_score"], ascending=False
    ).reset_index(drop=True)

    logger.info("\nModel comparison for %s:\n%s", disease, results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    logger.info("Best model for %s: %s", disease, best_name)

    # --- Save artifacts ---
    disease_dir = os.path.join(MODELS_DIR, disease)
    os.makedirs(disease_dir, exist_ok=True)

    if best_name == "tensorflow_nn":
        best_model.save(os.path.join(disease_dir, "best_model.keras"))
        model_type = "tensorflow"
    else:
        joblib.dump(best_model, os.path.join(disease_dir, "best_model.pkl"))
        model_type = "sklearn"

    joblib.dump(scaler, os.path.join(disease_dir, "scaler.pkl"))
    results_df.to_csv(os.path.join(disease_dir, "model_comparison.csv"), index=False)

    # Feature importance: use RF if the best model has no native importance
    # (e.g. NN), so the explainability module always has something to show.
    importance_source = best_model if best_name in ("random_forest", "xgboost", "logistic_regression") else trained_models["random_forest"]
    importance = get_global_feature_importance(importance_source, feature_order)

    metadata = {
        "disease": disease,
        "best_model_name": best_name,
        "model_type": model_type,
        "feature_order": feature_order,
        "feature_importance": importance,
        "metrics": results_df.iloc[0].to_dict(),
    }
    with open(os.path.join(disease_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved all artifacts to %s", disease_dir)
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=["heart", "diabetes", "all"], default="all")
    args = parser.parse_args()

    diseases = ["heart", "diabetes"] if args.disease == "all" else [args.disease]
    for disease in diseases:
        train_disease(disease)


if __name__ == "__main__":
    main()
