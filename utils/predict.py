"""
utils/predict.py

Loads the saved best model + the saved, ALREADY-FITTED scaler for a given
disease, and runs predictions.

IMPORTANT: Always uses scaler.transform() with the scaler that was fit
during training. NEVER creates/fits a new scaler on the incoming row —
doing so collapses every feature to 0 regardless of input (mean == the
row's own value, std == 0), which makes every prediction identical. This
was a real bug encountered in an earlier iteration of this project and is
explicitly guarded against here.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .preprocessing import get_config, validate_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Cache loaded artifacts per disease so they're loaded once, not per-request.
_CACHE = {}


def load_artifacts(disease: str, force_reload: bool = False):
    if disease in _CACHE and not force_reload:
        return _CACHE[disease]

    disease_dir = os.path.join(MODELS_DIR, disease)
    metadata_path = os.path.join(disease_dir, "metadata.json")
    scaler_path = os.path.join(disease_dir, "scaler.pkl")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No trained model found for '{disease}'. Run: python train_model.py --disease {disease}"
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    scaler = joblib.load(scaler_path)

    if metadata["model_type"] == "tensorflow":
        model = tf.keras.models.load_model(os.path.join(disease_dir, "best_model.keras"))
    else:
        model = joblib.load(os.path.join(disease_dir, "best_model.pkl"))

    logger.debug("Loaded %s model (%s) | scaler mean_[:3]=%s", disease, metadata["best_model_name"], scaler.mean_[:3])

    _CACHE[disease] = (model, scaler, metadata)
    return model, scaler, metadata


def get_risk_label(probability: float) -> str:
    if probability < 0.33:
        return "Low Risk"
    elif probability < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"


def _predict_proba(model, metadata, x_scaled, disease):
    """
    Returns P(disease), NOT necessarily P(model's class 1).

    ROOT CAUSE FIX: the heart.csv dataset's `target` column is labeled
    1 for LOWER risk and 0 for HIGHER risk (verified via correlation
    analysis — exang/oldpeak/ca correlate negatively with target=1,
    thalach correlates positively, all backwards from medical
    expectation). The model was trained correctly against that column,
    so model.predict_proba()[:, 1] is genuinely "P(class 1)" — but class
    1 in this dataset means "healthy", not "disease". Reporting that
    raw value as "probability of disease" inverted every result, which
    is why healthy inputs were scored as high risk.

    config["positive_class_is_disease"] (set per-disease in
    preprocessing.DISEASE_CONFIG) tells us whether class 1 = disease
    (diabetes: yes) or class 1 = healthy (heart: no, in this dataset).
    """
    if metadata["model_type"] == "tensorflow":
        proba_class1 = float(model.predict(x_scaled, verbose=0).ravel()[0])
    else:
        proba_class1 = float(model.predict_proba(x_scaled)[0][1])

    config = get_config(disease)
    if config["positive_class_is_disease"]:
        return proba_class1
    else:
        return 1.0 - proba_class1


def predict_one(record: dict, disease: str):
    """
    record: dict of raw (unscaled) field values, e.g. {"age": 63, "sex": 1, ...}
    disease: "heart" or "diabetes"

    Returns dict: {label, probability, risk_label, verdict}
    """
    model, scaler, metadata = load_artifacts(disease)
    config = get_config(disease)
    feature_order = config["feature_order"]

    # Validate + coerce types (raises ValueError with a clear message on bad input)
    clean_record = validate_record(record, disease)

    logger.debug("[%s] Raw record: %s", disease, record)
    logger.debug("[%s] Validated record: %s", disease, clean_record)

    ordered_values = [clean_record[f] for f in feature_order]
    x = pd.DataFrame([ordered_values], columns=feature_order)

    # *** Always transform with the FITTED scaler, never fit_transform ***
    x_scaled = scaler.transform(x)
    logger.debug("[%s] Scaled vector: %s", disease, x_scaled.tolist())

    proba = _predict_proba(model, metadata, x_scaled, disease)
    label = int(proba >= 0.5)
    risk_label = get_risk_label(proba)

    disease_name = "heart disease" if disease == "heart" else "diabetes"
    verdict = f"Likely {disease_name}" if label == 1 else f"Unlikely {disease_name}"

    logger.debug("[%s] label=%s proba=%.4f risk=%s", disease, label, proba, risk_label)

    return {
        "label": label,
        "probability": proba,
        "risk_label": risk_label,
        "verdict": verdict,
        "clean_record": clean_record,
    }
