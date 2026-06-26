"""
utils/explainability.py

Provides explainable-AI features for predictions:
- Global feature importance (from the trained model)
- Per-prediction "top contributing factors" (how far each of this
  patient's values is from the healthy-population average, weighted
  by that feature's global importance)
- A plain-language medical-style explanation string

This intentionally does NOT use SHAP/LIME (heavy dependencies, slow
on a small Flask demo app). Instead it uses two well-understood,
fast, fully-transparent techniques:
  1. Model-native feature importance (RandomForest.feature_importances_,
     or absolute LogisticRegression coefficients as a fallback)
  2. Z-score deviation of the patient's values from the training
     population mean, to say *why this particular patient* triggered
     the prediction.
"""

import numpy as np


FRIENDLY_NAMES = {
    # Heart disease
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol Level",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG Result",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Exercise)",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels Colored",
    "thal": "Thalassemia Result",
    # Diabetes
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Glucose Level",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin Level",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree (Family History) Score",
    "Age": "Age",
}


def get_global_feature_importance(model, feature_order):
    """
    Returns a dict {feature_name: importance_score} normalized to sum to 1.
    Works with tree-based models (feature_importances_) and linear models
    (coef_). Falls back to uniform importance if neither is available.
    """
    if hasattr(model, "feature_importances_"):
        raw = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        raw = np.ones(len(feature_order), dtype=float)

    total = raw.sum()
    if total == 0:
        raw = np.ones(len(feature_order), dtype=float)
        total = raw.sum()

    normalized = raw / total
    return {feat: float(score) for feat, score in zip(feature_order, normalized)}


def get_top_contributing_factors(record: dict, feature_order: list, scaler, importance_dict: dict, top_n: int = 4):
    """
    Combines global feature importance with how unusual THIS patient's
    value is (z-score vs the training population, available via the
    fitted scaler's mean_/scale_) to produce a per-patient ranked list
    of contributing factors.

    Returns a list of dicts: [{feature, friendly_name, contribution_score,
    patient_value, population_mean, direction}, ...] sorted by contribution_score desc.
    """
    contributions = []
    for i, feat in enumerate(feature_order):
        patient_value = record[feat]
        pop_mean = scaler.mean_[i]
        pop_std = scaler.scale_[i] if scaler.scale_[i] != 0 else 1.0

        z = (patient_value - pop_mean) / pop_std
        weight = importance_dict.get(feat, 0.0)
        contribution_score = abs(z) * weight

        contributions.append({
            "feature": feat,
            "friendly_name": FRIENDLY_NAMES.get(feat, feat),
            "contribution_score": float(contribution_score),
            "patient_value": float(patient_value),
            "population_mean": round(float(pop_mean), 2),
            "direction": "above average" if z > 0 else "below average",
        })

    contributions.sort(key=lambda c: c["contribution_score"], reverse=True)
    return contributions[:top_n]


def generate_explanation(disease: str, risk_label: str, top_factors: list) -> str:
    """Builds a short, plain-language explanation string for the result page."""
    disease_name = "heart disease" if disease == "heart" else "diabetes"

    if not top_factors:
        return f"The model assessed this patient's overall profile as {risk_label.lower()} for {disease_name}."

    factor_phrases = [
        f"{f['friendly_name']} ({f['direction']} the typical patient profile)"
        for f in top_factors
    ]

    if len(factor_phrases) == 1:
        factors_text = factor_phrases[0]
    else:
        factors_text = ", ".join(factor_phrases[:-1]) + ", and " + factor_phrases[-1]

    return (
        f"This patient was assessed as **{risk_label}** for {disease_name}. "
        f"The factors that most influenced this prediction were: {factors_text}. "
        f"This explanation is generated from the model's learned feature importance "
        f"combined with how far this patient's values deviate from the training "
        f"population average — it is informational only and not a medical diagnosis."
    )
