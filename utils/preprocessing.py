"""
utils/preprocessing.py

Shared data preprocessing pipeline for both supported diseases
(Heart Disease and Diabetes). Handles missing values, outliers,
feature scaling, and train/test splitting in one place so the
training script and the live prediction path use identical logic.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("preprocessing")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# --------------------------------------------------------------------------
# Disease configuration: feature order, target column, dataset path, and
# columns where 0 is not a real physiological value but a missing-data flag
# (this matters for the Pima diabetes dataset specifically).
# --------------------------------------------------------------------------
DISEASE_CONFIG = {
    "heart": {
        "dataset_path": os.path.join(DATASETS_DIR, "heart.csv"),
        "target_column": "target",
        "feature_order": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "zero_as_missing_columns": [],  # heart dataset has no such columns
        # IMPORTANT: verified empirically via feature/target correlation
        # (exang, oldpeak, ca all correlate NEGATIVELY with target=1, and
        # thalach correlates POSITIVELY with target=1 — all backwards from
        # medical expectation). In THIS dataset mirror, target=1 means
        # LOWER risk (healthier), target=0 means HIGHER risk. This flag
        # tells predict.py to invert the class-1 probability before
        # reporting it as "probability of disease".
        "positive_class_is_disease": False,
    },
    "diabetes": {
        "dataset_path": os.path.join(DATASETS_DIR, "diabetes.csv"),
        "target_column": "Outcome",
        "feature_order": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        # In the Pima dataset, 0 in these columns means "not recorded",
        # not a real clinical zero (you cannot have 0 blood pressure or 0 BMI).
        "zero_as_missing_columns": [
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
        ],
        # Verified correct: Outcome=1 correlates POSITIVELY with Glucose
        # (+0.47) and all other risk factors, matching medical expectation.
        "positive_class_is_disease": True,
    },
}


def get_config(disease: str) -> dict:
    if disease not in DISEASE_CONFIG:
        raise ValueError(f"Unknown disease '{disease}'. Must be one of {list(DISEASE_CONFIG.keys())}")
    return DISEASE_CONFIG[disease]


def load_data(disease: str) -> pd.DataFrame:
    config = get_config(disease)
    df = pd.read_csv(config["dataset_path"])
    logger.debug("Loaded %s dataset: %s", disease, df.shape)
    return df


def handle_missing_values(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    """
    1. Drop exact duplicate rows.
    2. Convert disease-specific 'zero means missing' columns to NaN.
    3. Impute remaining NaNs with the column median (robust to outliers,
       unlike the mean).
    """
    config = get_config(disease)
    df = df.drop_duplicates().copy()

    for col in config["zero_as_missing_columns"]:
        df[col] = df[col].replace(0, np.nan)

    n_missing_before = df.isnull().sum().sum()
    if n_missing_before > 0:
        logger.debug("Imputing %d missing values for %s using column medians", n_missing_before, disease)
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def handle_outliers(df: pd.DataFrame, disease: str, z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Clip extreme outliers using a z-score cap rather than dropping rows
    (dropping rows on a small medical dataset like this loses valuable,
    already-scarce positive cases). Any value beyond z_thresh standard
    deviations from the mean is clipped to that boundary.
    """
    config = get_config(disease)
    df = df.copy()
    for col in config["feature_order"]:
        mean, std = df[col].mean(), df[col].std()
        if std == 0:
            continue
        lower, upper = mean - z_thresh * std, mean + z_thresh * std
        df[col] = df[col].clip(lower, upper)
    return df


def clean_data(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    df = handle_missing_values(df, disease)
    df = handle_outliers(df, disease)
    return df


def split_and_scale(df: pd.DataFrame, disease: str, test_size: float = 0.2, random_state: int = 42):
    """Returns X_train, X_test, y_train, y_test, fitted scaler."""
    config = get_config(disease)
    X = df[config["feature_order"]]
    y = df[config["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_processed_data(disease: str):
    """Convenience wrapper used by the training script."""
    df = load_data(disease)
    df = clean_data(df, disease)
    return split_and_scale(df, disease)


def validate_record(record: dict, disease: str) -> dict:
    """
    Validates an incoming prediction record (e.g. from a Flask form).
    Returns a cleaned dict of floats in the correct feature order.
    Raises ValueError with a clear message on any problem.
    """
    config = get_config(disease)
    feature_order = config["feature_order"]

    missing = [f for f in feature_order if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    cleaned = {}
    for field in feature_order:
        raw = record[field]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{field}' must be a number, got: {raw!r}")
        if value < 0:
            raise ValueError(f"Field '{field}' cannot be negative, got: {value}")
        cleaned[field] = value

    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for disease in ["heart", "diabetes"]:
        df = load_data(disease)
        df = clean_data(df, disease)
        print(f"\n{disease.upper()} after cleaning: {df.shape}")
        print(df[get_config(disease)["target_column"]].value_counts())
