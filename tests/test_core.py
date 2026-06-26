"""
tests/test_core.py

Test suite covering:
- Preprocessing (missing values, outliers, scaling)
- Input validation
- Prediction (the critical "different inputs -> different outputs" guarantee)
- Risk label thresholds
- Explainability output shape

Run with: pytest tests/ -v
"""

import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessing import (
    load_data, clean_data, get_config, validate_record, DISEASE_CONFIG
)
from utils.predict import predict_one, get_risk_label
from utils.explainability import get_top_contributing_factors, generate_explanation
from utils.validators import validate_ranges


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
class TestPreprocessing:

    def test_heart_data_loads(self):
        df = load_data("heart")
        assert df.shape[0] > 0
        assert "target" in df.columns

    def test_diabetes_data_loads(self):
        df = load_data("diabetes")
        assert df.shape[0] > 0
        assert "Outcome" in df.columns

    def test_diabetes_zero_imputation(self):
        """Zeros in Glucose/BMI/etc should be treated as missing and imputed,
        never left as literal zero (which is clinically impossible)."""
        df = load_data("diabetes")
        cleaned = clean_data(df, "diabetes")
        for col in ["Glucose", "BMI", "BloodPressure"]:
            assert (cleaned[col] == 0).sum() == 0

    def test_no_nulls_after_cleaning(self):
        for disease in ["heart", "diabetes"]:
            df = load_data(disease)
            cleaned = clean_data(df, disease)
            assert cleaned.isnull().sum().sum() == 0

    def test_unknown_disease_raises(self):
        with pytest.raises(ValueError):
            get_config("unknown_disease")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:

    def test_valid_heart_record_passes(self):
        record = {
            "age": 60, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
            "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
            "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
        }
        cleaned = validate_record(record, "heart")
        assert cleaned["age"] == 60.0

    def test_missing_field_raises(self):
        record = {"age": 60}  # missing everything else
        with pytest.raises(ValueError):
            validate_record(record, "heart")

    def test_non_numeric_field_raises(self):
        record = {
            "age": "not_a_number", "sex": 1, "cp": 2, "trestbps": 140,
            "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
            "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
        }
        with pytest.raises(ValueError):
            validate_record(record, "heart")

    def test_negative_value_raises(self):
        record = {
            "age": -5, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
            "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
            "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
        }
        with pytest.raises(ValueError):
            validate_record(record, "heart")

    def test_out_of_range_gives_warning_not_error(self):
        record = {
            "age": 200, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
            "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
            "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
        }
        warnings = validate_ranges(record, "heart")
        assert len(warnings) > 0
        assert "age" in warnings[0]


# ---------------------------------------------------------------------------
# Prediction tests — the critical regression tests for the scaler bug
# ---------------------------------------------------------------------------
class TestPrediction:

    def test_different_heart_inputs_give_different_outputs(self):
        """Regression test: two very different patients must NOT produce
        the same probability. This catches the fit_transform-on-single-row
        bug that previously caused identical predictions for all inputs."""
        patient_a = {
            "age": 70, "sex": 1, "cp": 3, "trestbps": 180, "chol": 300,
            "fbs": 1, "restecg": 2, "thalach": 100, "exang": 1,
            "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 3
        }
        patient_b = {
            "age": 25, "sex": 0, "cp": 0, "trestbps": 110, "chol": 150,
            "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
            "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
        }
        result_a = predict_one(patient_a, "heart")
        result_b = predict_one(patient_b, "heart")
        assert result_a["probability"] != result_b["probability"]

    def test_different_diabetes_inputs_give_different_outputs(self):
        patient_a = {
            "Pregnancies": 8, "Glucose": 190, "BloodPressure": 90,
            "SkinThickness": 40, "Insulin": 200, "BMI": 38,
            "DiabetesPedigreeFunction": 1.2, "Age": 55
        }
        patient_b = {
            "Pregnancies": 0, "Glucose": 85, "BloodPressure": 65,
            "SkinThickness": 20, "Insulin": 70, "BMI": 21,
            "DiabetesPedigreeFunction": 0.2, "Age": 22
        }
        result_a = predict_one(patient_a, "diabetes")
        result_b = predict_one(patient_b, "diabetes")
        assert result_a["probability"] != result_b["probability"]
        # This specific pair should be unambiguous: clear high vs low risk
        assert result_a["probability"] > result_b["probability"]

    def test_probability_is_valid_range(self):
        patient = {
            "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 220,
            "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }
        result = predict_one(patient, "heart")
        assert 0.0 <= result["probability"] <= 1.0

    def test_invalid_disease_raises(self):
        with pytest.raises(Exception):
            predict_one({}, "cancer")


# ---------------------------------------------------------------------------
# Risk label tests
# ---------------------------------------------------------------------------
class TestRiskLabels:

    def test_low_risk_threshold(self):
        assert get_risk_label(0.10) == "Low Risk"
        assert get_risk_label(0.32) == "Low Risk"

    def test_medium_risk_threshold(self):
        assert get_risk_label(0.33) == "Medium Risk"
        assert get_risk_label(0.65) == "Medium Risk"

    def test_high_risk_threshold(self):
        assert get_risk_label(0.66) == "High Risk"
        assert get_risk_label(0.99) == "High Risk"


# ---------------------------------------------------------------------------
# Explainability tests
# ---------------------------------------------------------------------------
class TestExplainability:

    def test_explanation_mentions_risk_label(self):
        explanation = generate_explanation("heart", "High Risk", [])
        assert "high risk" in explanation.lower()

    def test_top_factors_returns_requested_count(self):
        import joblib
        from utils.predict import load_artifacts
        model, scaler, metadata = load_artifacts("heart")
        feature_order = get_config("heart")["feature_order"]
        record = {f: 1.0 for f in feature_order}
        factors = get_top_contributing_factors(
            record, feature_order, scaler, metadata["feature_importance"], top_n=3
        )
        assert len(factors) == 3
        assert all("friendly_name" in f for f in factors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
