"""
utils/validators.py

Defines acceptable ranges for each input field, per disease, used both
for server-side validation (security: never trust client input) and to
generate the min/max attributes on the frontend form fields.
"""

FIELD_RANGES = {
    "heart": {
        "age": (1, 120),
        "sex": (0, 1),
        "cp": (0, 3),
        "trestbps": (50, 250),
        "chol": (50, 700),
        "fbs": (0, 1),
        "restecg": (0, 2),
        "thalach": (50, 250),
        "exang": (0, 1),
        "oldpeak": (0, 10),
        "slope": (0, 2),
        "ca": (0, 4),
        "thal": (0, 3),
    },
    "diabetes": {
        "Pregnancies": (0, 20),
        "Glucose": (0, 300),
        "BloodPressure": (0, 200),
        "SkinThickness": (0, 100),
        "Insulin": (0, 900),
        "BMI": (0, 80),
        "DiabetesPedigreeFunction": (0, 3),
        "Age": (1, 120),
    },
}


def validate_ranges(record: dict, disease: str) -> list:
    """
    Checks each field against its known plausible clinical range.
    Returns a list of human-readable warning strings (empty list = all OK).
    This does NOT raise — out-of-range values are flagged to the user as
    warnings rather than hard errors, since edge cases can be legitimate,
    but it protects against obviously bad/garbage input (e.g. age = -5
    or age = 9999) reaching the model silently.
    """
    ranges = FIELD_RANGES.get(disease, {})
    warnings = []
    for field, value in record.items():
        if field not in ranges:
            continue
        low, high = ranges[field]
        if value < low or value > high:
            warnings.append(
                f"{field} = {value} is outside the typical clinical range ({low}-{high})"
            )
    return warnings
