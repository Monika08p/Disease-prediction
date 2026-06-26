"""
app.py

Main Flask application for the Disease Prediction System.
Supports Heart Disease and Diabetes prediction with explainable AI,
risk scoring, PDF report generation, and prediction history.
Authentication added via Flask Blueprint (auth.py).
"""

import os
import json
import logging
import secrets
from datetime import datetime

# FIX: load_dotenv() MUST be called before any project module is imported.
# Previously it was called AFTER 'from ai_assistant import ...' which, while
# technically safe (because _get_api_key() is lazy), is fragile and wrong
# practice. Any module that reads os.environ at import time would miss the
# .env values. Moving it here guarantees the env is populated first.
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from the project root into os.environ
except ImportError:
    pass  # python-dotenv not installed — rely on real environment variables

from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for, session

from utils.preprocessing import get_config, validate_record
from utils.predict import predict_one
from utils.explainability import (
    get_global_feature_importance, get_top_contributing_factors,
    generate_explanation, FRIENDLY_NAMES
)
from utils.validators import validate_ranges, FIELD_RANGES
from utils.pdf_report import generate_pdf_report
from auth import auth_bp, init_db, login_required
from ai_assistant import chat as ai_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "reports", "prediction_history.json")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))

# Register auth blueprint
app.register_blueprint(auth_bp)

# Initialise the users database on startup
with app.app_context():
    init_db()


# --------------------------------------------------------------------------
# History storage
# --------------------------------------------------------------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE) as f:
        return json.load(f)


def save_history_entry(entry: dict):
    history = load_history()
    history.insert(0, entry)
    history = history[:100]
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------------------
# Page routes — all protected with @login_required
# --------------------------------------------------------------------------
@app.route("/")
@login_required
def landing():
    return render_template("landing.html")


@app.route("/select")
@login_required
def select_disease():
    return render_template("select.html")


@app.route("/predict/<disease>", methods=["GET", "POST"])
@login_required
def predict_page(disease):
    if disease not in ("heart", "diabetes"):
        abort(404)

    config = get_config(disease)
    feature_order = config["feature_order"]
    error = None
    warnings = []

    if request.method == "POST":
        logger.debug("[%s] Raw form data: %s", disease, dict(request.form))
        try:
            patient_name = (request.form.get("patient_name") or "").strip()
            if not patient_name:
                raise ValueError("Patient name is required")

            raw_record = {field: request.form.get(field) for field in feature_order}
            for field, value in raw_record.items():
                if value is None or value.strip() == "":
                    raise ValueError(f"Field '{field}' was not submitted or is empty")

            prediction = predict_one(raw_record, disease)
            clean_record = prediction["clean_record"]

            warnings = validate_ranges(clean_record, disease)

            model, scaler, metadata = _load_model_bundle(disease)
            importance = metadata["feature_importance"]
            top_factors = get_top_contributing_factors(clean_record, feature_order, scaler, importance)
            explanation = generate_explanation(disease, prediction["risk_label"], top_factors)

            timestamp = datetime.now().isoformat()

            result = {
                "patient_name": patient_name,
                "disease": disease,
                "disease_label": "Heart Disease" if disease == "heart" else "Diabetes",
                "label": prediction["label"],
                "proba": round(prediction["probability"] * 100, 2),
                "risk_label": prediction["risk_label"],
                "verdict": prediction["verdict"],
                "top_factors": top_factors,
                "explanation": explanation,
                "record": clean_record,
                "model_used": metadata["best_model_name"],
                "timestamp": timestamp,
            }

            save_history_entry({
                "timestamp": timestamp,
                "patient_name": patient_name,
                "disease": disease,
                "record": clean_record,
                "proba": result["proba"],
                "risk_label": result["risk_label"],
                "verdict": result["verdict"],
            })

            logger.debug("[%s] Result: %s", disease, result)

            session[f"result_{disease}"] = result
            return redirect(url_for("result_page", disease=disease))

        except Exception as e:
            logger.exception("Prediction failed for %s", disease)
            error = str(e)

    return render_template(
        "predict.html",
        disease=disease,
        disease_label="Heart Disease" if disease == "heart" else "Diabetes",
        fields=feature_order,
        labels=FRIENDLY_NAMES,
        ranges=FIELD_RANGES.get(disease, {}),
        error=error,
        warnings=warnings,
        form_values=request.form,
    )


@app.route("/result/<disease>")
@login_required
def result_page(disease):
    if disease not in ("heart", "diabetes"):
        abort(404)

    result = session.get(f"result_{disease}")
    if not result:
        return redirect(url_for("predict_page", disease=disease))

    feature_order = get_config(disease)["feature_order"]

    return render_template(
        "result.html",
        disease=disease,
        feature_order=feature_order,
        labels=FRIENDLY_NAMES,
        result=result,
    )


@app.route("/history")
@login_required
def history_page():
    history = load_history()
    return render_template("history.html", history=history)


# --------------------------------------------------------------------------
# API endpoints (JSON)
# --------------------------------------------------------------------------
@app.route("/api/predict/<disease>", methods=["POST"])
@login_required
def api_predict(disease):
    if disease not in ("heart", "diabetes"):
        return jsonify({"error": "Unknown disease. Use 'heart' or 'diabetes'."}), 404

    try:
        payload = request.get_json(force=True)
        prediction = predict_one(payload, disease)
        return jsonify({
            "disease": disease,
            "label": prediction["label"],
            "probability": round(prediction["probability"], 4),
            "risk_label": prediction["risk_label"],
            "verdict": prediction["verdict"],
            "model_used": _load_model_bundle(disease)[2]["best_model_name"],
        })
    except Exception as e:
        logger.exception("API prediction failed for %s", disease)
        return jsonify({"error": str(e)}), 400


@app.route("/api/history")
@login_required
def api_history():
    return jsonify(load_history())


# --------------------------------------------------------------------------
# PDF report download
# --------------------------------------------------------------------------
@app.route("/download_report/<disease>", methods=["POST"])
@login_required
def download_report(disease):
    if disease not in ("heart", "diabetes"):
        abort(404)

    config = get_config(disease)
    feature_order = config["feature_order"]

    patient_name = (request.form.get("patient_name") or "Unknown Patient").strip()
    raw_record = {field: request.form.get(field) for field in feature_order}
    prediction = predict_one(raw_record, disease)
    clean_record = prediction["clean_record"]

    model, scaler, metadata = _load_model_bundle(disease)
    importance = metadata["feature_importance"]
    top_factors = get_top_contributing_factors(clean_record, feature_order, scaler, importance)
    explanation = generate_explanation(disease, prediction["risk_label"], top_factors)

    result = {
        "verdict": prediction["verdict"],
        "proba": round(prediction["probability"] * 100, 2),
        "risk_label": prediction["risk_label"],
    }

    filepath = generate_pdf_report(disease, patient_name, clean_record, FRIENDLY_NAMES, result, top_factors, explanation)
    return send_file(filepath, as_attachment=True, download_name=os.path.basename(filepath))


# --------------------------------------------------------------------------
# AI Health Assistant API
# --------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """
    Accepts JSON: { "message": str, "history": [...], "prediction_context": {...}|null }
    Returns JSON: { "reply": str }
    Gemini is ONLY used for health explanations — never for disease prediction.
    """
    try:
        payload = request.get_json(force=True) or {}
        user_message = (payload.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400

        history = payload.get("history") or []
        prediction_context = payload.get("prediction_context") or None

        reply = ai_chat(user_message, history, prediction_context)
        return jsonify({"reply": reply})
    except ValueError as e:
        logger.error("AI chat config error: %s", e)
        return jsonify({"reply": "The AI Assistant is not configured. Please set GEMINI_API_KEY in your .env file."}), 200
    except Exception as e:
        logger.exception("AI chat error")
        return jsonify({"reply": "An unexpected error occurred. Please try again."}), 200


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def _load_model_bundle(disease):
    from utils.predict import load_artifacts
    return load_artifacts(disease)


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error")
    return render_template("error.html", code=500, message="Internal server error"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
