"""
ai_assistant.py

AI Health Assistant module powered by Google Gemini REST API.
This module is completely independent from the prediction logic.
Gemini is ONLY used to explain results and answer health questions.
It never predicts, diagnoses, or overrides ML predictions.

BUGS FIXED (vs original):
  1. CRITICAL: 'system_instruction' renamed to 'systemInstruction'
       The Gemini REST API v1beta uses camelCase for all top-level fields.
       snake_case 'system_instruction' was silently ignored, so Gemini had
       no system prompt at all, causing 400 errors or uncontrolled responses.

  2. MINOR: Removed unused 'import re' (was imported but never used).

  3. IMPROVED: HTTP error handler now logs the full Gemini error body so the
       real reason (400 bad request, 401 invalid key, etc.) appears in the
       Flask log instead of being swallowed by a generic message.
"""

import os
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger("ai_assistant")

# ── Gemini REST endpoint ───────────────────────────────────────────────────
# Using v1beta which supports systemInstruction and gemini-2.0-flash.
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """You are a professional Health AI Assistant embedded in a Disease Prediction web application.

YOUR ROLE:
- Explain disease prediction results in simple, clear language
- Answer healthcare-related questions about heart disease, diabetes, and general health
- Explain medical terms (cholesterol, blood pressure, glucose, BMI, ECG, etc.)
- Provide healthy lifestyle suggestions and general wellness guidance
- Help users understand their generated medical reports

STRICT BOUNDARIES — YOU MUST NEVER:
- Predict whether someone has a disease (the ML model does that, not you)
- Override, question, or modify the ML model's prediction result
- Diagnose any medical condition
- Prescribe medicines or recommend dosages
- Claim a user definitely has or does not have any disease
- Answer questions about programming, coding, entertainment, politics, sports, or any topic unrelated to health

OFF-TOPIC RESPONSE:
If the user asks anything unrelated to health, medicine, or their prediction results, respond with exactly:
"I am the Health AI Assistant. I can answer questions related to heart disease, diabetes, health parameters, healthy lifestyle, and your prediction results."

TONE & STYLE:
- Be warm, professional, and empathetic
- Use plain language; avoid overwhelming jargon
- Keep responses concise (3–5 sentences for simple questions, slightly longer for complex ones)
- Always recommend consulting a qualified healthcare professional for personal medical decisions
- Never cause unnecessary alarm or false reassurance

DISCLAIMER AWARENESS:
You operate under the disclaimer that you provide educational information only, not medical advice.
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Read the Gemini API key from the environment.

    load_dotenv() in app.py populates os.environ from the .env file before
    any request is handled, so this function can safely read it at call time.
    Raises ValueError (caught in /api/chat) when the key is missing so the
    user gets an informative message instead of a traceback.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Create a .env file from .env.example and add your key."
        )
    return key


def build_context_prompt(prediction_context: dict | None) -> str:
    """Build a context string so Gemini can explain the ML result.

    The context is prepended to the user's first message.
    Gemini is explicitly told NOT to change the prediction — only explain it.
    """
    if not prediction_context:
        return ""
    ctx_parts = ["PREDICTION CONTEXT (provided by the ML model — do NOT change this):"]
    if prediction_context.get("disease_label"):
        ctx_parts.append(f"- Disease assessed: {prediction_context['disease_label']}")
    if prediction_context.get("risk_label"):
        ctx_parts.append(f"- Risk level: {prediction_context['risk_label']}")
    if prediction_context.get("proba") is not None:
        ctx_parts.append(f"- Prediction probability: {prediction_context['proba']}%")
    if prediction_context.get("verdict"):
        ctx_parts.append(f"- Verdict: {prediction_context['verdict']}")
    ctx_parts.append(
        "\nUse this context only to explain the result to the user in simple language. "
        "Never question or modify the ML prediction."
    )
    return "\n".join(ctx_parts)


def chat(user_message: str, history: list, prediction_context: dict | None = None) -> str:
    """Send a message to Gemini and return the assistant reply.

    Parameters
    ----------
    user_message : str
        The user's latest message.
    history : list
        List of {"role": "user"|"model", "parts": [{"text": "..."}]} dicts
        representing prior conversation turns (for multi-turn context).
    prediction_context : dict | None
        Optional dict with keys: disease_label, risk_label, proba, verdict.
        Provided automatically when the user is on the result page.
    """
    api_key = _get_api_key()

    # ── Build the message text ──────────────────────────────────────────────
    context_note = build_context_prompt(prediction_context)
    first_user_text = (
        f"{context_note}\n\nUser question: {user_message}"
        if context_note
        else user_message
    )

    # ── Assemble Gemini 'contents' array ────────────────────────────────────
    # Prior history turns first, then the new user message.
    contents = list(history) + [{"role": "user", "parts": [{"text": first_user_text}]}]

    # ── Build request payload ───────────────────────────────────────────────
    # FIX: top-level key is 'systemInstruction' (camelCase), NOT 'system_instruction'.
    # The Gemini REST API v1beta strictly uses camelCase for all request fields.
    # Sending 'system_instruction' (snake_case) causes it to be silently ignored,
    # meaning Gemini receives NO system prompt and can respond to any topic,
    # or may return HTTP 400 depending on the model version.
    payload = {
        "systemInstruction": {                  # ← FIXED (was "system_instruction")
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": contents,
        "generationConfig": {                   # generationConfig is camelCase — was already correct
            "temperature": 0.4,
            "maxOutputTokens": 512,
        },
    }

    url = f"{GEMINI_API_URL}?key={api_key}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        candidates = data.get("candidates", [])
        if not candidates:
            # Gemini returned an empty candidates list — log the full response
            logger.warning("Gemini returned no candidates. Full response: %s", data)
            return "I'm sorry, I couldn't generate a response right now. Please try again."

        return candidates[0]["content"]["parts"][0]["text"].strip()

    except urllib.error.HTTPError as e:
        # FIX: log the FULL Gemini error body so it appears in the Flask log.
        # Previously this was logged but the log level / format made it hard to find.
        err_body = e.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini HTTP %s error.\nURL: %s\nResponse body: %s",
            e.code, url.replace(api_key, "***"), err_body
        )
        # Return user-friendly messages (exact HTTP code visible in server log)
        if e.code == 400:
            return (
                "The request was rejected by the AI service (HTTP 400). "
                "This usually means the API key format is invalid or the payload is malformed. "
                "Check your server log for the full Gemini error message."
            )
        if e.code == 401:
            return (
                "Authentication failed (HTTP 401). "
                "Your GEMINI_API_KEY is invalid. "
                "Verify it at https://aistudio.google.com/app/apikey"
            )
        if e.code == 403:
            return (
                "Access denied (HTTP 403). "
                "Your API key does not have permission to use this model. "
                "Check your key at https://aistudio.google.com/app/apikey"
            )
        if e.code == 404:
            return (
                "Model not found (HTTP 404). "
                "The model 'gemini-2.5-flash' may not be available on your account. "
                "Check the server log for details."
            )
        if e.code == 429:
            return (
                "The AI service is temporarily rate-limited (HTTP 429). "
                "The free tier allows 15 requests per minute. "
                "Please wait a moment and try again."
            )
        return (
            f"The AI service returned an unexpected error (HTTP {e.code}). "
            "Check your server log for the full response."
        )

    except urllib.error.URLError as e:
        # Network-level error (DNS failure, refused connection, timeout)
        logger.error("Gemini network error: %s", e.reason)
        return (
            "Could not reach the Gemini API. "
            "Please check your internet connection and try again."
        )

    except Exception as exc:
        logger.exception("Unexpected error during Gemini request: %s", exc)
        return (
            "An unexpected error occurred while contacting the AI service. "
            "Check your server log for details."
        )
