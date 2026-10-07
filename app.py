"""
Flask web app for the Customer Support Ticket Classifier.

Loads the saved TF-IDF vectorizer and Logistic Regression model from
model/*.pkl and predicts Ticket Type from a user-submitted message.
"""

from __future__ import annotations

import os
import re
import string

import joblib
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths and Flask setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Text cleaning (must match training)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Apply the same cleaning steps used when the model was trained."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("{product_purchased}", "product")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    allowed = string.ascii_lowercase + string.digits + " "
    text = "".join(ch if ch in allowed else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Load trained model files once at startup
# ---------------------------------------------------------------------------
def load_artifacts():
    """Load the saved vectorizer and classifier from the model/ folder."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Missing vectorizer at {VECTORIZER_PATH}. "
            "Run train_ticket_classifier.py first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. "
            "Run train_ticket_classifier.py first."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


vectorizer, model = load_artifacts()


def predict_ticket_type(message: str) -> dict:
    """
    Clean the message, convert it with TF-IDF, and return the prediction.

    Returns a dictionary with:
      - predicted_type: the predicted Ticket Type label
      - confidence: probability of that label (0 to 1)
      - probabilities: all class probabilities
    """
    cleaned = clean_text(message)
    if not cleaned:
        return {
            "predicted_type": None,
            "confidence": 0.0,
            "probabilities": {},
            "error": "Please enter a ticket description.",
        }

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_names = list(model.classes_)

    prob_map = {
        class_name: float(round(prob, 4))
        for class_name, prob in zip(class_names, probabilities)
    }
    confidence = float(round(max(probabilities), 4))

    return {
        "predicted_type": prediction,
        "confidence": confidence,
        "probabilities": prob_map,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Used by the Jenkins deploy step to confirm the app is up."""
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET", "POST"])
def index():
    """Home page: form to enter a ticket message and see the prediction."""
    result = None
    message = ""

    if request.method == "POST":
        message = (request.form.get("message") or "").strip()
        result = predict_ticket_type(message)

    return render_template(
        "index.html",
        message=message,
        result=result,
        ticket_types=list(model.classes_),
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Simple JSON API for prediction.

    Example request body:
      {"message": "I was charged twice on my invoice."}
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    result = predict_ticket_type(message)

    status_code = 400 if result.get("error") else 200
    return jsonify(result), status_code


# ---------------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind all interfaces so Docker can publish the port.
    # Set FLASK_DEBUG=1 when you want the reloader while developing.
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
