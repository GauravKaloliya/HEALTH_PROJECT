from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_IMAGE_SIZE_MB = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("health_api")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def validate_clinical_inputs(payload: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, str]]:
    errors: Dict[str, str] = {}

    def get_float(field: str) -> float | None:
        value = payload.get(field)
        if value is None:
            errors[field] = "This field is required."
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            errors[field] = "Must be a number."
            return None

    age = get_float("age")
    tumor_size = get_float("tumor_size_mm")
    bmi = get_float("bmi")

    if age is not None and not 0 <= age <= 120:
        errors["age"] = "Age must be between 0 and 120."
    if tumor_size is not None and not 0 <= tumor_size <= 200:
        errors["tumor_size_mm"] = "Tumor size must be between 0 and 200mm."
    if bmi is not None and not 10 <= bmi <= 60:
        errors["bmi"] = "BMI must be between 10 and 60."

    cleaned = {
        "age": age,
        "tumor_size_mm": tumor_size,
        "bmi": bmi,
    }
    return cleaned, errors


def preprocess_features(inputs: Dict[str, float]) -> Dict[str, float]:
    return {
        "age": inputs["age"] / 120,
        "tumor_size_mm": inputs["tumor_size_mm"] / 200,
        "bmi": inputs["bmi"] / 60,
    }


def mock_risk_model(features: Dict[str, float]) -> Dict[str, Any]:
    score = (
        features["age"] * 0.35
        + features["tumor_size_mm"] * 0.45
        + features["bmi"] * 0.2
    )
    score = min(max(score, 0.0), 1.0)
    confidence = 0.7 + (0.3 * (1 - math.fabs(score - 0.5) * 2))
    if score >= 0.7:
        level = "high"
    elif score >= 0.4:
        level = "moderate"
    else:
        level = "low"

    return {
        "risk_score": round(score, 3),
        "risk_level": level,
        "confidence": round(confidence, 3),
    }


def validate_image_upload(file_storage) -> str | None:
    if file_storage is None:
        return "Image file is required."
    if file_storage.mimetype not in ALLOWED_IMAGE_TYPES:
        return "Unsupported file type."
    file_storage.stream.seek(0, os.SEEK_END)
    size_mb = file_storage.stream.tell() / (1024 * 1024)
    file_storage.stream.seek(0)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return f"File must be under {MAX_IMAGE_SIZE_MB}MB."
    return None


def mock_image_classifier(file_storage) -> Dict[str, Any]:
    filename = (file_storage.filename or "scan").lower()
    if "benign" in filename:
        label = "benign"
        confidence = 0.88
    elif "malignant" in filename:
        label = "malignant"
        confidence = 0.9
    else:
        label = "indeterminate"
        confidence = 0.75

    return {
        "label": label,
        "confidence": confidence,
    }


def build_error_response(errors: Dict[str, str], status_code: int = 400):
    return jsonify({"status": "error", "errors": errors}), status_code


@app.route("/api/v1/predict-risk", methods=["POST"])
def predict_risk():
    payload = request.get_json(silent=True) or {}
    cleaned, errors = validate_clinical_inputs(payload)
    if errors:
        logger.warning("Validation errors on predict-risk: %s", errors)
        return build_error_response(errors)

    features = preprocess_features(cleaned)
    prediction = mock_risk_model(features)
    logger.info("Risk prediction generated with score %s", prediction["risk_score"])
    return jsonify({"status": "success", "data": prediction})


@app.route("/api/v1/classify-image", methods=["POST"])
def classify_image():
    file_storage = request.files.get("image")
    error = validate_image_upload(file_storage)
    if error:
        logger.warning("Image validation error: %s", error)
        return build_error_response({"image": error})

    classification = mock_image_classifier(file_storage)
    logger.info("Image classified as %s", classification["label"])
    return jsonify({"status": "success", "data": classification})


@app.route("/api/v1/final-assessment", methods=["POST"])
def final_assessment():
    payload = request.get_json(silent=True) or {}
    clinical = payload.get("clinical_result")
    imaging = payload.get("image_result")

    errors: Dict[str, str] = {}
    if not clinical:
        errors["clinical_result"] = "Clinical result data is required."
    if not imaging:
        errors["image_result"] = "Image result data is required."
    if errors:
        logger.warning("Final assessment errors: %s", errors)
        return build_error_response(errors)

    risk_score = float(clinical.get("risk_score", 0))
    image_confidence = float(imaging.get("confidence", 0))
    image_label = imaging.get("label", "indeterminate")

    combined_score = min(max((risk_score * 0.7) + (image_confidence * 0.3), 0), 1)
    if image_label == "malignant":
        combined_score = min(1, combined_score + 0.15)
    elif image_label == "benign":
        combined_score = max(0, combined_score - 0.1)

    if combined_score >= 0.7:
        final_level = "high"
    elif combined_score >= 0.4:
        final_level = "moderate"
    else:
        final_level = "low"

    response = {
        "final_score": round(combined_score, 3),
        "final_risk_level": final_level,
        "summary": f"Overall risk is {final_level} based on combined indicators.",
    }

    logger.info("Final assessment completed with level %s", final_level)
    return jsonify({"status": "success", "data": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
