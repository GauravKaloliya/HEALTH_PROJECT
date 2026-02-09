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

# Valid enum values
VALID_GENDERS = {"male", "female", "other"}
VALID_LEVELS = {"low", "medium", "high"}


def validate_clinical_inputs(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    errors: Dict[str, str] = {}
    cleaned: Dict[str, Any] = {}

    # Age validation
    age = payload.get("age")
    if age is None:
        errors["age"] = "This field is required."
    else:
        try:
            age = float(age)
            if not 0 <= age <= 120:
                errors["age"] = "Age must be between 0 and 120."
            else:
                cleaned["age"] = age
        except (TypeError, ValueError):
            errors["age"] = "Must be a number."

    # Gender validation
    gender = payload.get("gender")
    if not gender:
        errors["gender"] = "Gender is required."
    elif gender.lower() not in VALID_GENDERS:
        errors["gender"] = "Gender must be Male, Female, or Other."
    else:
        cleaned["gender"] = gender.lower()

    # Tumor size validation
    tumor_size = payload.get("tumor_size_mm")
    if tumor_size is None:
        errors["tumor_size_mm"] = "This field is required."
    else:
        try:
            tumor_size = float(tumor_size)
            if not 0 <= tumor_size <= 200:
                errors["tumor_size_mm"] = "Tumor size must be between 0 and 200mm."
            else:
                cleaned["tumor_size_mm"] = tumor_size
        except (TypeError, ValueError):
            errors["tumor_size_mm"] = "Must be a number."

    # BMI validation
    bmi = payload.get("bmi")
    if bmi is None:
        errors["bmi"] = "This field is required."
    else:
        try:
            bmi = float(bmi)
            if not 10 <= bmi <= 60:
                errors["bmi"] = "BMI must be between 10 and 60."
            else:
                cleaned["bmi"] = bmi
        except (TypeError, ValueError):
            errors["bmi"] = "Must be a number."

    # Boolean fields validation
    boolean_fields = [
        "family_history",
        "smoking_history", 
        "diabetes",
        "inflammatory_bowel_disease",
        "genetic_mutation",
        "screening_history"
    ]
    for field in boolean_fields:
        value = payload.get(field)
        if value is None:
            errors[field] = "This field is required."
        else:
            cleaned[field] = bool(value)

    # Enum fields validation
    enum_fields = {
        "alcohol_consumption": "Alcohol consumption",
        "diet_risk": "Diet risk",
        "physical_activity": "Physical activity"
    }
    for field, label in enum_fields.items():
        value = payload.get(field)
        if not value:
            errors[field] = f"{label} is required."
        elif value.lower() not in VALID_LEVELS:
            errors[field] = f"{label} must be Low, Medium, or High."
        else:
            cleaned[field] = value.lower()

    return cleaned, errors


def preprocess_features(inputs: Dict[str, Any]) -> Dict[str, float]:
    # Encode categorical variables numerically
    gender_map = {"male": 0.5, "female": 0.5, "other": 0.5}
    level_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    
    features = {
        "age": inputs["age"] / 120,
        "tumor_size_mm": inputs["tumor_size_mm"] / 200,
        "bmi": inputs["bmi"] / 60,
        "gender": gender_map.get(inputs["gender"], 0.5),
        "family_history": 1.0 if inputs["family_history"] else 0.0,
        "smoking_history": 1.0 if inputs["smoking_history"] else 0.0,
        "alcohol_consumption": level_map.get(inputs["alcohol_consumption"], 0.5),
        "diet_risk": level_map.get(inputs["diet_risk"], 0.5),
        "physical_activity": 1.0 - level_map.get(inputs["physical_activity"], 0.5),  # Inverse: low activity = higher risk
        "diabetes": 1.0 if inputs["diabetes"] else 0.0,
        "inflammatory_bowel_disease": 1.0 if inputs["inflammatory_bowel_disease"] else 0.0,
        "genetic_mutation": 1.0 if inputs["genetic_mutation"] else 0.0,
        "screening_history": 1.0 if inputs["screening_history"] else 0.0,
    }
    return features


def mock_risk_model(features: Dict[str, float]) -> Dict[str, Any]:
    # Weighted risk calculation based on clinical factors
    # Higher weights for more significant risk factors
    score = (
        features["age"] * 0.15 +
        features["tumor_size_mm"] * 0.20 +
        features["bmi"] * 0.08 +
        features["family_history"] * 0.12 +
        features["smoking_history"] * 0.10 +
        features["alcohol_consumption"] * 0.06 +
        features["diet_risk"] * 0.05 +
        features["physical_activity"] * 0.05 +
        features["diabetes"] * 0.08 +
        features["inflammatory_bowel_disease"] * 0.04 +
        features["genetic_mutation"] * 0.15 +
        (1.0 - features["screening_history"]) * 0.02  # No screening = slightly higher risk
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
