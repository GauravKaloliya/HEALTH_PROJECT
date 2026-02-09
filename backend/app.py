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


def validate_clinical_inputs(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
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

    def get_bool(field: str) -> bool | None:
        value = payload.get(field)
        if value is None:
            errors[field] = "This field is required."
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
        errors[field] = "Must be true or false."
        return None

    def get_enum(field: str, allowed_values: list[str]) -> str | None:
        value = payload.get(field)
        if value is None:
            errors[field] = "This field is required."
            return None
        if value not in allowed_values:
            errors[field] = f"Must be one of: {', '.join(allowed_values)}."
            return None
        return value

    age = get_float("age")
    tumor_size = get_float("tumor_size_mm")
    bmi = get_float("bmi")
    
    # New clinical fields
    gender = get_enum("gender", ["male", "female", "other"])
    family_history = get_bool("family_history")
    smoking_history = get_bool("smoking_history")
    alcohol_consumption = get_enum("alcohol_consumption", ["none", "moderate", "high"])
    diet_risk = get_enum("diet_risk", ["low", "moderate", "high"])
    physical_activity = get_enum("physical_activity", ["sedentary", "light", "moderate", "vigorous"])
    diabetes = get_bool("diabetes")
    ibd = get_bool("ibd")
    genetic_mutation = get_bool("genetic_mutation")
    screening_history = get_bool("screening_history")

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
        "gender": gender,
        "family_history": family_history,
        "smoking_history": smoking_history,
        "alcohol_consumption": alcohol_consumption,
        "diet_risk": diet_risk,
        "physical_activity": physical_activity,
        "diabetes": diabetes,
        "ibd": ibd,
        "genetic_mutation": genetic_mutation,
        "screening_history": screening_history,
    }
    return cleaned, errors


def preprocess_features(inputs: Dict[str, Any]) -> Dict[str, float]:
    features = {}
    
    # Numerical features (normalized)
    features["age"] = inputs["age"] / 120
    features["tumor_size_mm"] = inputs["tumor_size_mm"] / 200
    features["bmi"] = inputs["bmi"] / 60
    
    # Categorical features (one-hot encoded)
    # Gender encoding
    features["gender_male"] = 1 if inputs["gender"] == "male" else 0
    features["gender_female"] = 1 if inputs["gender"] == "female" else 0
    features["gender_other"] = 1 if inputs["gender"] == "other" else 0
    
    # Boolean features (as 0/1)
    features["family_history"] = 1 if inputs["family_history"] else 0
    features["smoking_history"] = 1 if inputs["smoking_history"] else 0
    features["diabetes"] = 1 if inputs["diabetes"] else 0
    features["ibd"] = 1 if inputs["ibd"] else 0
    features["genetic_mutation"] = 1 if inputs["genetic_mutation"] else 0
    features["screening_history"] = 1 if inputs["screening_history"] else 0
    
    # Ordinal categorical features (encoded as ordinal values)
    # Alcohol consumption: none=0, moderate=0.5, high=1
    alcohol_map = {"none": 0, "moderate": 0.5, "high": 1}
    features["alcohol_consumption"] = alcohol_map[inputs["alcohol_consumption"]]
    
    # Diet risk: low=0, moderate=0.5, high=1
    diet_map = {"low": 0, "moderate": 0.5, "high": 1}
    features["diet_risk"] = diet_map[inputs["diet_risk"]]
    
    # Physical activity: sedentary=0, light=0.33, moderate=0.67, vigorous=1
    activity_map = {"sedentary": 0, "light": 0.33, "moderate": 0.67, "vigorous": 1}
    features["physical_activity"] = activity_map[inputs["physical_activity"]]
    
    return features


def mock_risk_model(features: Dict[str, float]) -> Dict[str, Any]:
    # Calculate risk score using weighted factors
    # Base numerical factors
    score = (
        features["age"] * 0.15  # Age is significant but reduced weight for new factors
        + features["tumor_size_mm"] * 0.25  # Tumor size is still important
        + features["bmi"] * 0.1  # BMI weight reduced
        # Gender factors
        + features["gender_male"] * 0.05  # Slightly higher risk for males
        + features["gender_female"] * 0.02  # Lower baseline risk for females
        # Lifestyle risk factors
        + features["family_history"] * 0.15  # Strong genetic factor
        + features["smoking_history"] * 0.1  # Smoking is a significant risk
        + features["alcohol_consumption"] * 0.08  # Alcohol consumption risk
        + features["diet_risk"] * 0.07  # Poor diet increases risk
        # Health conditions
        + features["diabetes"] * 0.08  # Diabetes increases risk
        + features["ibd"] * 0.05  # IBD has moderate risk association
        + features["genetic_mutation"] * 0.12  # Genetic factors are significant
        # Protective factors
        - features["physical_activity"] * 0.06  # Physical activity is protective
        + features["screening_history"] * 0.03  # Regular screening slightly increases detection
    )
    
    # Ensure score is within valid range
    score = min(max(score, 0.0), 1.0)
    
    # Calculate confidence based on number of risk factors present
    risk_factors_count = sum([
        features["family_history"],
        features["smoking_history"],
        features["diabetes"],
        features["genetic_mutation"],
        features["diet_risk"],
        features["alcohol_consumption"],
        1 if features["tumor_size_mm"] > 0.5 else 0,  # Large tumor
        1 if features["age"] > 0.6 else 0,  # Older age
        1 if features["bmi"] > 0.7 else 0,  # High BMI
    ])
    
    # Base confidence increases with more comprehensive data
    confidence = 0.75 + (0.2 * min(risk_factors_count / 8, 1))
    confidence = min(confidence, 0.95)  # Cap at 95%
    
    # Adjust risk levels based on comprehensive scoring
    if score >= 0.65:
        level = "high"
    elif score >= 0.35:
        level = "moderate"
    else:
        level = "low"

    return {
        "risk_score": round(score, 3),
        "risk_level": level,
        "confidence": round(confidence, 3),
        "risk_factors_count": risk_factors_count,
        "contributing_factors": {
            "age": round(features["age"], 3),
            "tumor_size": round(features["tumor_size_mm"], 3),
            "bmi": round(features["bmi"], 3),
            "family_history": bool(features["family_history"]),
            "smoking_history": bool(features["smoking_history"]),
            "genetic_mutation": bool(features["genetic_mutation"]),
            "diabetes": bool(features["diabetes"]),
        }
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
