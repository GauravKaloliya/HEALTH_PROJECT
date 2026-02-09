import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

# Load models at startup
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try:
    clinical_model = joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl'))
    image_model = joblib.load(os.path.join(MODELS_DIR, 'image_model.pkl'))
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    clinical_model = None
    image_model = None

def preprocess_clinical_features(inputs: Dict[str, Any]) -> np.ndarray:
    """Preprocess clinical inputs for the model"""
    # Encode categorical variables numerically
    gender_map = {"male": 0, "female": 1, "other": 2}
    level_map = {"low": 0, "medium": 1, "high": 2}
    
    features = np.array([[
        inputs["age"],
        gender_map.get(inputs["gender"], 1),
        inputs["tumor_size_mm"],
        inputs["bmi"],
        1.0 if inputs["family_history"] else 0.0,
        1.0 if inputs["smoking_history"] else 0.0,
        level_map.get(inputs["alcohol_consumption"], 1),
        level_map.get(inputs["diet_risk"], 1),
        level_map.get(inputs["physical_activity"], 1),
        1.0 if inputs["diabetes"] else 0.0,
        1.0 if inputs["inflammatory_bowel_disease"] else 0.0,
        1.0 if inputs["genetic_mutation"] else 0.0,
        1.0 if inputs["screening_history"] else 0.0,
    ]])
    
    return features

def predict_risk_with_model(features: np.ndarray) -> Dict[str, Any]:
    """Predict risk using the trained clinical model"""
    if clinical_model is None:
        # Fallback to mock prediction if model not available
        return mock_risk_model_from_features(features[0])
    
    # Make prediction
    prediction = clinical_model.predict(features)
    proba = clinical_model.predict_proba(features)
    
    # Get confidence from probabilities
    confidence = np.max(proba)
    
    # Map prediction to risk level and score
    risk_level = prediction[0]
    
    # Create a risk score based on probabilities
    proba_dict = dict(zip(clinical_model.classes_, proba[0]))
    if risk_level == 'high':
        risk_score = 0.7 + 0.3 * proba_dict['high']
    elif risk_level == 'moderate':
        risk_score = 0.4 + 0.3 * proba_dict['moderate']
    else:  # low
        risk_score = 0.1 + 0.3 * proba_dict['low']
    
    return {
        "risk_score": round(float(risk_score), 3),
        "risk_level": risk_level,
        "confidence": round(float(confidence), 3),
    }

def mock_risk_model_from_features(features: Dict[str, float]) -> Dict[str, Any]:
    """Fallback mock risk calculation based on clinical factors"""
    # This is a simplified version of the original mock model
    score = (
        features[0] * 0.15 +          # age
        features[2] * 0.20 +          # tumor_size_mm
        features[3] * 0.08 +          # bmi
        features[4] * 0.12 +          # family_history
        features[5] * 0.10 +          # smoking_history
        features[6] * 0.06 +          # alcohol_consumption
        features[7] * 0.05 +          # diet_risk
        features[8] * 0.05 +          # physical_activity
        features[9] * 0.08 +          # diabetes
        features[10] * 0.04 +         # inflammatory_bowel_disease
        features[11] * 0.15 +         # genetic_mutation
        (1.0 - features[12]) * 0.02  # screening_history
    )
    
    score = min(max(score, 0.0), 1.0)
    confidence = 0.7 + (0.3 * (1 - abs(score - 0.5) * 2))
    
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

def extract_image_features_from_filename(filename: str) -> np.ndarray:
    """Extract features from image filename for mock image classification"""
    # This is a simple mock feature extraction based on filename
    # In a real scenario, you would extract features from the actual image
    filename = filename.lower()
    
    # Simple heuristic based on filename
    if "benign" in filename:
        texture = np.random.uniform(0.1, 0.4)
        color_var = np.random.uniform(0.2, 0.5)
        edge_int = np.random.uniform(0.3, 0.6)
        symmetry = np.random.uniform(0.7, 0.9)
        size = np.random.uniform(5, 30)
    elif "malignant" in filename:
        texture = np.random.uniform(0.6, 0.9)
        color_var = np.random.uniform(0.7, 0.9)
        edge_int = np.random.uniform(0.8, 1.0)
        symmetry = np.random.uniform(0.2, 0.5)
        size = np.random.uniform(30, 80)
    else:
        texture = np.random.uniform(0.3, 0.7)
        color_var = np.random.uniform(0.4, 0.7)
        edge_int = np.random.uniform(0.5, 0.8)
        symmetry = np.random.uniform(0.4, 0.7)
        size = np.random.uniform(15, 50)
    
    return np.array([[texture, color_var, edge_int, symmetry, size]])

def predict_image_with_model(file_storage) -> Dict[str, Any]:
    """Predict image classification using the trained image model"""
    if image_model is None:
        # Fallback to mock prediction if model not available
        return mock_image_classifier(file_storage)
    
    # Extract features from filename (in real scenario, extract from image)
    filename = (file_storage.filename or "scan").lower()
    features = extract_image_features_from_filename(filename)
    
    # Make prediction
    prediction = image_model.predict(features)
    proba = image_model.predict_proba(features)
    
    # Get confidence from probabilities
    confidence = np.max(proba)
    label = prediction[0]
    
    return {
        "label": label,
        "confidence": round(float(confidence), 3),
    }

def mock_image_classifier(file_storage) -> Dict[str, Any]:
    """Fallback mock image classification"""
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
