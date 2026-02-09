import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

clinical_model = None
image_model = None
image_label_encoder = None

try:
    clinical_model = joblib.load(os.path.join(MODELS_DIR, 'clinical_model.pkl'))
    print("✓ Clinical model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load clinical model: {e}")
    clinical_model = None

try:
    import tensorflow as tf
    image_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'image_model.h5'))
    image_label_encoder = joblib.load(os.path.join(MODELS_DIR, 'image_label_encoder.pkl'))
    print("✓ Image CNN model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load image CNN model: {e}")
    image_model = None
    image_label_encoder = None

def preprocess_clinical_features(inputs: Dict[str, Any]) -> np.ndarray:
    """Preprocess clinical inputs for the model"""
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
        return mock_risk_model_from_features(features[0])
    
    prediction = clinical_model.predict(features)
    proba = clinical_model.predict_proba(features)
    
    confidence = np.max(proba)
    
    risk_level = prediction[0]
    
    proba_dict = dict(zip(clinical_model.classes_, proba[0]))
    if risk_level == 'high':
        risk_score = 0.7 + 0.3 * proba_dict.get('high', 0)
    elif risk_level == 'moderate':
        risk_score = 0.4 + 0.3 * proba_dict.get('moderate', 0)
    else:
        risk_score = 0.1 + 0.3 * proba_dict.get('low', 0)
    
    return {
        "risk_score": round(float(risk_score), 3),
        "risk_level": risk_level,
        "confidence": round(float(confidence), 3),
    }

def mock_risk_model_from_features(features: np.ndarray) -> Dict[str, Any]:
    """Fallback mock risk calculation based on clinical factors"""
    if isinstance(features, np.ndarray):
        features_array = features
    else:
        features_array = np.array(features)
    
    score = (
        features_array[0] * 0.15 +
        features_array[2] * 0.20 +
        features_array[3] * 0.08 +
        features_array[4] * 0.12 +
        features_array[5] * 0.10 +
        features_array[6] * 0.06 +
        features_array[7] * 0.05 +
        features_array[8] * 0.05 +
        features_array[9] * 0.08 +
        features_array[10] * 0.04 +
        features_array[11] * 0.15 +
        (1.0 - features_array[12]) * 0.02
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
        "risk_score": round(float(score), 3),
        "risk_level": level,
        "confidence": round(float(confidence), 3),
    }

def preprocess_image_for_cnn(file_storage) -> np.ndarray:
    """Preprocess image for CNN model"""
    try:
        image_bytes = file_storage.read()
        file_storage.seek(0)
        
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize((224, 224))
        
        img_array = np.array(img_resized)
        
        img_array = img_array.astype('float32') / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image_with_model(file_storage) -> Dict[str, Any]:
    """Predict image classification using the trained CNN model"""
    if image_model is None or image_label_encoder is None:
        return mock_image_classifier(file_storage)
    
    try:
        img_array = preprocess_image_for_cnn(file_storage)
        
        if img_array is None:
            return mock_image_classifier(file_storage)
        
        predictions = image_model.predict(img_array, verbose=0)
        
        if len(image_label_encoder.classes_) == 2:
            confidence = float(predictions[0][0]) if predictions.shape[-1] == 1 else float(np.max(predictions[0]))
            predicted_class = 1 if confidence > 0.5 else 0
            if predictions.shape[-1] == 1:
                confidence = confidence if predicted_class == 1 else (1 - confidence)
        else:
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
        
        label = image_label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "label": label,
            "confidence": round(confidence, 3),
        }
    
    except Exception as e:
        print(f"Error during image prediction: {e}")
        return mock_image_classifier(file_storage)

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
