import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Since we can't download from Kaggle without credentials, let's create synthetic datasets
# that mimic the structure of the real datasets

def create_synthetic_clinical_data():
    """Create synthetic clinical data similar to colorectal cancer dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'tumor_size_mm': np.random.uniform(5, 100, n_samples),
        'bmi': np.random.uniform(18, 40, n_samples),
        'family_history': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'smoking_history': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
        'alcohol_consumption': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.4, 0.4, 0.2]),
        'diet_risk': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2]),
        'physical_activity': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.4, 0.3]),
        'diabetes': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'inflammatory_bowel_disease': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'genetic_mutation': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
        'screening_history': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
    }
    
    # Create target variable based on risk factors (synthetic)
    df = pd.DataFrame(data)
    risk_score = (
        df['age'] * 0.02 +
        df['tumor_size_mm'] * 0.05 +
        df['bmi'] * 0.03 +
        df['family_history'].astype(int) * 15 +
        df['smoking_history'].astype(int) * 12 +
        df['genetic_mutation'].astype(int) * 20 +
        df['alcohol_consumption'].map({'low': 0, 'medium': 5, 'high': 10}) +
        df['diet_risk'].map({'low': 0, 'medium': 5, 'high': 10}) +
        (1 - df['physical_activity'].map({'low': 0, 'medium': 0.5, 'high': 1})) * 8 +
        df['diabetes'].astype(int) * 10 +
        df['inflammatory_bowel_disease'].astype(int) * 15 +
        (1 - df['screening_history'].astype(int)) * 5
    )
    
    # Normalize risk score to 0-1 range
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    
    # Create binary target (high risk vs low risk)
    df['risk_level'] = np.where(risk_score > 0.7, 'high', np.where(risk_score > 0.4, 'moderate', 'low'))
    df['risk_score'] = risk_score
    
    return df

def create_synthetic_image_data():
    """Create synthetic image data structure"""
    # For image data, we'll create a simple CSV with image features
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'image_id': [f'image_{i}' for i in range(n_samples)],
        'texture_score': np.random.uniform(0, 1, n_samples),
        'color_variation': np.random.uniform(0, 1, n_samples),
        'edge_intensity': np.random.uniform(0, 1, n_samples),
        'symmetry_score': np.random.uniform(0, 1, n_samples),
        'size_mm': np.random.uniform(5, 80, n_samples),
        'label': np.random.choice(['benign', 'malignant'], n_samples, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)

def preprocess_clinical_data(df):
    """Preprocess clinical data for modeling"""
    # Encode categorical variables
    gender_encoded = df['gender'].map({'male': 0, 'female': 1, 'other': 2})
    alcohol_encoded = df['alcohol_consumption'].map({'low': 0, 'medium': 1, 'high': 2})
    diet_encoded = df['diet_risk'].map({'low': 0, 'medium': 1, 'high': 2})
    activity_encoded = df['physical_activity'].map({'low': 0, 'medium': 1, 'high': 2})
    
    # Create feature matrix
    X = pd.DataFrame({
        'age': df['age'],
        'gender': gender_encoded,
        'tumor_size_mm': df['tumor_size_mm'],
        'bmi': df['bmi'],
        'family_history': df['family_history'].astype(int),
        'smoking_history': df['smoking_history'].astype(int),
        'alcohol_consumption': alcohol_encoded,
        'diet_risk': diet_encoded,
        'physical_activity': activity_encoded,
        'diabetes': df['diabetes'].astype(int),
        'inflammatory_bowel_disease': df['inflammatory_bowel_disease'].astype(int),
        'genetic_mutation': df['genetic_mutation'].astype(int),
        'screening_history': df['screening_history'].astype(int)
    })
    
    # Target variable - we'll predict risk level
    y = df['risk_level']
    
    return X, y

def preprocess_image_data(df):
    """Preprocess image data for modeling"""
    X = df[['texture_score', 'color_variation', 'edge_intensity', 'symmetry_score', 'size_mm']]
    y = df['label']
    
    return X, y

def train_clinical_model(X_train, y_train):
    """Train a clinical risk prediction model"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def train_image_model(X_train, y_train):
    """Train an image classification model"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def main():
    print("Creating synthetic datasets...")
    
    # Create clinical dataset
    clinical_df = create_synthetic_clinical_data()
    clinical_df.to_csv('data/clinical_dataset.csv', index=False)
    print(f"Clinical dataset saved: {clinical_df.shape}")
    
    # Create image dataset
    image_df = create_synthetic_image_data()
    image_df.to_csv('data/image_dataset.csv', index=False)
    print(f"Image dataset saved: {image_df.shape}")
    
    # Preprocess clinical data
    X_clinical, y_clinical = preprocess_clinical_data(clinical_df)
    X_clinical_train, X_clinical_test, y_clinical_train, y_clinical_test = train_test_split(
        X_clinical, y_clinical, test_size=0.2, random_state=42
    )
    
    # Train clinical model
    print("Training clinical model...")
    clinical_model = train_clinical_model(X_clinical_train, y_clinical_train)
    
    # Evaluate clinical model
    clinical_accuracy = clinical_model.score(X_clinical_test, y_clinical_test)
    print(f"Clinical model accuracy: {clinical_accuracy:.3f}")
    
    # Save clinical model
    joblib.dump(clinical_model, 'models/clinical_model.pkl')
    print("Clinical model saved")
    
    # Preprocess image data
    X_image, y_image = preprocess_image_data(image_df)
    X_image_train, X_image_test, y_image_train, y_image_test = train_test_split(
        X_image, y_image, test_size=0.2, random_state=42
    )
    
    # Train image model
    print("Training image model...")
    image_model = train_image_model(X_image_train, y_image_train)
    
    # Evaluate image model
    image_accuracy = image_model.score(X_image_test, y_image_test)
    print(f"Image model accuracy: {image_accuracy:.3f}")
    
    # Save image model
    joblib.dump(image_model, 'models/image_model.pkl')
    print("Image model saved")
    
    # Save model metadata
    model_metadata = {
        'clinical_model_accuracy': clinical_accuracy,
        'image_model_accuracy': image_accuracy,
        'clinical_features': list(X_clinical.columns),
        'image_features': list(X_image.columns),
        'clinical_target': 'risk_level',
        'image_target': 'label'
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("All models trained and saved successfully!")

if __name__ == '__main__':
    main()
