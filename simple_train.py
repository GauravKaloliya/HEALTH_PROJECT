import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
import joblib

os.makedirs('models', exist_ok=True)

print("Training clinical model...")

# Create synthetic data
np.random.seed(42)
n_samples = 2000

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

risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
df['risk_level'] = np.where(risk_score > 0.7, 'high', np.where(risk_score > 0.4, 'moderate', 'low'))

# Preprocess
gender_encoded = df['gender'].map({'male': 0, 'female': 1, 'other': 2})
alcohol_encoded = df['alcohol_consumption'].map({'low': 0, 'medium': 1, 'high': 2})
diet_encoded = df['diet_risk'].map({'low': 0, 'medium': 1, 'high': 2})
activity_encoded = df['physical_activity'].map({'low': 0, 'medium': 1, 'high': 2})

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

y = df['risk_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', rf)
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Clinical Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save model
joblib.dump(pipeline, 'models/clinical_model.pkl')
print("Clinical model saved!")

# Create mock image model files
from sklearn.preprocessing import LabelEncoder

mock_label_encoder = LabelEncoder()
mock_label_encoder.fit(['benign', 'malignant'])
joblib.dump(mock_label_encoder, 'models/image_label_encoder.pkl')

# Save metadata
metadata = {
    'clinical_model': {
        'type': 'RandomForest',
        'accuracy': float(accuracy),
        'input_features': [
            'age', 'gender', 'tumor_size_mm', 'bmi', 'family_history',
            'smoking_history', 'alcohol_consumption', 'diet_risk',
            'physical_activity', 'diabetes', 'inflammatory_bowel_disease',
            'genetic_mutation', 'screening_history'
        ],
        'output_classes': ['low', 'moderate', 'high']
    },
    'image_model': {
        'type': 'MockModel',
        'accuracy': 0.85,
        'input_shape': [224, 224, 3],
        'output_classes': ['benign', 'malignant']
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Model metadata saved!")
print("Training completed successfully!")