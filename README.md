# Health Assessment Application with Machine Learning Models

A comprehensive health assessment platform with a ReactJS frontend and Flask API backend. The system combines clinical input data and imaging analysis using trained machine learning models to produce a final risk assessment, complete with validation, confidence scores, and a medical disclaimer.

## Project Structure

```
.
 backend
   ├── app.py                  # Flask API with ML model integration
   ├── model_utils.py          # Model loading and inference utilities
   └── requirements.txt        # Python dependencies
 data
   ├── clinical_dataset.csv    # Synthetic clinical training data
   └── image_dataset.csv       # Synthetic image training data
 models
   ├── clinical_model.pkl      # Trained clinical risk prediction model
   ├── image_model.pkl         # Trained image classification model
   └── model_metadata.json     # Model performance metrics
 frontend
   ├── package.json
   ├── public
   │   └── index.html
   └── src
       ├── App.js
       ├── components
       │   ├── ClinicalDataForm.js
       │   ├── ImageUpload.js
       │   ├── LoadingSpinner.js
       │   ├── MedicalDisclaimer.js
       │   └── ResultsDashboard.js
       ├── index.js
       ├── services
       │   └── api.js
       └── styles.css
 docker-compose.yml
```

## Machine Learning Models

The application now uses trained machine learning models instead of mock functions:

1. **Clinical Risk Prediction Model**
   - **Type**: Random Forest Classifier
   - **Accuracy**: 89.5%
   - **Input Features**: Age, gender, tumor size, BMI, family history, smoking history, alcohol consumption, diet risk, physical activity, diabetes, inflammatory bowel disease, genetic mutation, screening history
   - **Output**: Risk level (low, moderate, high) with confidence score

2. **Image Classification Model**
   - **Type**: Random Forest Classifier
   - **Accuracy**: 54.0%
   - **Input Features**: Texture score, color variation, edge intensity, symmetry score, size
   - **Output**: Classification (benign, malignant) with confidence score

## Backend Setup (Flask)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The API will be available at `http://localhost:5000`.

### API Endpoints

- `POST /api/v1/predict-risk`
  - Body: `{ "age": 45, "gender": "male", "tumor_size_mm": 30, "bmi": 24.5, "family_history": true, "smoking_history": false, "alcohol_consumption": "low", "diet_risk": "medium", "physical_activity": "high", "diabetes": false, "inflammatory_bowel_disease": false, "genetic_mutation": false, "screening_history": true }`
  - Response: risk score, level, and confidence from the trained clinical model.

- `POST /api/v1/classify-image`
  - Multipart form-data with `image` file (PNG/JPG, up to 5MB).
  - Response: label and confidence from the trained image classification model.

- `POST /api/v1/final-assessment`
  - Body: `{ "clinical_result": { ... }, "image_result": { ... } }`
  - Response: final score and risk level combining both model outputs.

## Model Training

The models were trained on synthetic datasets that mimic the structure of real colorectal cancer datasets:

1. **Clinical Dataset**: 1000 samples with 13 features and risk level targets
2. **Image Dataset**: 500 samples with 5 image features and classification targets

To retrain the models:

```bash
python download_and_preprocess.py
```

This will:
1. Generate synthetic datasets
2. Train both models using Random Forest classifiers
3. Save the trained models as .pkl files
4. Save model metadata with accuracy scores

## Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

The UI will be available at `http://localhost:3000` and expects the backend on `http://localhost:5000`.

To configure a different backend URL, set `REACT_APP_API_BASE_URL` in a `.env` file.

## Docker Compose

Run both services together:

```bash
docker-compose up --build
```

The frontend will be exposed on port `3000` and the backend on port `5000`.

## Security & Validation

- Client and server validation for clinical metrics.
- Image file type and size validation.
- Structured error handling with clear messaging.
- Medical disclaimer for compliance in healthcare contexts.
- Model fallback to mock predictions if trained models are unavailable.

## Model Performance

- **Clinical Model Accuracy**: 89.5%
- **Image Model Accuracy**: 54.0%
- **Combined Assessment**: Weighted combination (70% clinical, 30% image) with adjustments based on image classification

## Technical Notes

- Models are loaded at startup and cached for performance
- Fallback to mock predictions ensures system availability even if models fail to load
- StandardScaler is used for feature normalization
- Random Forest classifiers provide good accuracy with interpretable results
- Synthetic data generation allows for testing without requiring real patient data
