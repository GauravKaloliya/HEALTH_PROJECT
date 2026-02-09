# Health Assessment Application

A comprehensive health assessment platform with a ReactJS frontend and Flask API backend. The system combines clinical input data and imaging analysis to produce a final risk assessment, complete with validation, confidence scores, and a medical disclaimer.

## Project Structure

```
.
├── backend
│   ├── app.py
│   └── requirements.txt
├── frontend
│   ├── package.json
│   ├── public
│   │   └── index.html
│   └── src
│       ├── App.js
│       ├── components
│       │   ├── ClinicalDataForm.js
│       │   ├── ImageUpload.js
│       │   ├── LoadingSpinner.js
│       │   ├── MedicalDisclaimer.js
│       │   └── ResultsDashboard.js
│       ├── index.js
│       ├── services
│       │   └── api.js
│       └── styles.css
└── docker-compose.yml
```

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
  - Body: `{ "age": 45, "tumor_size_mm": 30, "bmi": 24.5 }`
  - Response: risk score, level, and confidence.

- `POST /api/v1/classify-image`
  - Multipart form-data with `image` file (PNG/JPG, up to 5MB).
  - Response: label and confidence.

- `POST /api/v1/final-assessment`
  - Body: `{ "clinical_result": { ... }, "image_result": { ... } }`
  - Response: final score and risk level.

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
