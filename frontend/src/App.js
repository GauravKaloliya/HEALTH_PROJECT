import React, { useState } from "react";
import ClinicalDataForm from "./components/ClinicalDataForm";
import ImageUpload from "./components/ImageUpload";
import ResultsDashboard from "./components/ResultsDashboard";
import LoadingSpinner from "./components/LoadingSpinner";
import MedicalDisclaimer from "./components/MedicalDisclaimer";
import { classifyImage, finalAssessment, predictRisk } from "./services/api";

const steps = ["Clinical Data", "Imaging", "Results"];

const App = () => {
  const [step, setStep] = useState(0);
  const [clinicalResult, setClinicalResult] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [finalResult, setFinalResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClinicalSubmit = async (data) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictRisk(data);
      setClinicalResult(result);
      setStep(1);
    } catch (err) {
      setError(err.response?.data?.errors || "Unable to submit clinical data.");
    } finally {
      setLoading(false);
    }
  };

  const handleImageSubmit = async (file) => {
    setLoading(true);
    setError(null);
    try {
      const result = await classifyImage(file);
      setImageResult(result);
      const finalData = await finalAssessment(clinicalResult, result);
      setFinalResult(finalData);
      setStep(2);
    } catch (err) {
      setError(err.response?.data?.errors || "Unable to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  const resetAssessment = () => {
    setStep(0);
    setClinicalResult(null);
    setImageResult(null);
    setFinalResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <p className="app__eyebrow">Healthcare Risk Assessment</p>
          <h1>Comprehensive Health Evaluation</h1>
        </div>
        <button className="app__reset" onClick={resetAssessment}>
          Start New Assessment
        </button>
      </header>

      <div className="stepper">
        {steps.map((label, index) => (
          <div
            key={label}
            className={`stepper__step ${step >= index ? "stepper__step--active" : ""}`}
          >
            <span className="stepper__index">{index + 1}</span>
            <span>{label}</span>
          </div>
        ))}
      </div>

      {loading && (
        <div className="card">
          <LoadingSpinner />
          <p className="muted">Analyzing data securely...</p>
        </div>
      )}

      {!loading && step === 0 && (
        <ClinicalDataForm onSubmit={handleClinicalSubmit} error={error} />
      )}

      {!loading && step === 1 && (
        <ImageUpload onSubmit={handleImageSubmit} error={error} />
      )}

      {!loading && step === 2 && (
        <ResultsDashboard
          clinicalResult={clinicalResult}
          imageResult={imageResult}
          finalResult={finalResult}
        />
      )}

      <MedicalDisclaimer />
    </div>
  );
};

export default App;
