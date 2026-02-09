import React from "react";

const RiskBadge = ({ label }) => {
  const className = `badge badge--${label}`;
  return <span className={className}>{label.toUpperCase()}</span>;
};

const ResultsDashboard = ({ clinicalResult, imageResult, finalResult }) => {
  return (
    <div className="card">
      <h2>Assessment Results</h2>
      <p className="muted">
        The following insights combine clinical metrics with imaging analysis.
      </p>

      <div className="results-grid">
        <div className="result-panel">
          <h3>Clinical Risk</h3>
          <RiskBadge label={clinicalResult?.risk_level || "low"} />
          <p className="result-score">Score: {clinicalResult?.risk_score}</p>
          <p className="muted">
            Confidence: {Math.round((clinicalResult?.confidence || 0) * 100)}%
          </p>
        </div>

        <div className="result-panel">
          <h3>Imaging Classification</h3>
          <p className="result-score">{imageResult?.label}</p>
          <p className="muted">
            Confidence: {Math.round((imageResult?.confidence || 0) * 100)}%
          </p>
        </div>

        <div className="result-panel result-panel--highlight">
          <h3>Final Assessment</h3>
          <RiskBadge label={finalResult?.final_risk_level || "low"} />
          <p className="result-score">Score: {finalResult?.final_score}</p>
          <p className="muted">{finalResult?.summary}</p>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;
