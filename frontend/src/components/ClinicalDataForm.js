import React, { useState } from "react";

const defaultValues = {
  age: "",
  tumor_size_mm: "",
  bmi: "",
  gender: "",
  family_history: "",
  smoking_history: "",
  alcohol_consumption: "",
  diet_risk: "",
  physical_activity: "",
  diabetes: "",
  ibd: "",
  genetic_mutation: "",
  screening_history: "",
};

const validateField = (name, value) => {
  if (value === "") {
    return "This field is required.";
  }

  // Numeric validation
  if (["age", "tumor_size_mm", "bmi"].includes(name)) {
    const numericValue = Number(value);
    if (Number.isNaN(numericValue)) {
      return "Must be a number.";
    }
    if (name === "age" && (numericValue < 0 || numericValue > 120)) {
      return "Age must be between 0 and 120.";
    }
    if (
      name === "tumor_size_mm" &&
      (numericValue < 0 || numericValue > 200)
    ) {
      return "Tumor size must be between 0 and 200mm.";
    }
    if (name === "bmi" && (numericValue < 10 || numericValue > 60)) {
      return "BMI must be between 10 and 60.";
    }
  }

  // Boolean validation
  if (["family_history", "smoking_history", "diabetes", "ibd", "genetic_mutation", "screening_history"].includes(name)) {
    if (!["true", "false", "yes", "no"].includes(value.toLowerCase())) {
      return "Must be Yes or No.";
    }
  }

  // Enum validation
  const enumValidations = {
    gender: ["male", "female", "other"],
    alcohol_consumption: ["none", "moderate", "high"],
    diet_risk: ["low", "moderate", "high"],
    physical_activity: ["sedentary", "light", "moderate", "vigorous"],
  };

  if (enumValidations[name] && !enumValidations[name].includes(value)) {
    return `Must be one of: ${enumValidations[name].join(", ")}`;
  }

  return "";
};

const ClinicalDataForm = ({ onSubmit, error }) => {
  const [values, setValues] = useState(defaultValues);
  const [errors, setErrors] = useState({});

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues((prev) => ({ ...prev, [name]: value }));
    setErrors((prev) => ({ ...prev, [name]: validateField(name, value) }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const validationErrors = Object.keys(values).reduce((acc, field) => {
      const message = validateField(field, values[field]);
      if (message) {
        acc[field] = message;
      }
      return acc;
    }, {});
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length === 0) {
      // Convert string values to appropriate types for API
      const submitData = {
        age: Number(values.age),
        tumor_size_mm: Number(values.tumor_size_mm),
        bmi: Number(values.bmi),
        gender: values.gender,
        family_history: values.family_history.toLowerCase() === "true" || values.family_history.toLowerCase() === "yes",
        smoking_history: values.smoking_history.toLowerCase() === "true" || values.smoking_history.toLowerCase() === "yes",
        alcohol_consumption: values.alcohol_consumption,
        diet_risk: values.diet_risk,
        physical_activity: values.physical_activity,
        diabetes: values.diabetes.toLowerCase() === "true" || values.diabetes.toLowerCase() === "yes",
        ibd: values.ibd.toLowerCase() === "true" || values.ibd.toLowerCase() === "yes",
        genetic_mutation: values.genetic_mutation.toLowerCase() === "true" || values.genetic_mutation.toLowerCase() === "yes",
        screening_history: values.screening_history.toLowerCase() === "true" || values.screening_history.toLowerCase() === "yes",
      };
      onSubmit(submitData);
    }
  };

  return (
    <div className="card">
      <h2>Clinical Data</h2>
      <p className="muted">
        Provide comprehensive patient clinical data to generate an accurate risk profile.
      </p>

      <form className="form" onSubmit={handleSubmit}>
        <div className="form-section">
          <h3 className="form-section-title">Basic Demographics</h3>
          <div className="form-row">
            <label>
              Age
              <input
                type="number"
                name="age"
                value={values.age}
                onChange={handleChange}
                min="0"
                max="120"
              />
              {errors.age && <span className="error">{errors.age}</span>}
            </label>

            <label>
              Gender
              <select
                name="gender"
                value={values.gender}
                onChange={handleChange}
              >
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
              {errors.gender && <span className="error">{errors.gender}</span>}
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Clinical Measurements</h3>
          <div className="form-row">
            <label>
              Tumor Size (mm)
              <input
                type="number"
                name="tumor_size_mm"
                value={values.tumor_size_mm}
                onChange={handleChange}
                min="0"
                max="200"
              />
              {errors.tumor_size_mm && (
                <span className="error">{errors.tumor_size_mm}</span>
              )}
            </label>

            <label>
              BMI
              <input
                type="number"
                name="bmi"
                value={values.bmi}
                onChange={handleChange}
                min="10"
                max="60"
                step="0.1"
              />
              {errors.bmi && <span className="error">{errors.bmi}</span>}
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Family & Medical History</h3>
          <div className="form-row">
            <label>
              Family History of Cancer
              <select
                name="family_history"
                value={values.family_history}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.family_history && <span className="error">{errors.family_history}</span>}
            </label>

            <label>
              Genetic Mutation
              <select
                name="genetic_mutation"
                value={values.genetic_mutation}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.genetic_mutation && <span className="error">{errors.genetic_mutation}</span>}
            </label>

            <label>
              Screening History
              <select
                name="screening_history"
                value={values.screening_history}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.screening_history && <span className="error">{errors.screening_history}</span>}
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Lifestyle Factors</h3>
          <div className="form-row">
            <label>
              Smoking History
              <select
                name="smoking_history"
                value={values.smoking_history}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.smoking_history && <span className="error">{errors.smoking_history}</span>}
            </label>

            <label>
              Alcohol Consumption
              <select
                name="alcohol_consumption"
                value={values.alcohol_consumption}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="none">None</option>
                <option value="moderate">Moderate</option>
                <option value="high">High</option>
              </select>
              {errors.alcohol_consumption && <span className="error">{errors.alcohol_consumption}</span>}
            </label>
          </div>

          <div className="form-row">
            <label>
              Diet Risk
              <select
                name="diet_risk"
                value={values.diet_risk}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="low">Low Risk</option>
                <option value="moderate">Moderate Risk</option>
                <option value="high">High Risk</option>
              </select>
              {errors.diet_risk && <span className="error">{errors.diet_risk}</span>}
            </label>

            <label>
              Physical Activity
              <select
                name="physical_activity"
                value={values.physical_activity}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="sedentary">Sedentary</option>
                <option value="light">Light Activity</option>
                <option value="moderate">Moderate Activity</option>
                <option value="vigorous">Vigorous Activity</option>
              </select>
              {errors.physical_activity && <span className="error">{errors.physical_activity}</span>}
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Health Conditions</h3>
          <div className="form-row">
            <label>
              Diabetes
              <select
                name="diabetes"
                value={values.diabetes}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.diabetes && <span className="error">{errors.diabetes}</span>}
            </label>

            <label>
              IBD (Inflammatory Bowel Disease)
              <select
                name="ibd"
                value={values.ibd}
                onChange={handleChange}
              >
                <option value="">Select</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
              {errors.ibd && <span className="error">{errors.ibd}</span>}
            </label>
          </div>
        </div>

        {error && (
          <div className="alert">
            {typeof error === "string"
              ? error
              : "Please review the highlighted fields."}
          </div>
        )}

        <button type="submit" className="button">
          Generate Comprehensive Risk Score
        </button>
      </form>
    </div>
  );
};

export default ClinicalDataForm;
