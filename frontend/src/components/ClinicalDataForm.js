import React, { useState } from "react";

const defaultValues = {
  age: "",
  gender: "",
  tumor_size_mm: "",
  family_history: "",
  smoking_history: "",
  alcohol_consumption: "",
  bmi: "",
  diet_risk: "",
  physical_activity: "",
  diabetes: "",
  inflammatory_bowel_disease: "",
  genetic_mutation: "",
  screening_history: "",
};

const ClinicalDataForm = ({ onSubmit, error }) => {
  const [values, setValues] = useState(defaultValues);
  const [errors, setErrors] = useState({});

  const handleChange = (event) => {
    const { name, value, type } = event.target;
    let processedValue = value;
    
    // Convert radio button values to boolean
    if (type === "radio" && (value === "true" || value === "false")) {
      processedValue = value === "true";
    }
    
    setValues((prev) => ({ ...prev, [name]: processedValue }));
    setErrors((prev) => ({ ...prev, [name]: "" }));
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Required fields
    const requiredFields = Object.keys(defaultValues);
    requiredFields.forEach((field) => {
      if (values[field] === "" || values[field] === null || values[field] === undefined) {
        newErrors[field] = "This field is required.";
      }
    });

    // Numeric validations
    if (values.age) {
      const age = Number(values.age);
      if (isNaN(age) || age < 0 || age > 120) {
        newErrors.age = "Age must be between 0 and 120.";
      }
    }

    if (values.tumor_size_mm) {
      const size = Number(values.tumor_size_mm);
      if (isNaN(size) || size < 0 || size > 200) {
        newErrors.tumor_size_mm = "Tumor size must be between 0 and 200mm.";
      }
    }

    if (values.bmi) {
      const bmi = Number(values.bmi);
      if (isNaN(bmi) || bmi < 10 || bmi > 60) {
        newErrors.bmi = "BMI must be between 10 and 60.";
      }
    }

    return newErrors;
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const validationErrors = validateForm();
    
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    // Prepare payload
    const payload = {
      age: Number(values.age),
      gender: values.gender,
      tumor_size_mm: Number(values.tumor_size_mm),
      family_history: values.family_history === true || values.family_history === "true",
      smoking_history: values.smoking_history === true || values.smoking_history === "true",
      alcohol_consumption: values.alcohol_consumption,
      bmi: Number(values.bmi),
      diet_risk: values.diet_risk,
      physical_activity: values.physical_activity,
      diabetes: values.diabetes === true || values.diabetes === "true",
      inflammatory_bowel_disease: values.inflammatory_bowel_disease === true || values.inflammatory_bowel_disease === "true",
      genetic_mutation: values.genetic_mutation === true || values.genetic_mutation === "true",
      screening_history: values.screening_history === true || values.screening_history === "true",
    };

    onSubmit(payload);
  };

  const RadioGroup = ({ name, label, value, error }) => (
    <div className="form-field">
      <label className="field-label">{label}</label>
      <div className="radio-group">
        <label className="radio-label">
          <input
            type="radio"
            name={name}
            value="true"
            checked={value === true || value === "true"}
            onChange={handleChange}
          />
          <span>Yes</span>
        </label>
        <label className="radio-label">
          <input
            type="radio"
            name={name}
            value="false"
            checked={value === false || value === "false"}
            onChange={handleChange}
          />
          <span>No</span>
        </label>
      </div>
      {error && <span className="error">{error}</span>}
    </div>
  );

  const SelectField = ({ name, label, value, options, error }) => (
    <div className="form-field">
      <label className="field-label" htmlFor={name}>{label}</label>
      <select
        id={name}
        name={name}
        value={value}
        onChange={handleChange}
        className={error ? "error-input" : ""}
      >
        <option value="">Select...</option>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {error && <span className="error">{error}</span>}
    </div>
  );

  const NumberField = ({ name, label, value, min, max, step, error }) => (
    <div className="form-field">
      <label className="field-label" htmlFor={name}>{label}</label>
      <input
        id={name}
        type="number"
        name={name}
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        step={step}
        className={error ? "error-input" : ""}
      />
      {error && <span className="error">{error}</span>}
    </div>
  );

  return (
    <div className="card">
      <h2>Clinical Data</h2>
      <p className="muted">
        Provide comprehensive patient clinical metrics to generate an accurate risk profile.
      </p>

      <form className="clinical-form" onSubmit={handleSubmit}>
        {/* Demographics Section */}
        <div className="form-section">
          <h3>Demographics</h3>
          <div className="form-grid">
            <NumberField
              name="age"
              label="Age"
              value={values.age}
              min="0"
              max="120"
              error={errors.age}
            />
            <SelectField
              name="gender"
              label="Gender"
              value={values.gender}
              options={[
                { value: "male", label: "Male" },
                { value: "female", label: "Female" },
                { value: "other", label: "Other" },
              ]}
              error={errors.gender}
            />
          </div>
        </div>

        {/* Tumor Details Section */}
        <div className="form-section">
          <h3>Tumor Details</h3>
          <div className="form-grid">
            <NumberField
              name="tumor_size_mm"
              label="Tumor Size (mm)"
              value={values.tumor_size_mm}
              min="0"
              max="200"
              error={errors.tumor_size_mm}
            />
          </div>
        </div>

        {/* Lifestyle Factors Section */}
        <div className="form-section">
          <h3>Lifestyle Factors</h3>
          <div className="form-grid">
            <NumberField
              name="bmi"
              label="BMI"
              value={values.bmi}
              min="10"
              max="60"
              step="0.1"
              error={errors.bmi}
            />
            <SelectField
              name="alcohol_consumption"
              label="Alcohol Consumption"
              value={values.alcohol_consumption}
              options={[
                { value: "low", label: "Low" },
                { value: "medium", label: "Medium" },
                { value: "high", label: "High" },
              ]}
              error={errors.alcohol_consumption}
            />
            <SelectField
              name="diet_risk"
              label="Diet Risk"
              value={values.diet_risk}
              options={[
                { value: "low", label: "Low" },
                { value: "medium", label: "Medium" },
                { value: "high", label: "High" },
              ]}
              error={errors.diet_risk}
            />
            <SelectField
              name="physical_activity"
              label="Physical Activity"
              value={values.physical_activity}
              options={[
                { value: "low", label: "Low" },
                { value: "medium", label: "Medium" },
                { value: "high", label: "High" },
              ]}
              error={errors.physical_activity}
            />
          </div>
        </div>

        {/* Medical History Section */}
        <div className="form-section">
          <h3>Medical History</h3>
          <div className="radio-grid">
            <RadioGroup
              name="family_history"
              label="Family History"
              value={values.family_history}
              error={errors.family_history}
            />
            <RadioGroup
              name="smoking_history"
              label="Smoking History"
              value={values.smoking_history}
              error={errors.smoking_history}
            />
            <RadioGroup
              name="diabetes"
              label="Diabetes"
              value={values.diabetes}
              error={errors.diabetes}
            />
            <RadioGroup
              name="inflammatory_bowel_disease"
              label="Inflammatory Bowel Disease"
              value={values.inflammatory_bowel_disease}
              error={errors.inflammatory_bowel_disease}
            />
            <RadioGroup
              name="genetic_mutation"
              label="Genetic Mutation"
              value={values.genetic_mutation}
              error={errors.genetic_mutation}
            />
            <RadioGroup
              name="screening_history"
              label="Screening History"
              value={values.screening_history}
              error={errors.screening_history}
            />
          </div>
        </div>

        {error && (
          <div className="alert">
            {typeof error === "string"
              ? error
              : "Please review the highlighted fields."}
          </div>
        )}

        <button type="submit" className="button submit-button">
          Generate Risk Score
        </button>
      </form>
    </div>
  );
};

export default ClinicalDataForm;
