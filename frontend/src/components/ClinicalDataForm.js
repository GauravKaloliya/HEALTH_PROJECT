import React, { useState } from "react";

const defaultValues = {
  age: "",
  tumor_size_mm: "",
  bmi: "",
};

const validateField = (name, value) => {
  if (value === "") {
    return "This field is required.";
  }
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
      onSubmit({
        age: Number(values.age),
        tumor_size_mm: Number(values.tumor_size_mm),
        bmi: Number(values.bmi),
      });
    }
  };

  return (
    <div className="card">
      <h2>Clinical Data</h2>
      <p className="muted">
        Provide the patient clinical metrics to generate an initial risk profile.
      </p>

      <form className="form" onSubmit={handleSubmit}>
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

        {error && (
          <div className="alert">
            {typeof error === "string"
              ? error
              : "Please review the highlighted fields."}
          </div>
        )}

        <button type="submit" className="button">
          Generate Risk Score
        </button>
      </form>
    </div>
  );
};

export default ClinicalDataForm;
