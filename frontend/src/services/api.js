import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || "http://localhost:5000",
  timeout: 10000,
});

export const predictRisk = async (payload) => {
  const response = await apiClient.post("/api/v1/predict-risk", payload);
  return response.data.data;
};

export const classifyImage = async (file) => {
  const formData = new FormData();
  formData.append("image", file);
  const response = await apiClient.post("/api/v1/classify-image", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data.data;
};

export const finalAssessment = async (clinicalResult, imageResult) => {
  const response = await apiClient.post("/api/v1/final-assessment", {
    clinical_result: clinicalResult,
    image_result: imageResult,
  });
  return response.data.data;
};
