import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 1800000,
  headers: {
    "ngrok-skip-browser-warning": "true",
  },
});


export async function analyzeImage({
  imageFile,
  mode = "quick",
  selectedModels = [],
  selectedMethods = [],
  targetMode = "predicted",
}) {
  const formData = new FormData();

  formData.append(
    "image",
    imageFile
  );

  formData.append(
    "mode",
    mode
  );

  formData.append(
    "selected_models",
    JSON.stringify(
      selectedModels
    )
  );

  formData.append(
    "selected_methods",
    JSON.stringify(
      selectedMethods
    )
  );

  formData.append(
    "target_mode",
    targetMode
  );

  const response = await api.post(
    "/api/analyze/custom",
    formData,
    {
      headers: {
        "Content-Type":
          "multipart/form-data",
      },
    }
  );

  return response.data;
}


export async function quickAnalyzeImage(
  imageFile
) {
  return analyzeImage({
    imageFile,
    mode: "quick",
    targetMode: "predicted",
  });
}


export async function fullAnalyzeImage(
  imageFile
) {
  return analyzeImage({
    imageFile,
    mode: "full",
    targetMode: "predicted",
  });
}


export async function getHealthStatus() {
  const response = await api.get(
    "/api/health"
  );

  return response.data;
}


export function buildStaticUrl(
  relativeUrl
) {
  if (!relativeUrl) {
    return "";
  }

  return (
    `${API_BASE_URL}${relativeUrl}`
  );
}