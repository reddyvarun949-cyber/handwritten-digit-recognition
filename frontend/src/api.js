// All communication with the Flask backend lives in this one file.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Turns any failed request into an Error with a message a person can read.
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API_URL + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    throw new Error("Cannot reach the API. Is the backend running on " + API_URL + "?");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || "Request failed with status " + response.status);
    error.fields = data.fields || {};
    throw error;
  }
  return data;
}

// Which model is the API using (Keras CNN or the scikit-learn fallback)?
export function getHealth() {
  return request("/api/health");
}

// image: a 2D array (rows) of grayscale numbers 0-255, e.g. 28 x 28.
export function predictDigit(image) {
  return request("/api/predict", { method: "POST", body: JSON.stringify({ image }) });
}
