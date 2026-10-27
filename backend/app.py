"""Flask API for the Handwritten Digit Recognition project.

Run from the project root:
    python backend/app.py
The API starts on http://localhost:5000

Endpoints
    GET  /api/health    -> is the API up, and which model is it using (Keras CNN or scikit-learn MLP)?
    POST /api/predict   -> a drawing (or 64 pixel values) -> predicted digit + probability for every digit

Which model is used?
    1. models/digits_cnn.keras  if that file exists AND TensorFlow is installed (created by the Keras cells of the notebook)
    2. otherwise models/digits_mlp_sklearn.joblib  (the scikit-learn MLP, always created by the notebook)
Set DIGITS_MODEL=sklearn to force the scikit-learn model.

The models are loaded ONCE when the server starts. We never retrain on a request.
"""
import json
import math
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

from preprocess import IMAGE_SIZE, PIXEL_MAX, image_to_8x8, to_model_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SKLEARN_PATH = os.path.join(MODELS_DIR, "digits_mlp_sklearn.joblib")
KERAS_PATH = os.path.join(MODELS_DIR, "digits_cnn.keras")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

MAX_IMAGE_SIDE = 400  # biggest drawing we accept (rows and columns)

# Which website may call this API from a browser (the Vite dev server by default).
ALLOWED_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

app = Flask(__name__)


# ---- load a model once at start-up ----
def load_model():
    """Return (predict_proba_function, model_name). Prefer the Keras CNN, fall back to the scikit-learn MLP."""
    if os.environ.get("DIGITS_MODEL", "auto") != "sklearn":
        if not os.path.exists(KERAS_PATH):
            print(f"[startup] {os.path.relpath(KERAS_PATH, BASE_DIR)} not found "
                  "(run the Keras cells of the notebook to create it). Using the scikit-learn model.")
        else:
            try:
                from tensorflow import keras  # imported only here: TensorFlow is optional
                cnn = keras.models.load_model(KERAS_PATH)
                print("[startup] Using the Keras CNN from models/digits_cnn.keras")

                def keras_proba(rows):
                    images = rows.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)   # the CNN wants (n, 8, 8, 1)
                    return cnn.predict(images, verbose=0)
                return keras_proba, "Keras CNN (TensorFlow)"
            except ImportError:
                print("[startup] models/digits_cnn.keras exists but TensorFlow is not installed. "
                      "Using the scikit-learn model.")

    if not os.path.exists(SKLEARN_PATH):
        raise SystemExit("Model files not found. Run notebook/Handwritten_Digit_Recognition.ipynb first to create models/.")
    mlp = joblib.load(SKLEARN_PATH)
    print("[startup] Using the scikit-learn MLP from models/digits_mlp_sklearn.joblib")
    return mlp.predict_proba, "MLPClassifier (scikit-learn)"


predict_proba, MODEL_NAME = load_model()
info = {}
if os.path.exists(INFO_PATH):
    with open(INFO_PATH) as f:
        info = json.load(f)


@app.after_request
def add_cors_headers(response):
    """Allow the React app (a different port = a different 'origin') to call this API."""
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def is_number(value):
    """True for real numbers (not text, not True/False, not NaN or infinity)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and not math.isnan(value) and not math.isinf(value))


def validate_pixels(pixels):
    """Check a list of 64 values 0-16. Return (8x8 array, None) or (None, error message)."""
    if not isinstance(pixels, list):
        return None, "Must be a list of 64 numbers."
    if len(pixels) != IMAGE_SIZE * IMAGE_SIZE:
        return None, f"Must contain exactly 64 numbers (an 8x8 image), got {len(pixels)}."
    if not all(is_number(p) for p in pixels):
        return None, "Every value must be a number."
    if not all(0 <= p <= PIXEL_MAX for p in pixels):
        return None, "Every value must be between 0 and 16."
    return np.array(pixels, dtype=float).reshape(IMAGE_SIZE, IMAGE_SIZE), None


def validate_image(image):
    """Check a 2D grayscale image (list of equal-length rows, values 0-255). Return (8x8 array, None) or (None, error)."""
    if not isinstance(image, list) or len(image) == 0 or not all(isinstance(row, list) for row in image):
        return None, "Must be a 2D list of rows, e.g. [[0, 0, 255], [0, 255, 0], ...]."
    height, width = len(image), len(image[0])
    if any(len(row) != width for row in image):
        return None, "All rows must have the same length."
    if height < IMAGE_SIZE or width < IMAGE_SIZE:
        return None, f"Image must be at least {IMAGE_SIZE}x{IMAGE_SIZE} pixels, got {height}x{width}."
    if height > MAX_IMAGE_SIDE or width > MAX_IMAGE_SIDE:
        return None, f"Image must be at most {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE} pixels, got {height}x{width}."
    if not all(is_number(p) for row in image for p in row):
        return None, "Every pixel must be a number."
    array = np.array(image, dtype=float)
    if array.min() < 0 or array.max() > 255:
        return None, "Pixel values must be between 0 and 255."
    try:
        return image_to_8x8(array), None          # crop, centre and shrink exactly like the notebook
    except ValueError as error:                    # e.g. an empty (all black) drawing
        return None, str(error)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "using_fallback": MODEL_NAME.startswith("MLPClassifier"),
        "test_accuracy": info.get("test_accuracy", {}),
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":           # browser "preflight" check before a POST
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)."}), 400
    if not isinstance(payload, dict):
        return jsonify({"error": "Some inputs are invalid.",
                        "fields": {"body": "Send a JSON object with either 'image' or 'pixels'."}}), 422

    has_image, has_pixels = "image" in payload, "pixels" in payload
    if has_image == has_pixels:               # both or neither
        message = "Send exactly one of 'image' (2D drawing, 0-255) or 'pixels' (64 numbers, 0-16)."
        return jsonify({"error": "Some inputs are invalid.", "fields": {"image": message, "pixels": message}}), 422

    field = "image" if has_image else "pixels"
    small, error = validate_image(payload["image"]) if has_image else validate_pixels(payload["pixels"])
    if error:
        return jsonify({"error": "Some inputs are invalid.", "fields": {field: error}}), 422

    probabilities = predict_proba(to_model_input(small))[0]
    digit = int(np.argmax(probabilities))
    return jsonify({
        "digit": digit,
        "confidence": round(float(probabilities[digit]), 4),
        "probabilities": [round(float(p), 4) for p in probabilities],   # index = digit 0..9
        "model": MODEL_NAME,
        "input_type": field,
        "model_input_8x8": np.rint(small).astype(int).tolist(),          # what the model actually saw
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
