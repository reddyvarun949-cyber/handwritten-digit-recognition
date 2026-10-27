# Handwritten Digit Recognition

**Methods practised:** neural networks / ANN — layers, ReLU and softmax, cross-entropy loss, overfitting and
regularisation (Batch 2 Module 05); TensorFlow/Keras — `Sequential`, `compile`, `fit`, `EarlyStopping`, a small CNN,
saving `.keras` models (Module 06); plus model saving and a simple API.

## Problem statement

A learning app wants children to draw a digit with the mouse or a finger and see whether the computer can read it.
Given a drawing of **one digit (0–9)**, predict which digit it is and show how confident the model is about every digit.

**Expected result:** neural networks that clearly beat a "guess the most common digit" baseline and a logistic
regression baseline on held-out test images, served through an API and a web page with a drawing canvas. The notebook
also shows honestly how the model does on *drawn* digits, which look different from the training images.
(The build run's actual numbers are in `VERIFICATION.md` and the executed notebook.)

## Folder structure

```
01_Handwritten_Digit_Recognition/
├── data/
│   ├── generate_data.py       # saves scikit-learn's digits dataset to digits.csv
│   ├── digits.csv             # 1,797 images: pixel_0 … pixel_63 (0-16) + label
│   └── DATA_DICTIONARY.md
├── notebook/
│   └── Handwritten_Digit_Recognition.ipynb   # explore → train → evaluate → drawings → save
├── models/                    # created by the notebook, loaded by the API
│   ├── digits_mlp_sklearn.joblib     # scikit-learn MLP (verified fallback model)
│   ├── model_info.json               # input format + test accuracies
│   └── digits_cnn.keras              # only after you run the Keras cells (needs TensorFlow)
├── backend/
│   ├── preprocess.py          # drawing → 8x8 model input (shared by notebook and API)
│   ├── app.py                 # Flask API
│   └── test_app.py            # automated API tests
├── frontend/                  # React + Vite app with a drawing canvas
│   ├── package.json, vite.config.js, index.html, .env.example
│   └── src/ (main.jsx, App.jsx, api.js, App.css)
├── requirements.txt           # everything except TensorFlow
├── requirements-tf.txt        # adds TensorFlow for the Keras cells
└── VERIFICATION.md            # what was actually run and tested
```

## Dataset

`load_digits` from scikit-learn: 1,797 images of handwritten digits from 43 writers, each **8×8 pixels with values
0–16** (0 = no ink). `data/generate_data.py` writes it to `data/digits.csv` so you can open it like any table: one row
per image, 64 pixel columns and the `label` (the target). The ten digits are almost equally common. See
[data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md).

**Optional bigger dataset — MNIST** (70,000 images, 28×28, values 0–255). With TensorFlow installed:

```python
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()   # downloads about 11 MB once
```

It is not used in this project, so everything runs offline. See "Ideas to extend".

## How to run (three terminals)

**1. Python environment and notebook** (from this project folder):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt            # or: pip install -r requirements-tf.txt  (adds TensorFlow, Python 3.10-3.12)
python data/generate_data.py               # creates data/digits.csv (already included)
jupyter notebook notebook/Handwritten_Digit_Recognition.ipynb     # Run All: this creates models/
```

Without TensorFlow, the cells starting with `# NO-RUN` will fail with `ModuleNotFoundError` — skip them; the rest of
the notebook (and the app) works with the scikit-learn model. With TensorFlow, run them too: they save
`models/digits_cnn.keras`, and the API will use the CNN automatically.

The scikit-learn model is already included in `models/`, so you can skip straight to step 2 if you only want the app.

**2. Backend API** (same environment, from this project folder):

```bash
python backend/app.py                 # http://localhost:5000  (prints which model it loaded)
python -m unittest discover -s backend -p "test_*.py" -v    # optional: run the tests
```

Try it without the frontend:

```bash
curl http://localhost:5000/api/health
# 64 pixel values (0-16): the first "0" from the dataset
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" \
  -d '{"pixels":[0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}'
# a tiny 8x8 "drawing" (values 0-255): a vertical bar = a 1
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" \
  -d '{"image":[[0,0,0,0,0,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,255,255,0,0,0],[0,0,0,0,0,0,0,0]]}'
```

To force the scikit-learn model even when the CNN exists: `DIGITS_MODEL=sklearn python backend/app.py`
(Windows PowerShell: `$env:DIGITS_MODEL="sklearn"; python backend/app.py`).

**3. Frontend** (needs Node.js 18+):

```bash
cd frontend
npm install
npm run dev                           # open http://localhost:5173
```

Draw a digit with the mouse (or a finger on a touch screen), press **Predict**, and press **Clear** to start again.
If the API runs somewhere else, copy `.env.example` to `.env` and change `VITE_API_URL`. If the frontend runs on a
different address, start the backend with `FRONTEND_ORIGIN=http://that-address python backend/app.py`.

## Architecture and request flow

```
 Browser (React, frontend/src/App.jsx)          Flask API (backend/app.py)                       models/
 ─────────────────────────────────────          ──────────────────────────                       ───────
 page opens ──────── GET /api/health ────────▶  report the model loaded at start-up
            ◀──── {model, using_fallback} ────     (Keras CNN if digits_cnn.keras + TensorFlow,  ◀── digits_cnn.keras
                                                    else scikit-learn MLP)                       ◀── digits_mlp_sklearn.joblib
 user draws on a 280x280 black canvas
 (white pen, 22 px wide)
 Predict: offscreen canvas shrinks it to 28x28
          → 2D array of grey values 0-255
            ──── POST /api/predict {"image": [[...28 rows...]]} ──▶
                                                1. parse JSON (400 if not JSON)
                                                2. validate shape/values (422 + message per field)
                                                3. preprocess.image_to_8x8: invert if needed, crop to
                                                   the ink, pad to square, average to 8x8, scale 0-16
                                                4. to_model_input: /16 → model.predict_proba
            ◀── 200 {digit, confidence, probabilities[10], model, model_input_8x8} ──
 shows the digit, 10 probability bars (plain divs) and the 8x8 image the model saw
```

- The models are **trained once in the notebook** and loaded **once when the API starts**.
- `backend/preprocess.py` is imported by **both** the notebook and the API, so a drawing is converted in exactly the
  same way in both places. The notebook (section 8) shows why this step matters: without cropping and centring, most
  drawings are misread.
- TensorFlow is imported **only** inside the API's model loader, so the API also starts on computers without it.

### API reference

| Method | Path | Request | Success response | Errors |
|---|---|---|---|---|
| GET | `/api/health` | — | `{"status": "ok", "model": "...", "using_fallback": true/false, "test_accuracy": {...}}` | — |
| POST | `/api/predict` | `{"image": [[...], ...]}` — 2D grayscale array, 8×8 up to 400×400, values 0–255 (light digit on dark, or dark on light: it is inverted automatically) **or** `{"pixels": [64 numbers 0–16]}` — an image already in the dataset format | `{"digit": 7, "confidence": 0.93, "probabilities": [10 numbers, index = digit], "model": "...", "input_type": "image"/"pixels", "model_input_8x8": [[8 rows of 0–16]]}` | 400 body not JSON; 422 `{"error", "fields": {"image" or "pixels" or "body": message}}` |

Validation rules: send exactly one of `image` or `pixels`; `pixels` must be 64 numbers between 0 and 16; `image` must be
a rectangular 2D list (all rows the same length), at least 8×8 and at most 400×400, with numbers between 0 and 255,
and it must contain some ink (an all-black drawing is rejected as empty).

## Ideas to extend

- Add a **"not sure"** message when the top probability is below, say, 50%, and test it on scribbles and letters.
- **Collect drawings** from the app (save the 28×28 array and the correct digit), then measure accuracy on real
  drawings and fine-tune the model on them.
- Train the CNN on **MNIST** (28×28) and change `preprocess.py` to produce 28×28 inputs (MNIST digits are centred in a
  20×20 box inside the 28×28 image). Add data augmentation with `keras.layers.RandomRotation` and
  `RandomTranslation`.
- Show the **second-best guess** in the frontend, or let the user draw several digits and split them into separate
  images.
