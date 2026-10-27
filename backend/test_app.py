"""Automated tests for the API. Run from the project root:

    python -m unittest discover -s backend -p "test_*.py" -v

They use Flask's built-in test client, so no server needs to be running.
They force the scikit-learn fallback model (DIGITS_MODEL=sklearn), so they run without TensorFlow.
"""
import os
import sys
import unittest

import numpy as np

os.environ["DIGITS_MODEL"] = "sklearn"          # must be set before app.py is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app  # noqa: E402
from preprocess import image_to_8x8  # noqa: E402

# The first "0" of the dataset (row 0 of data/digits.csv), as 64 values 0-16
ZERO_PIXELS = [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
               0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]


def ring_drawing(size=28):
    """A thick white ring on black, like a '0' drawn on the canvas and shrunk to 28x28 (values 0-255)."""
    rows, cols = np.mgrid[0:size, 0:size]
    distance = np.hypot((rows - size / 2) / 0.55, (cols - size / 2) / 0.38)   # taller than wide
    return np.where(np.abs(distance - size * 0.75) < 3.5, 255, 0).tolist()


def line_drawing(size=28):
    """A vertical white bar: the simplest '1'."""
    image = np.zeros((size, size))
    image[4:24, 12:16] = 255
    return image.tolist()


class DigitApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post("/api/predict", json=payload)

    # ---- health ----
    def test_health_reports_fallback_model(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["using_fallback"])
        self.assertIn("MLPClassifier", body["model"])

    # ---- happy paths: both input formats ----
    def test_pixels_input(self):
        r = self.post({"pixels": ZERO_PIXELS})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["digit"], 0)
        self.assertEqual(len(body["probabilities"]), 10)
        self.assertAlmostEqual(sum(body["probabilities"]), 1.0, places=2)
        self.assertEqual(body["input_type"], "pixels")
        self.assertEqual(np.array(body["model_input_8x8"]).shape, (8, 8))
        print("\n  pixels prediction:", {k: body[k] for k in ("digit", "confidence", "model")})

    def test_image_input_ring_is_zero(self):
        r = self.post({"image": ring_drawing()})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["digit"], 0)
        self.assertEqual(body["input_type"], "image")
        print("\n  drawn ring prediction:", {k: body[k] for k in ("digit", "confidence")})

    def test_image_input_line_is_one(self):
        body = self.post({"image": line_drawing()}).get_json()
        self.assertEqual(body["digit"], 1)

    def test_large_canvas_and_inverted_colours(self):
        big = np.kron(np.array(ring_drawing()), np.ones((10, 10)))     # 280x280, like the full canvas
        r = self.post({"image": (255 - big).tolist()})                  # dark digit on white
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["digit"], 0)

    def test_dataset_image_upscaled_round_trip(self):
        # A training image blown up to 32x32 (values 0-255) should survive preprocessing and be recognised
        small = np.array(ZERO_PIXELS).reshape(8, 8)
        big = np.kron(small * 255 / 16, np.ones((4, 4)))
        self.assertEqual(image_to_8x8(big).shape, (8, 8))
        self.assertEqual(self.post({"image": big.tolist()}).get_json()["digit"], 0)

    # ---- validation errors (422 with a message per field) ----
    def test_neither_field(self):
        r = self.post({"something": 1})
        self.assertEqual(r.status_code, 422)
        self.assertIn("image", r.get_json()["fields"])

    def test_both_fields(self):
        r = self.post({"pixels": ZERO_PIXELS, "image": line_drawing()})
        self.assertEqual(r.status_code, 422)

    def test_pixels_wrong_length(self):
        r = self.post({"pixels": [0] * 10})
        self.assertEqual(r.status_code, 422)
        self.assertIn("exactly 64", r.get_json()["fields"]["pixels"])

    def test_pixels_out_of_range(self):
        r = self.post({"pixels": [20] * 64})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between 0 and 16", r.get_json()["fields"]["pixels"])

    def test_pixels_not_numbers(self):
        r = self.post({"pixels": ["a"] * 64})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["pixels"], "Every value must be a number.")

    def test_pixels_not_a_list(self):
        r = self.post({"pixels": "0,0,5"})
        self.assertEqual(r.status_code, 422)

    def test_image_not_2d(self):
        r = self.post({"image": [0, 255, 0]})
        self.assertEqual(r.status_code, 422)
        self.assertIn("2D list", r.get_json()["fields"]["image"])

    def test_image_ragged_rows(self):
        image = line_drawing()
        image[3] = image[3][:10]
        r = self.post({"image": image})
        self.assertEqual(r.status_code, 422)
        self.assertIn("same length", r.get_json()["fields"]["image"])

    def test_image_too_large(self):
        r = self.post({"image": [[0] * 401] * 401})
        self.assertEqual(r.status_code, 422)
        self.assertIn("at most 400x400", r.get_json()["fields"]["image"])

    def test_image_too_small(self):
        r = self.post({"image": [[0, 255], [255, 0]]})
        self.assertEqual(r.status_code, 422)
        self.assertIn("at least 8x8", r.get_json()["fields"]["image"])

    def test_image_values_out_of_range(self):
        image = line_drawing()
        image[10][13] = 300
        r = self.post({"image": image})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between 0 and 255", r.get_json()["fields"]["image"])

    def test_empty_drawing(self):
        r = self.post({"image": [[0] * 28] * 28})
        self.assertEqual(r.status_code, 422)
        self.assertIn("empty", r.get_json()["fields"]["image"])

    def test_body_is_a_list(self):
        r = self.post([1, 2, 3])
        self.assertEqual(r.status_code, 422)
        self.assertIn("body", r.get_json()["fields"])

    # ---- protocol ----
    def test_body_not_json(self):
        r = self.client.post("/api/predict", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_cors_headers_present(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        pre = self.client.options("/api/predict")
        self.assertEqual(pre.status_code, 204)


if __name__ == "__main__":
    unittest.main()
