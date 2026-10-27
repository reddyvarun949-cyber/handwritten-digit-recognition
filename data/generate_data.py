"""Save scikit-learn's handwritten digits dataset as a plain CSV file.

Run from the project root:  python data/generate_data.py
The output is data/digits.csv (1,797 rows, 64 pixel columns + label).

The data comes bundled with scikit-learn (no download needed). It is the "UCI ML hand-written digits"
dataset: 43 people wrote digits on paper, the scans were cut into 32x32 black-and-white images, and each
image was shrunk to 8x8 by counting the ink pixels in every 4x4 block. So every pixel is a whole number
from 0 (no ink) to 16 (all 16 small pixels were ink).

Writing it to CSV lets students open the data in Excel or pandas and see that an image is "just numbers".
Nothing random happens here, so everyone gets exactly the same file.
"""
import os

import pandas as pd
from sklearn.datasets import load_digits


def make_dataset():
    digits = load_digits()
    # pixel_0 is the top-left pixel, pixel_7 the top-right, pixel_63 the bottom-right (row by row)
    df = pd.DataFrame(digits.data.astype(int), columns=[f"pixel_{i}" for i in range(64)])
    df["label"] = digits.target
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digits.csv")
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows and {data.shape[1]} columns to {out}")
    print("Images per digit:", data["label"].value_counts().sort_index().to_dict())
