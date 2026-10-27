# Data dictionary — digits.csv

Scikit-learn's bundled copy of the **UCI ML hand-written digits** dataset, written to CSV by `generate_data.py`
(no download, no randomness). 1,797 images of the digits 0–9 written by 43 people. Licence: UCI dataset distributed
with scikit-learn (CC BY 4.0); free to use for teaching.

| Column | Type | Description | Values / range | Missing |
|---|---|---|---|---|
| `pixel_0` … `pixel_63` | integer | Amount of ink in one pixel of the 8×8 image, stored **row by row**: `pixel_0`–`pixel_7` is the top row (left to right), `pixel_8`–`pixel_15` the second row, …, `pixel_63` the bottom-right pixel. Pixel at row `r`, column `c` is `pixel_{8*r + c}`. | 0 (no ink) – 16 (full ink) | none |
| **`label`** | integer | **Target.** The digit the person wrote. | 0–9, between 174 and 183 images each | none |

## How the pixel values were made

Each digit was written on a form, scanned and cut out as a 32×32 black-and-white image. That image was divided into
64 blocks of 4×4 pixels, and each block's value is the **number of black pixels in it** (0–16). This reduces the
image to 8×8 and makes it less sensitive to small shifts.

## Things to know

- The digits fill most of the 8×8 box and are centred. Drawings from the web app are cropped and centred by
  `backend/preprocess.py` so they look similar before they reach the model.
- There are no missing values or duplicates; the data is much cleaner than most real image data.
- Columns `pixel_0`, `pixel_8`, `pixel_16`, … (the left edge) and `pixel_7`, `pixel_15`, … (the right edge) are almost
  always 0, because the writing seldom touches the sides.
- To show an image: `row[pixel_cols].to_numpy().reshape(8, 8)` and `plt.imshow(..., cmap="gray")`.

## Optional bigger dataset: MNIST

MNIST has 70,000 digits of 28×28 pixels (values 0–255). It is not included here; download it with
`keras.datasets.mnist.load_data()` (see the README, "Ideas to extend").
