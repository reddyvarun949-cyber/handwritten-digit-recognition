"""Turn a drawing into the 8x8 format the digit models were trained on.

This file is shared by the notebook (to show the conversion step by step) and by the Flask API
(to convert every drawing it receives), so both always use exactly the same steps.

The training images (scikit-learn's load_digits) are 8x8 pixels with values from 0 (background)
to 16 (full ink), and the digit fills most of the 8 rows. A drawing from the browser is different:
a large image (for example 28x28 or 280x280) with values 0-255, and the digit can be small or
off-centre. image_to_8x8() closes that gap:

    1. make sure the digit is light on a dark background (invert if the background is bright)
    2. find the box around the ink and cut it out (removes empty space, centres the digit)
    3. pad the box to a square so the digit is not stretched
    4. shrink to 8x8 by averaging the pixels that fall into each of the 64 cells
    5. rescale so the darkest ink cell becomes 16, like the training data

Only NumPy is needed, so this works without Pillow or OpenCV.
"""
import numpy as np

IMAGE_SIZE = 8       # the models expect 8 x 8 images
PIXEL_MAX = 16       # training pixel values go from 0 to 16
INK_THRESHOLD = 0.2  # a pixel brighter than 20% of the maximum counts as "ink" when finding the box
MARGIN = 0.0         # extra empty border around the digit (share of box size); the training digits touch the edges


def image_to_8x8(image):
    """Convert a 2D grayscale drawing (any size, values 0-255) to an 8x8 array with values 0-16.

    Raises ValueError if the image contains no ink (an empty drawing).
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("The image must be a 2D array (rows of pixel values).")

    # 1. Light digit on dark background. If the border is mostly bright, the user drew dark on light: invert.
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if border.mean() > 127:
        img = 255 - img

    if img.max() <= 0:
        raise ValueError("The drawing is empty. Draw a digit first.")
    img = img / img.max()                       # now 0 = background, 1 = strongest ink

    # 2. Bounding box of the ink
    rows = np.where((img > INK_THRESHOLD).any(axis=1))[0]
    cols = np.where((img > INK_THRESHOLD).any(axis=0))[0]
    if len(rows) == 0:
        raise ValueError("The drawing is empty. Draw a digit first.")
    crop = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 3. Pad to a square (plus a small margin), keeping the digit in the centre
    h, w = crop.shape
    side = int(np.ceil(max(h, w) * (1 + 2 * MARGIN)))
    side = max(side, IMAGE_SIZE)                # never smaller than the 8x8 output
    square = np.zeros((side, side))
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = crop

    # 4. Shrink to 8x8: each output cell is the average of the pixels it covers
    edges = np.linspace(0, side, IMAGE_SIZE + 1).round().astype(int)
    small = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            block = square[edges[i]:max(edges[i + 1], edges[i] + 1), edges[j]:max(edges[j + 1], edges[j] + 1)]
            small[i, j] = block.mean()

    # 5. Rescale to 0-16 whole numbers like the training images
    small = small / small.max() * PIXEL_MAX
    return np.rint(small).astype(int)


def to_model_input(pixels):
    """Scale pixel values (0-16) to 0-1 and shape them as rows of 64 numbers: (n_images, 64).

    Works for one image (64 numbers or an 8x8 array) and for many images (n x 64).
    The models were trained on exactly this scaling, so the API and the notebook must both use it.
    """
    flat = np.asarray(pixels, dtype=float).reshape(-1, IMAGE_SIZE * IMAGE_SIZE)
    return flat / PIXEL_MAX
