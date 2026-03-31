"""Preprocessing pipeline for professional architectural line drawings.

Optimized for uncolored floorplans with:
- Grid lines (blue), door swings (blue), site boundaries (red), accent marks (orange)
- Hatched or double-line walls
- Dense interior furniture/fixture detail
- Open doorways creating porous room boundaries

Steps:
1. HSV color filtering — remove non-wall colored elements
2. Adaptive binarization — convert to black/white
3. Wall thickness isolation — erode thin furniture lines, preserve thick walls
4. Aggressive gap closing — bridge doorway openings
"""

import cv2
import numpy as np


def preprocess_linedraw(
    image: np.ndarray,
    filter_colors: bool = True,
    block_size: int = 51,
    erode_px: int = 0,
    dilate_px: int = 1,
    close_gap_px: int = 15,
) -> dict:
    """Preprocess an architectural line drawing for room segmentation.

    Args:
        image: RGB image (numpy array).
        filter_colors: If True, remove blue/red/orange elements before binarization.
        block_size: Adaptive threshold block size.
        erode_px: Erosion kernel size to remove thin furniture lines.
        dilate_px: Dilation kernel size to restore wall thickness after erosion.
        close_gap_px: Morphological closing kernel to bridge doorway gaps.

    Returns:
        dict with keys:
            binary: Cleaned binary image (walls = white, background = black)
            color_mask: The color filter mask applied (for debugging), or None
    """
    working = image.copy()
    color_mask = None

    # --- Step 1: Color filtering ---
    if filter_colors:
        working, color_mask = _filter_colors(working)

    # --- Step 2: Binarization ---
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold — handles varying background brightness
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, 10,
    )

    # --- Step 3: Wall thickness isolation ---
    # Erode to remove thin lines (furniture, dimensions, hatching detail)
    if erode_px > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        binary = cv2.erode(binary, erode_kernel, iterations=1)

    # Dilate back slightly larger to restore and thicken wall lines
    if dilate_px > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        binary = cv2.dilate(binary, dilate_kernel, iterations=1)

    # --- Step 4: Aggressive gap closing ---
    if close_gap_px > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_gap_px, close_gap_px)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # --- Step 5: Remove small noise components ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_noise = 100  # Slightly larger threshold than standard preprocessor
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_noise:
            binary[labels == i] = 0

    return {
        "binary": binary,
        "color_mask": color_mask,
    }


def _filter_colors(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove blue, red, and orange colored elements from the image.

    Replaces colored pixels with white (background) so they don't appear
    as dark lines in the binarized output.

    Returns:
        (filtered_image, color_mask) where color_mask shows what was removed.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Blue: hue 90-130 (in OpenCV's 0-180 scale), moderate+ saturation
    blue_mask = (h >= 90) & (h <= 130) & (s > 40)

    # Red: hue 0-15 or 165-180, moderate+ saturation
    red_mask = ((h <= 15) | (h >= 165)) & (s > 40)

    # Orange: hue 10-25, moderate+ saturation
    orange_mask = (h >= 10) & (h <= 25) & (s > 50)

    # Combined mask of all colored elements to remove
    color_mask = (blue_mask | red_mask | orange_mask).astype(np.uint8) * 255

    # Dilate the mask slightly to catch anti-aliased edges around colored lines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.dilate(color_mask, dilate_kernel, iterations=1)

    # Replace colored pixels with white (background)
    result = image.copy()
    result[color_mask > 0] = [255, 255, 255]

    return result, color_mask
