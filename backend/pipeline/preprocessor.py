"""Pre-process floorplan images for wall detection."""
import cv2
import numpy as np

def preprocess_image(image: np.ndarray, block_size: int = 51, closing_kernel_size: int = 3) -> dict:
    if image.size == 0:
        raise ValueError("Empty image provided")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_component_area = 50
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            binary[labels == i] = 0
    return {"gray": gray, "binary": binary}
