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


def detect_margin_regions(
    binary: np.ndarray,
    margin_fraction: float = 0.15,
    density_threshold: float = 0.15,
    strip_width_px: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Detect dense grid/table regions in the margins of a floorplan.

    Scans strips along each edge. If a strip has ink density above the
    threshold, the margin extends inward from that edge. Returns a list
    of (x, y, w, h) bounding boxes to exclude.
    """
    h, w = binary.shape
    max_margin_h = int(h * margin_fraction)
    max_margin_w = int(w * margin_fraction)
    regions: list[tuple[int, int, int, int]] = []

    def _strip_density(img: np.ndarray) -> float:
        return float(np.count_nonzero(img)) / max(img.size, 1)

    # Right margin — scan from right edge inward
    right_x = w
    for x in range(w - strip_width_px, w - max_margin_w, -strip_width_px):
        strip = binary[:, max(x, 0) : x + strip_width_px]
        if _strip_density(strip) >= density_threshold:
            right_x = max(x, 0)
        else:
            break
    if right_x < w:
        regions.append((right_x, 0, w - right_x, h))

    # Bottom margin — scan from bottom edge upward
    bottom_y = h
    for y in range(h - strip_width_px, h - max_margin_h, -strip_width_px):
        strip = binary[max(y, 0) : y + strip_width_px, :]
        if _strip_density(strip) >= density_threshold:
            bottom_y = max(y, 0)
        else:
            break
    if bottom_y < h:
        regions.append((0, bottom_y, w, h - bottom_y))

    # Top margin
    top_y = 0
    for y in range(0, max_margin_h, strip_width_px):
        strip = binary[y : y + strip_width_px, :]
        if _strip_density(strip) >= density_threshold:
            top_y = y + strip_width_px
        else:
            break
    if top_y > 0:
        regions.append((0, 0, w, top_y))

    # Left margin
    left_x = 0
    for x in range(0, max_margin_w, strip_width_px):
        strip = binary[:, x : x + strip_width_px]
        if _strip_density(strip) >= density_threshold:
            left_x = x + strip_width_px
        else:
            break
    if left_x > 0:
        regions.append((0, 0, left_x, h))

    return regions


def detect_title_block(
    binary: np.ndarray,
    min_line_fraction: float = 0.9,
    search_right_fraction: float = 0.45,
    min_hline_count: int = 8,
    min_hline_density_ratio: float = 1.8,
) -> tuple[int, int, int, int] | None:
    """Detect the title block region using structural line analysis.

    Architectural drawings have a title block bounded by a long vertical line
    on its left edge, with many short horizontal grid lines inside (table rows).
    This function finds that boundary and returns the region to exclude.

    Returns (x, y, w, h) bounding box or None if no title block found.
    """
    h, w = binary.shape

    # 1. Find long vertical lines using morphology (must span >90% of image)
    min_line_len = int(h * min_line_fraction)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # 2. Search the right portion for vertical line positions
    search_start_x = int(w * (1.0 - search_right_fraction))
    right_region = v_lines[:, search_start_x:]
    col_sums = np.sum(right_region, axis=0) / 255

    # Lines are columns where the sum exceeds a significant fraction of height
    line_threshold = h * 0.3
    line_cols = np.where(col_sums > line_threshold)[0]
    if len(line_cols) == 0:
        return None

    # Group nearby columns into line positions
    line_positions: list[int] = []
    group_start = line_cols[0]
    for i in range(1, len(line_cols)):
        if line_cols[i] - line_cols[i - 1] > 10:
            mid = (group_start + line_cols[i - 1]) // 2 + search_start_x
            line_positions.append(mid)
            group_start = line_cols[i]
    mid = (group_start + line_cols[-1]) // 2 + search_start_x
    line_positions.append(mid)

    if not line_positions:
        return None

    # 3. For each candidate boundary (leftmost first), check if the region
    #    to its right has many short horizontal lines (title block grid pattern)
    h_kernel_len = max(w // 10, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    for boundary_x in line_positions:
        region_w = w - boundary_x
        if region_w < 50:
            continue

        # Count horizontal lines in the candidate title block region
        tb_region = h_lines[:, boundary_x:]
        row_has_line = np.sum(tb_region, axis=1) / 255 > region_w * 0.3
        # Count transitions from no-line to line
        hline_count = 0
        in_line = False
        for has_line in row_has_line:
            if has_line and not in_line:
                hline_count += 1
                in_line = True
            elif not has_line:
                in_line = False

        # Count horizontal lines in the floorplan region (left of boundary)
        fp_region = h_lines[:, :boundary_x]
        fp_row_has_line = np.sum(fp_region, axis=1) / 255 > boundary_x * 0.3
        fp_hline_count = 0
        in_line = False
        for has_line in fp_row_has_line:
            if has_line and not in_line:
                fp_hline_count += 1
                in_line = True
            elif not has_line:
                in_line = False

        # Title block should have many horizontal lines, and a higher density
        # of them per unit width compared to the floorplan area
        if hline_count >= min_hline_count:
            tb_density = hline_count / max(region_w, 1)
            fp_density = fp_hline_count / max(boundary_x, 1)
            if fp_density == 0 or (tb_density / fp_density) >= min_hline_density_ratio:
                return (boundary_x, 0, w - boundary_x, h)

    return None
