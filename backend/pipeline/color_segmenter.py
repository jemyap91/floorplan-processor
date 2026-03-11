"""Segment rooms from color-coded floorplan zones.

Architectural floorplans often use pastel fills to indicate room zones.
This module detects those colored regions and extracts room polygons,
complementing the wall-based segmenter for unfilled areas.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def segment_rooms_by_color(
    image: np.ndarray,
    min_color_diff: int = 15,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.4,
    wall_darkness: int = 80,
    wall_dilate_px: int = 3,
    color_quant_step: int = 24,
    simplify_tolerance: float = 3.0,
    excluded_regions: list | None = None,
) -> list[dict]:
    """Extract rooms from color-coded zones in a floorplan image.

    Args:
        image: RGB image (numpy array, shape HxWx3)
        min_color_diff: Minimum channel difference to consider a pixel colored
        min_area_ratio: Minimum room area as fraction of image area
        max_area_ratio: Maximum room area as fraction of image area
        wall_darkness: Pixels darker than this in all channels are wall lines
        wall_dilate_px: Dilation kernel size for wall line barriers
        color_quant_step: Color quantization step size (lower = more color groups)
        simplify_tolerance: Douglas-Peucker polygon simplification tolerance
        excluded_regions: List of (x, y, w, h) bounding boxes to exclude

    Returns:
        List of room dicts with polygon, area_px, perimeter_px, centroid, etc.
    """
    if image.size == 0 or len(image.shape) != 3:
        return []

    h, w = image.shape[:2]
    total_px = h * w
    min_area_px = int(total_px * min_area_ratio)
    max_area_px = int(total_px * max_area_ratio)

    work = image.copy()

    # Mask excluded regions
    if excluded_regions:
        for rx, ry, rw, rh in excluded_regions:
            work[ry : ry + rh, rx : rx + rw] = 255  # white out

    # Step 1: Identify colored pixels
    colored_mask = _extract_colored_mask(work, min_color_diff)

    # Step 2: Extract wall lines as barriers
    wall_mask = _extract_wall_lines(work, wall_darkness, wall_dilate_px)

    # Step 3: Remove wall lines from colored mask (splits same-color rooms)
    colored_mask[wall_mask > 0] = 0

    # Step 4: Quantize colors and find connected components per color group
    quantized = (work // color_quant_step) * color_quant_step
    rooms = _extract_rooms_from_color_zones(
        colored_mask, quantized, min_area_px, max_area_px, simplify_tolerance
    )

    return rooms


def _extract_colored_mask(image: np.ndarray, min_diff: int) -> np.ndarray:
    """Create a binary mask of pixels that have noticeable color (not gray/white/black)."""
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    # Channel differences
    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    diff_rb = np.abs(r - b)
    max_diff = np.maximum(np.maximum(diff_rg, diff_gb), diff_rb)

    has_color = max_diff > min_diff

    # Exclude near-white and near-black
    is_white = (r > 245) & (g > 245) & (b > 245)
    is_black = (r < 25) & (g < 25) & (b < 25)

    mask = has_color & ~is_white & ~is_black
    return mask.astype(np.uint8) * 255


def _extract_wall_lines(
    image: np.ndarray, darkness: int, dilate_px: int
) -> np.ndarray:
    """Extract dark wall lines from the image."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    dark = (r < darkness) & (g < darkness) & (b < darkness)
    wall_mask = dark.astype(np.uint8) * 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px, dilate_px)
        )
        wall_mask = cv2.dilate(wall_mask, kernel)

    return wall_mask


def _contour_to_room(
    component_mask: np.ndarray,
    simplify_tolerance: float,
    min_fill_ratio: float = 0.15,
    offset_x: int = 0,
    offset_y: int = 0,
) -> dict | None:
    """Convert a binary component mask to a room dict, or None if invalid."""
    # Check fill ratio
    ys, xs = np.where(component_mask > 0)
    if len(ys) == 0:
        return None
    bx = xs.max() - xs.min() + 1
    by = ys.max() - ys.min() + 1
    area = int(np.sum(component_mask > 0))
    if bx * by > 0 and area / (bx * by) < min_fill_ratio:
        return None

    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, simplify_tolerance, True)
    if len(approx) < 3:
        return None

    points = [(int(p[0][0]) + offset_x, int(p[0][1]) + offset_y) for p in approx]
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type != "Polygon":
                return None
    except Exception:
        return None

    centroid = poly.centroid
    coords = list(poly.exterior.coords)
    boundary_lengths = []
    for j in range(len(coords) - 1):
        dx = coords[j + 1][0] - coords[j][0]
        dy = coords[j + 1][1] - coords[j][1]
        boundary_lengths.append(float(np.sqrt(dx**2 + dy**2)))

    return {
        "polygon": poly,
        "area_px": float(poly.area),
        "perimeter_px": float(poly.length),
        "centroid": (float(centroid.x), float(centroid.y)),
        "boundary_lengths_px": boundary_lengths,
        "contour": approx,
        "source": "color",
    }


def _extract_rooms_from_color_zones(
    colored_mask: np.ndarray,
    quantized: np.ndarray,
    min_area: int,
    max_area: int,
    simplify_tolerance: float,
    min_fill_ratio: float = 0.15,
    split_large_factor: float = 12.0,
) -> list[dict]:
    """Find connected components of similar color and extract room polygons."""
    h, w = colored_mask.shape

    # Create a combined label image: each unique quantized color in the
    # colored mask gets its own label space
    # First, create a single-channel color ID from quantized RGB
    q_r = quantized[:, :, 0].astype(np.int32)
    q_g = quantized[:, :, 1].astype(np.int32)
    q_b = quantized[:, :, 2].astype(np.int32)
    color_id = q_r * 1000000 + q_g * 1000 + q_b

    # Zero out non-colored pixels
    color_id[colored_mask == 0] = 0

    # Get unique color IDs (excluding 0)
    unique_ids = np.unique(color_id)
    unique_ids = unique_ids[unique_ids != 0]

    rooms: list[dict] = []
    # Threshold above which we attempt to split merged rooms
    split_threshold = int(min_area * split_large_factor)

    for uid in unique_ids:
        # Binary mask for this color group
        zone_mask = (color_id == uid).astype(np.uint8) * 255

        # Clean up: close small gaps within zones (3x3 to avoid bridging
        # wall gaps between adjacent rooms)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_CLOSE, close_kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            zone_mask, connectivity=8
        )

        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            # Reject thin strips / snake-like artifacts via bounding-box fill ratio
            bx = stats[i, cv2.CC_STAT_WIDTH]
            by = stats[i, cv2.CC_STAT_HEIGHT]
            bbox_area = bx * by
            if bbox_area > 0 and area / bbox_area < min_fill_ratio:
                continue

            component_mask = (labels == i).astype(np.uint8) * 255

            # For large components, try splitting without close kernel
            # (morphological close can bridge thin wall gaps between rooms)
            if area > split_threshold:
                # Re-extract this component from the raw (un-closed) mask
                raw_zone = (color_id == uid).astype(np.uint8) * 255
                # Restrict to the bounding box of this component
                sx = stats[i, cv2.CC_STAT_LEFT]
                sy = stats[i, cv2.CC_STAT_TOP]
                raw_crop = raw_zone[sy : sy + by, sx : sx + bx]
                sub_n, sub_labels, sub_stats, sub_cents = (
                    cv2.connectedComponentsWithStats(raw_crop, connectivity=8)
                )
                # If the raw mask splits into multiple valid sub-rooms, use them
                valid_subs = [
                    j
                    for j in range(1, sub_n)
                    if sub_stats[j, cv2.CC_STAT_AREA] >= min_area
                ]
                if len(valid_subs) > 1:
                    for j in valid_subs:
                        sub_mask = (sub_labels == j).astype(np.uint8) * 255
                        room = _contour_to_room(
                            sub_mask, simplify_tolerance, min_fill_ratio,
                            offset_x=sx, offset_y=sy,
                        )
                        if room:
                            rooms.append(room)
                    continue

            room = _contour_to_room(
                component_mask, simplify_tolerance, min_fill_ratio,
            )
            if room:
                rooms.append(room)

    return rooms


def merge_room_lists(
    color_rooms: list[dict],
    wall_rooms: list[dict],
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """Merge color-detected and wall-detected rooms, removing duplicates.

    Wall rooms that overlap significantly with a color room are discarded.
    """
    merged = list(color_rooms)

    for wr in wall_rooms:
        wp: Polygon = wr["polygon"]
        is_duplicate = False

        for cr in color_rooms:
            cp: Polygon = cr["polygon"]
            try:
                if not wp.intersects(cp):
                    continue
                intersection = wp.intersection(cp).area
                overlap = intersection / wp.area if wp.area > 0 else 0
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            except Exception:
                continue

        if not is_duplicate:
            merged.append(wr)

    # Sort by area descending
    merged.sort(key=lambda r: r["area_px"], reverse=True)
    return merged
