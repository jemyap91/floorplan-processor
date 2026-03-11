"""Segment rooms from wall mask using flood-fill and contour extraction."""
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def segment_rooms(
    wall_mask: np.ndarray,
    min_area_px: int = 500,
    max_area_ratio: float = 0.5,
    min_area_ratio: float = 0.001,
    simplify_tolerance: float = 2.0,
    min_compactness: float = 0.03,
    close_gap_px: int = 5,
    excluded_regions: list | None = None,
) -> list[dict]:
    if wall_mask.size == 0 or wall_mask.max() == 0:
        return []

    h, w = wall_mask.shape
    total_px = h * w
    max_area_px = total_px * max_area_ratio
    # Use the larger of the fixed minimum or the relative minimum
    effective_min_area = max(min_area_px, int(total_px * min_area_ratio))

    # Apply excluded regions by zeroing them out in a working copy
    work_mask = wall_mask.copy()
    if excluded_regions:
        for region in excluded_regions:
            rx, ry, rw, rh = region
            work_mask[ry : ry + rh, rx : rx + rw] = 0

    # Close small gaps in walls so rooms are properly enclosed
    if close_gap_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_gap_px, close_gap_px)
        )
        work_mask = cv2.morphologyEx(work_mask, cv2.MORPH_CLOSE, kernel)

    # Strategy 1: find holes in the wall mask (enclosed rooms appear as holes
    # in the wall contour hierarchy — contours whose parent != -1).
    contours, hierarchy = cv2.findContours(
        work_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    room_contours = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            h_info = hierarchy[0][i]
            parent = h_info[3]
            if parent != -1:
                # This contour is a hole inside a wall region — it is a room
                room_contours.append(contour)

    # Strategy 2: fallback — if no hole contours found, try inverted mask
    # (handles cases where walls are thin lines rather than thick filled areas)
    if not room_contours:
        room_mask = cv2.bitwise_not(work_mask)
        contours_inv, _ = cv2.findContours(
            room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Exclude contours touching the image border (background)
        for contour in contours_inv:
            x, y, cw, ch = cv2.boundingRect(contour)
            touches_border = x == 0 or y == 0 or (x + cw) >= w or (y + ch) >= h
            if not touches_border:
                room_contours.append(contour)

    rooms = []
    for contour in room_contours:
        area = cv2.contourArea(contour)
        if area < effective_min_area or area > max_area_px:
            continue

        # Filter by compactness (4π × area / perimeter²)
        # Rooms are roughly compact shapes; thin slivers and furniture are not
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            if compactness < min_compactness:
                continue

        # Filter by aspect ratio — rooms shouldn't be extremely elongated
        _, (bw, bh), _ = cv2.minAreaRect(contour)
        if bw > 0 and bh > 0:
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.1:
                continue

        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type != "Polygon":
                    continue
        except Exception:
            continue

        centroid = poly.centroid
        coords = list(poly.exterior.coords)
        boundary_lengths = []
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            boundary_lengths.append(float(np.sqrt(dx**2 + dy**2)))

        rooms.append(
            {
                "polygon": poly,
                "area_px": float(poly.area),
                "perimeter_px": float(poly.length),
                "centroid": (float(centroid.x), float(centroid.y)),
                "boundary_lengths_px": boundary_lengths,
                "contour": approx,
            }
        )

    return rooms
