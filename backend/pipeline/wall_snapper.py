"""Wall snapper — refine Gemini polygon vertices by snapping to nearest dark pixels.

For each vertex in a polygon, searches a configurable radius in the grayscale
image for the nearest cluster of dark pixels (wall/ink lines). Vertices near
walls get pulled onto the wall; vertices far from any wall stay unchanged.
"""
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid


def _find_nearest_dark_pixel(
    gray: np.ndarray,
    point: tuple[int, int],
    radius: int = 40,
    dark_threshold: int = 80,
) -> tuple[int, int] | None:
    """Find the nearest dark pixel to a point within a search radius.

    Args:
        gray: Grayscale image (0=black, 255=white).
        point: (x, y) pixel coordinate to search around.
        radius: Search radius in pixels.
        dark_threshold: Pixels with value <= this are considered dark.

    Returns:
        (x, y) of nearest dark pixel, or None if none found within radius.
    """
    px, py = point
    h, w = gray.shape

    # Define search window, clipped to image bounds
    x1 = max(px - radius, 0)
    y1 = max(py - radius, 0)
    x2 = min(px + radius + 1, w)
    y2 = min(py + radius + 1, h)

    crop = gray[y1:y2, x1:x2]

    # Find dark pixels
    dark_ys, dark_xs = np.where(crop <= dark_threshold)
    if len(dark_xs) == 0:
        return None

    # Convert back to image coordinates
    dark_xs_img = dark_xs + x1
    dark_ys_img = dark_ys + y1

    # Filter to circular radius
    dx = dark_xs_img - px
    dy = dark_ys_img - py
    distances_sq = dx * dx + dy * dy
    within_radius = distances_sq <= radius * radius

    if not np.any(within_radius):
        return None

    # Find the nearest dark pixel
    distances_sq_filtered = distances_sq[within_radius]
    xs_filtered = dark_xs_img[within_radius]
    ys_filtered = dark_ys_img[within_radius]

    nearest_idx = np.argmin(distances_sq_filtered)
    return (int(xs_filtered[nearest_idx]), int(ys_filtered[nearest_idx]))


def snap_polygon_to_walls(
    polygon: list[tuple[int, int]],
    gray: np.ndarray,
    radius: int = 40,
    dark_threshold: int = 80,
) -> list[tuple[int, int]]:
    """Snap polygon vertices to nearest dark pixels (wall lines).

    For each vertex, searches within `radius` pixels for the nearest dark pixel.
    If found, the vertex is moved to that location. If not found, the vertex
    stays unchanged. After snapping, validates the polygon — if snapping caused
    a self-intersection, returns the original polygon.

    Args:
        polygon: List of (x, y) vertex coordinates.
        gray: Grayscale image (0=black, 255=white).
        radius: Search radius in pixels.
        dark_threshold: Pixels with value <= this are considered dark/wall.

    Returns:
        List of (x, y) snapped vertex coordinates.
    """
    if len(polygon) < 3:
        return polygon

    snapped = []
    for vertex in polygon:
        nearest = _find_nearest_dark_pixel(gray, vertex, radius=radius, dark_threshold=dark_threshold)
        if nearest is not None:
            snapped.append(nearest)
        else:
            snapped.append(vertex)

    # Validate: if snapping broke the polygon, revert
    snapped_poly = ShapelyPolygon(snapped)
    if not snapped_poly.is_valid:
        # Try to fix with make_valid
        fixed = make_valid(snapped_poly)
        if fixed.geom_type == "Polygon" and fixed.is_valid:
            coords = list(fixed.exterior.coords)[:-1]  # drop closing duplicate
            return [(int(round(x)), int(round(y))) for x, y in coords]
        # Can't fix — return original
        return polygon

    return snapped
