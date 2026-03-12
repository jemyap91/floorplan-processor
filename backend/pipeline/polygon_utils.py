"""Shared polygon post-processing utilities."""
import math
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def merge_collinear_segments(
    polygon: Polygon,
    angle_threshold_deg: float = 8.0,
    min_segment_ratio: float = 0.02,
) -> Polygon:
    """Merge near-collinear consecutive segments in a polygon.

    This cleans up noisy contour vertices that create many tiny wall segments
    on what should be straight walls.

    Args:
        polygon: Input Shapely polygon
        angle_threshold_deg: Max angle deviation (degrees) to consider segments collinear
        min_segment_ratio: Segments shorter than this fraction of perimeter are
                          candidates for merging with neighbors

    Returns:
        Simplified polygon with collinear segments merged
    """
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:  # triangle or degenerate — nothing to merge
        return polygon

    # Remove closing duplicate
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 4:
        return polygon

    perimeter = polygon.length
    min_seg_len = perimeter * min_segment_ratio

    def _angle_between(p1, p2, p3):
        """Angle at p2 between segments p1->p2 and p2->p3, in degrees."""
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        if len1 == 0 or len2 == 0:
            return 0
        # Angle between the two direction vectors
        cos_angle = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def _seg_len(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # Iteratively merge until stable
    changed = True
    while changed:
        changed = False
        new_coords = []
        n = len(coords)
        if n < 4:
            break

        skip = set()
        for i in range(n):
            if i in skip:
                continue
            p_prev = coords[(i - 1) % n]
            p_curr = coords[i]
            p_next = coords[(i + 1) % n]

            seg_len = _seg_len(p_curr, p_next)
            deviation = abs(180 - _angle_between(p_prev, p_curr, p_next))

            # If the angle at this vertex is nearly straight (close to 180°),
            # and at least one adjacent segment is short, remove this vertex
            if deviation < angle_threshold_deg and seg_len < min_seg_len:
                skip.add(i)
                changed = True
            elif deviation < angle_threshold_deg:
                prev_seg_len = _seg_len(p_prev, p_curr)
                if prev_seg_len < min_seg_len:
                    skip.add(i)
                    changed = True
                else:
                    new_coords.append(p_curr)
            else:
                new_coords.append(p_curr)

        if changed and len(new_coords) >= 3:
            coords = new_coords
        else:
            break

    if len(coords) < 3:
        return polygon

    # Close the polygon
    coords.append(coords[0])
    try:
        result = Polygon(coords)
        if not result.is_valid:
            result = make_valid(result)
            if result.geom_type != "Polygon":
                return polygon
        return result
    except Exception:
        return polygon
