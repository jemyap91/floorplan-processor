"""Furnished floorplan analyzer — all-in OpenCV pipeline.

Pipeline:
1. Downscale 2x for better signal-to-noise
2. Color filter (remove red/green/blue/orange annotation lines)
3. Furniture-resistant morphological filtering (erode thin, keep thick)
4. H/V wall line extraction
5. Grid line removal (lines spanning >60% of image)
6. Door arc detection (quarter-circle swing arcs)
7. Wall endpoint gap closing at door locations
8. Directional morphological closing fallback
9. Room segmentation via contour hierarchy
10. Gemini labels detected rooms (names and types only)
11. Upscale polygon coordinates back to original resolution
"""
import cv2
import logging
import math
import os
import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Downscale
# ---------------------------------------------------------------------------

def _downscale(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downscale image by integer factor using area interpolation."""
    h, w = image.shape[:2]
    return cv2.resize(image, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Step 2: Color filtering
# ---------------------------------------------------------------------------

def _filter_colors(image: np.ndarray) -> np.ndarray:
    """Remove colored annotations (red, green, blue, orange) from image.

    Returns a BGR image where colored pixels have been replaced with white,
    keeping only dark/black ink (walls, furniture, text).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = []

    # Red (wraps around hue 0): hue 0-10 and 170-180
    masks.append(cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255])))

    # Orange: hue 10-25
    masks.append(cv2.inRange(hsv, np.array([10, 80, 80]), np.array([25, 255, 255])))

    # Green: hue 35-85
    masks.append(cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255])))

    # Blue: hue 95-135
    masks.append(cv2.inRange(hsv, np.array([95, 80, 80]), np.array([135, 255, 255])))

    # Magenta/pink: hue 140-170
    masks.append(cv2.inRange(hsv, np.array([140, 60, 60]), np.array([170, 255, 255])))

    colored_mask = masks[0]
    for m in masks[1:]:
        colored_mask = cv2.bitwise_or(colored_mask, m)

    # Replace colored pixels with white
    result = image.copy()
    result[colored_mask > 0] = (255, 255, 255)

    removed = np.count_nonzero(colored_mask)
    _logger.info(f"Color filter: removed {removed} colored pixels")
    return result


# ---------------------------------------------------------------------------
# Step 3-4: Wall extraction via color + thickness filtering
# ---------------------------------------------------------------------------

def _extract_walls(
    gray: np.ndarray,
    wall_grey_lo: int = 80,
    wall_grey_hi: int = 140,
    thick_radius: int = 3,
    faint_grey_lo: int = 60,
    faint_grey_hi: int = 130,
    faint_radius: int = 2,
    adaptive_radius: int = 2,
    dilate_size: int = 5,
) -> dict:
    """Extract wall lines using color-band isolation + distance-transform thickness filtering.

    Walls in furnished floorplans are thick dark grey lines (~8-20px wide,
    grey value ~96-102). This differs from furniture (thin black ink, 1-3px)
    and annotations (colored, thin). We exploit both properties:

    Pass 1 (main walls): Isolate grey-band pixels (80-140), then keep only
    features ≥ 2*thick_radius px wide via distance transform. Catches the
    dominant wall color with zero furniture noise.

    Pass 2 (faint walls): CLAHE-enhance then isolate a wider grey band (60-130),
    keep features ≥ 2*faint_radius px wide. Catches walls drawn with lighter
    line weight (common in right-side units).

    Pass 3 (adaptive walls): Use adaptive thresholding to capture walls at ANY
    grey shade (including lighter interior walls at grey 140-200 that passes 1-2
    miss), then keep features ≥ 2*adaptive_radius px wide. This catches walls
    regardless of absolute grey value as long as they're thick enough.

    Returns dict with 'binary' (all ink), 'eroded' (thick features only),
    'wall_mask' (combined wall cores, dilated back to approximate wall width).
    """
    h, w = gray.shape

    # Standard binary for door arc detection (kept for compatibility)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10,
    )

    # --- Pass 1: Grey-band + distance transform (main walls) ---
    grey_mask = ((gray >= wall_grey_lo) & (gray <= wall_grey_hi)).astype(np.uint8) * 255
    dist1 = cv2.distanceTransform(grey_mask, cv2.DIST_L2, 5)
    thick_cores = (dist1 >= thick_radius).astype(np.uint8) * 255

    # --- Pass 2: CLAHE-enhanced grey-band + distance transform (faint walls) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    faint_mask = ((enhanced >= faint_grey_lo) & (enhanced <= faint_grey_hi)).astype(np.uint8) * 255
    dist2 = cv2.distanceTransform(faint_mask, cv2.DIST_L2, 5)
    faint_cores = (dist2 >= faint_radius).astype(np.uint8) * 255

    # Combine passes 1+2 (grey-band only — safe for footprint detection)
    wall_cores = cv2.bitwise_or(thick_cores, faint_cores)

    # Dilate wall cores back to approximate wall width
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    wall_mask = cv2.dilate(wall_cores, dk, iterations=1)

    # Eroded = thick cores only (used for door arc detection)
    eroded = thick_cores

    p1_count = np.count_nonzero(thick_cores)
    p2_only = np.count_nonzero(wall_cores) - p1_count
    _logger.info(
        f"Wall extraction (color+thickness): {w}x{h}, "
        f"grey_band={np.count_nonzero(grey_mask)}, "
        f"thick_cores(r>={thick_radius})={p1_count}, "
        f"faint_cores_added(r>={faint_radius})={p2_only}, "
        f"total_wall_mask={np.count_nonzero(wall_mask)}"
    )
    return {
        "binary": binary, "eroded": eroded, "wall_mask": wall_mask,
        "adaptive_radius": adaptive_radius,
    }


# ---------------------------------------------------------------------------
# Step 5: Grid line removal
# ---------------------------------------------------------------------------

def _remove_grid_lines(wall_mask: np.ndarray, span_threshold: float = 0.55) -> np.ndarray:
    """Remove lines that span more than span_threshold of image width/height.

    These are typically structural grid lines (labeled 1-5, A-E) that extend
    across the entire drawing and create false room boundaries.
    """
    h, w = wall_mask.shape
    result = wall_mask.copy()

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)

    removed = 0
    for i in range(1, num_labels):  # skip background (0)
        x, y, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Horizontal grid line: spans most of image width, very thin vertically
        if cw > w * span_threshold and ch < h * 0.02:
            result[labels == i] = 0
            removed += 1

        # Vertical grid line: spans most of image height, very thin horizontally
        elif ch > h * span_threshold and cw < w * 0.02:
            result[labels == i] = 0
            removed += 1

    _logger.info(f"Grid line removal: removed {removed} grid lines")
    return result


# ---------------------------------------------------------------------------
# Step 5b: Building footprint detection
# ---------------------------------------------------------------------------

def _detect_building_footprint(
    wall_mask: np.ndarray,
    margin_px: int = 20,
) -> tuple[int, int, int, int]:
    """Detect the building footprint bounding box from wall pixel locations.

    Uses the bounding box of all nonzero wall pixels. The upstream wall
    extraction (grey-band + distance-transform) already filters out non-wall
    features (text, furniture, annotations), so any surviving wall pixel is
    part of the building.

    Returns (x1, y1, x2, y2) bounding box of the building area.
    """
    h, w = wall_mask.shape

    cols_with_walls = np.any(wall_mask > 0, axis=0)
    rows_with_walls = np.any(wall_mask > 0, axis=1)

    if not np.any(cols_with_walls) or not np.any(rows_with_walls):
        _logger.warning("Building footprint detection: no wall pixels found, using full image")
        return (0, 0, w, h)

    x1 = max(0, int(np.argmax(cols_with_walls)) - margin_px)
    x2 = min(w, w - int(np.argmax(cols_with_walls[::-1])) + margin_px)
    y1 = max(0, int(np.argmax(rows_with_walls)) - margin_px)
    y2 = min(h, h - int(np.argmax(rows_with_walls[::-1])) + margin_px)

    _logger.info(
        f"Building footprint: ({x1},{y1})-({x2},{y2}) = "
        f"{x2-x1}x{y2-y1} of {w}x{h} ({100*(x2-x1)*(y2-y1)/(w*h):.0f}% of image)"
    )
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Step 6: Door arc detection
# ---------------------------------------------------------------------------

def _detect_door_arcs(binary: np.ndarray, eroded: np.ndarray) -> list[tuple]:
    """Detect door swing arcs by finding thin curved features near wall gaps.

    Door arcs are thin quarter-circle curves that exist in the original binary
    but get removed by erosion (they're thinner than walls).

    Returns list of (cx, cy, radius) tuples for detected door locations.
    """
    h, w = binary.shape

    # Thin features = present in binary but not in eroded (furniture + door arcs + text)
    thin_features = cv2.subtract(binary, eroded)

    # Clean up: remove tiny noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thin_features = cv2.morphologyEx(thin_features, cv2.MORPH_OPEN, clean_kernel)

    # Find contours in thin features
    contours, _ = cv2.findContours(thin_features, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doors = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > 20000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 20:
            continue

        # Door arcs have high perimeter relative to area (they're thin curves)
        # A filled circle has ratio perimeter²/area = 4π ≈ 12.6
        # A thin arc has much higher ratio (100+)
        if perimeter > 0:
            thinness = (perimeter * perimeter) / max(area, 1)
            if thinness < 30:
                continue  # too filled/compact — not an arc

        # Check bounding box — door arcs are roughly square (quarter circle)
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.3:
            continue  # too elongated — probably a line, not an arc

        # Estimate radius from bounding box (quarter circle fits in a square)
        radius = max(bw, bh)

        # Filter by reasonable door size (at downscaled resolution)
        # Doors are typically 30-80px wide at half resolution
        if radius < 15 or radius > 120:
            continue

        # Check solidity (area / convex hull area) — arcs have low solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.5:
                continue  # too solid — not a thin arc

        # Door location is the corner of the bounding box (hinge point)
        # The arc sweeps from one edge — use centroid as approximate location
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            doors.append((cx, cy, radius))

    _logger.info(f"Door detection: found {len(doors)} door arcs")
    return doors


# ---------------------------------------------------------------------------
# Step 7: Wall endpoint gap closing at door locations
# ---------------------------------------------------------------------------

def _close_door_gaps(
    wall_mask: np.ndarray,
    doors: list[tuple],
    search_radius_mult: float = 1.5,
) -> np.ndarray:
    """Close wall gaps at detected door locations.

    For each door arc, search the surrounding area for wall endpoints and
    draw a line across the gap to close the door opening.
    """
    result = wall_mask.copy()
    h, w = wall_mask.shape
    closed_count = 0

    for cx, cy, radius in doors:
        search_r = int(radius * search_radius_mult)

        # Extract search region around door
        x1 = max(cx - search_r, 0)
        y1 = max(cy - search_r, 0)
        x2 = min(cx + search_r, w)
        y2 = min(cy + search_r, h)

        crop = result[y1:y2, x1:x2]
        if crop.size == 0 or np.count_nonzero(crop) == 0:
            continue

        # Find wall pixels in the search region
        wall_ys, wall_xs = np.where(crop > 0)
        if len(wall_xs) < 2:
            continue

        # Convert to absolute coordinates
        wall_xs_abs = wall_xs + x1
        wall_ys_abs = wall_ys + y1

        # Find wall pixels that are near the edge of wall regions (endpoints)
        # Use morphological erosion on the crop to find interior vs edge
        erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        interior = cv2.erode(crop, erode_k, iterations=1)
        edge = cv2.subtract(crop, interior)

        edge_ys, edge_xs = np.where(edge > 0)
        if len(edge_xs) < 2:
            continue

        edge_xs_abs = edge_xs + x1
        edge_ys_abs = edge_ys + y1

        # Cluster edge pixels into wall segments using connected components
        num_edge, edge_labeled = cv2.connectedComponents(edge, connectivity=8)

        if num_edge < 3:  # need at least 2 wall segments (label 0 is background)
            continue

        # Find the centroid of each wall segment cluster
        segment_centers = []
        for label_id in range(1, num_edge):
            seg_ys, seg_xs = np.where(edge_labeled == label_id)
            if len(seg_xs) < 3:
                continue
            scx = int(np.mean(seg_xs)) + x1
            scy = int(np.mean(seg_ys)) + y1
            segment_centers.append((scx, scy))

        if len(segment_centers) < 2:
            continue

        # Find the two closest segments that are on opposite sides of the door
        # (i.e., the gap endpoints)
        best_pair = None
        best_dist = float('inf')
        for i in range(len(segment_centers)):
            for j in range(i + 1, len(segment_centers)):
                sx1, sy1 = segment_centers[i]
                sx2, sy2 = segment_centers[j]
                dist = math.sqrt((sx2 - sx1)**2 + (sy2 - sy1)**2)
                # Gap should be roughly door-sized
                if radius * 0.5 < dist < radius * 3.0 and dist < best_dist:
                    best_dist = dist
                    best_pair = (segment_centers[i], segment_centers[j])

        if best_pair is not None:
            pt1, pt2 = best_pair
            # Draw a thick line to close the gap
            thickness = max(3, int(radius * 0.15))
            cv2.line(result, pt1, pt2, 255, thickness)
            closed_count += 1

    _logger.info(f"Door gap closing: closed {closed_count}/{len(doors)} door gaps")
    return result


# ---------------------------------------------------------------------------
# Step 8: Sealed flood-fill room detection
# ---------------------------------------------------------------------------

def _detect_rooms_floodfill(
    wall_mask: np.ndarray,
    footprint: tuple[int, int, int, int],
    gap_close_px: int | None = None,
    min_area_px: int = 800,
    max_area_ratio: float = 0.85,
    simplify_tolerance: float = 2.0,
) -> list[dict]:
    """Detect rooms by sealing the building perimeter and flood-filling.

    Algorithm:
    1. Draw a solid rectangle around the building footprint to seal the exterior
    2. Apply directional morphological closing to bridge door-sized gaps
       (long narrow kernels close gaps without fattening walls)
    3. Invert the mask and find connected components
    4. The component touching the image border is exterior — discard it
    5. Each remaining component is a room candidate

    This approach is more robust than multi-scale dilation because:
    - Directional closing bridges door gaps without merging parallel walls
    - Flood-fill cleanly separates exterior from interior
    - No deduplication or containment filtering needed
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid
    from backend.pipeline.polygon_utils import merge_collinear_segments

    sh, sw = wall_mask.shape
    dim = max(sw, sh)

    # Scale-adaptive gap close size (~45px at 3300px image width)
    if gap_close_px is None:
        gap_close_px = max(30, int(dim * 0.014))

    bx1, by1, bx2, by2 = footprint

    # Step 1: Seal building perimeter
    sealed = wall_mask.copy()
    cv2.rectangle(sealed, (bx1, by1), (bx2, by2), 255, 5)

    # Step 2: Directional close to bridge door gaps
    # Horizontal kernel closes gaps in vertical walls (door openings)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_close_px, 3))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, hk)
    # Vertical kernel closes gaps in horizontal walls
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap_close_px))
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, vk)

    # Step 3: Connected components on inverted mask
    inv = cv2.bitwise_not(sealed)
    num_labels, labels = cv2.connectedComponents(inv, connectivity=8)

    # Step 4: Identify exterior (component touching the image border)
    exterior_label = labels[0, 0]
    # Check all corners in case (0,0) is wall
    for cy, cx in [(0, 0), (0, sw - 1), (sh - 1, 0), (sh - 1, sw - 1)]:
        lbl = labels[cy, cx]
        if lbl > 0:
            exterior_label = lbl
            break

    # Step 5: Each non-exterior component is a room candidate
    # Use footprint area (not total image) as the denominator — the building
    # footprint masking already excludes margins/title blocks, so we only
    # need to guard against one room dominating the *building* area.
    fp_w = max(1, bx2 - bx1)
    fp_h = max(1, by2 - by1)
    footprint_px = fp_w * fp_h
    max_area_px = int(footprint_px * max_area_ratio)
    rooms = []

    for lbl in range(1, num_labels):
        if lbl == exterior_label:
            continue

        area = int(np.count_nonzero(labels == lbl))
        if area < min_area_px or area > max_area_px:
            continue

        # Extract contour for this component.
        # Dilate by half the wall thickness so contours follow wall
        # centerlines rather than the inner edge of empty space.
        component = (labels == lbl).astype(np.uint8) * 255
        wall_half = 5  # ~10px walls at downscaled res → half = 5
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_half * 2 + 1, wall_half * 2 + 1))
        component = cv2.dilate(component, dk, iterations=1)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Skip rooms that border the sealed perimeter rectangle on 3+ sides
        # (these are exterior artifacts from the seal, not real rooms).
        # Rooms touching exactly 2 sides are likely corner rooms (bedrooms etc.)
        # and should only be filtered if they're very large (>30% of footprint).
        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2)
        seal_margin = 10  # pixels from perimeter edge
        touches_left = bool(pts[:, 0].min() <= bx1 + seal_margin)
        touches_right = bool(pts[:, 0].max() >= bx2 - seal_margin)
        touches_top = bool(pts[:, 1].min() <= by1 + seal_margin)
        touches_bottom = bool(pts[:, 1].max() >= by2 - seal_margin)
        border_sides = sum([touches_left, touches_right, touches_top, touches_bottom])
        if border_sides >= 3:
            _logger.debug(f"Skipping room (label {lbl}): touches {border_sides} perimeter sides")
            continue
        if border_sides >= 2 and area > int(footprint_px * 0.30):
            _logger.debug(f"Skipping large border room (label {lbl}): area={area}, touches {border_sides} sides")
            continue

        approx = cv2.approxPolyDP(contour, simplify_tolerance, True)
        if len(approx) < 3:
            continue

        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        try:
            poly = ShapelyPolygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type != "Polygon":
                    continue
        except Exception:
            continue

        poly = merge_collinear_segments(poly)
        centroid = poly.centroid
        coords = list(poly.exterior.coords)
        boundary_lengths = []
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            boundary_lengths.append(float(np.sqrt(dx ** 2 + dy ** 2)))

        rooms.append({
            "polygon": poly,
            "area_px": float(poly.area),
            "perimeter_px": float(poly.length),
            "centroid": (float(centroid.x), float(centroid.y)),
            "boundary_lengths_px": boundary_lengths,
            "contour": approx,
        })

    _logger.info(
        f"Flood-fill room detection: {len(rooms)} rooms "
        f"(gap_close={gap_close_px}px, {num_labels} components, "
        f"exterior=label_{exterior_label})"
    )
    return rooms


def _subdivide_large_rooms(
    rooms: list[dict],
    wall_mask: np.ndarray,
    footprint: tuple[int, int, int, int],
    area_threshold_ratio: float = 0.03,
    min_sub_area: int = 400,
) -> list[dict]:
    """Split large merged rooms into sub-rooms via local flood-fill.

    Rooms bigger than area_threshold_ratio of the building footprint are
    likely multiple rooms connected through wide door gaps. For each,
    we seal the room boundary, apply a local gap close proportional to
    the room's own dimensions, and re-run flood-fill within that region.
    """
    from shapely.affinity import translate

    bx1, by1, bx2, by2 = footprint
    fp_area = max(1, (bx2 - bx1) * (by2 - by1))
    threshold = fp_area * area_threshold_ratio

    small_rooms = []
    large_rooms = []
    for r in rooms:
        if r["area_px"] > threshold:
            large_rooms.append(r)
        else:
            small_rooms.append(r)

    if not large_rooms:
        return rooms

    subdivided = []
    for lr in large_rooms:
        poly = lr["polygon"]
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        margin = 20
        rx1 = max(0, int(bounds[0]) - margin)
        ry1 = max(0, int(bounds[1]) - margin)
        rx2 = min(wall_mask.shape[1], int(bounds[2]) + margin)
        ry2 = min(wall_mask.shape[0], int(bounds[3]) + margin)

        # Crop wall mask; seal the room boundary so flood-fill can work
        region_mask = wall_mask[ry1:ry2, rx1:rx2].copy()
        contour = lr["contour"]
        shifted = contour.copy()
        shifted[:, 0, 0] -= rx1
        shifted[:, 0, 1] -= ry1
        cv2.drawContours(region_mask, [shifted], -1, 255, 3)

        rh, rw = region_mask.shape
        local_gap = max(30, int(max(rw, rh) * 0.05))
        local_fp = (0, 0, rw, rh)
        sub_rooms = _detect_rooms_floodfill(
            region_mask, local_fp,
            gap_close_px=local_gap,
            min_area_px=min_sub_area,
        )

        if len(sub_rooms) > 1:
            # Offset sub-room coordinates back to global
            for sr in sub_rooms:
                sr["centroid"] = (sr["centroid"][0] + rx1, sr["centroid"][1] + ry1)
                sr["polygon"] = translate(sr["polygon"], xoff=rx1, yoff=ry1)
                coords = list(sr["polygon"].exterior.coords)
                sr["boundary_lengths_px"] = []
                for i in range(len(coords) - 1):
                    dx = coords[i + 1][0] - coords[i][0]
                    dy = coords[i + 1][1] - coords[i][1]
                    sr["boundary_lengths_px"].append(float(np.sqrt(dx ** 2 + dy ** 2)))
                sr["contour"] = np.array(
                    [[[int(x), int(y)]] for x, y in coords[:-1]], dtype=np.int32
                )
            subdivided.extend(sub_rooms)
            _logger.info(
                f"Subdivided large room (area={lr['area_px']:.0f}) "
                f"into {len(sub_rooms)} sub-rooms (local_gap={local_gap})"
            )
        else:
            subdivided.append(lr)

    result = small_rooms + subdivided
    _logger.info(
        f"Room subdivision: {len(rooms)} -> {len(result)} rooms "
        f"({len(large_rooms)} large rooms processed)"
    )
    return result


def _snap_rooms_to_walls(
    rooms: list[dict],
    wall_mask: np.ndarray,
    radius: int | None = None,
) -> list[dict]:
    """Snap room polygon vertices to nearest wall pixels.

    For each room, extracts the polygon vertices and snaps each one to the
    nearest white pixel in the wall_mask within a search radius. This pulls
    polygon edges onto actual wall lines without the fragility of flood-fill.

    Uses wall_mask (binary: wall=255, background=0) inverted to grayscale
    where walls=0 (dark) so the wall_snapper's dark-pixel search works.
    """
    from backend.pipeline.wall_snapper import snap_polygon_to_walls
    from shapely.geometry import Polygon as ShapelyPolygon

    sh, sw = wall_mask.shape
    dim = max(sw, sh)
    if radius is None:
        radius = max(20, int(dim * 0.008))  # ~40px at 5000px

    # Invert wall mask: walls=0 (dark), background=255 (light)
    gray_for_snap = cv2.bitwise_not(wall_mask)

    snapped_count = 0
    refined = []
    for room in rooms:
        poly = room["polygon"]
        coords = list(poly.exterior.coords)[:-1]  # drop closing duplicate
        vertices = [(int(round(x)), int(round(y))) for x, y in coords]

        snapped = snap_polygon_to_walls(
            vertices, gray_for_snap,
            radius=radius, dark_threshold=10,  # very strict — only actual wall pixels
        )

        if snapped != vertices:
            snapped_count += 1
            # Rebuild room dict with snapped polygon
            try:
                new_poly = ShapelyPolygon(snapped)
                if not new_poly.is_valid:
                    from shapely.validation import make_valid
                    new_poly = make_valid(new_poly)
                    if new_poly.geom_type != "Polygon":
                        refined.append(room)
                        continue

                centroid = new_poly.centroid
                new_coords = list(new_poly.exterior.coords)
                blens = []
                for j in range(len(new_coords) - 1):
                    dx = new_coords[j + 1][0] - new_coords[j][0]
                    dy = new_coords[j + 1][1] - new_coords[j][1]
                    blens.append(float(np.sqrt(dx**2 + dy**2)))

                refined.append({
                    "polygon": new_poly,
                    "area_px": float(new_poly.area),
                    "perimeter_px": float(new_poly.length),
                    "centroid": (float(centroid.x), float(centroid.y)),
                    "boundary_lengths_px": blens,
                    "contour": room.get("contour"),
                })
            except Exception:
                refined.append(room)
        else:
            refined.append(room)

    _logger.info(f"Wall snapping: {snapped_count}/{len(rooms)} room polygons snapped to walls")
    return refined


# ---------------------------------------------------------------------------
# Debug images
# ---------------------------------------------------------------------------

def _save_debug_images(
    image: np.ndarray,
    wall_mask: np.ndarray,
    rooms: list[dict],
    debug_dir: str,
    doors: list[tuple] | None = None,
    scale_factor: int = 1,
    footprint: tuple[int, int, int, int] | None = None,
):
    """Save annotated debug images at original resolution."""
    os.makedirs(debug_dir, exist_ok=True)
    sf = scale_factor

    # Image 1: wall mask overlay (upscale wall mask to original resolution)
    if sf > 1:
        h_orig, w_orig = image.shape[:2]
        wall_vis = cv2.resize(wall_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        wall_vis = wall_mask

    img_walls = image.copy()
    wall_color = np.zeros_like(image)
    wall_color[wall_vis > 0] = (0, 255, 0)  # green walls
    img_walls = cv2.addWeighted(img_walls, 0.7, wall_color, 0.3, 0)

    # Draw detected doors as circles
    if doors:
        for cx, cy, radius in doors:
            cv2.circle(img_walls, (cx * sf, cy * sf), radius * sf, (0, 0, 255), 2)

    # Draw building footprint bounding box
    if footprint:
        fx1, fy1, fx2, fy2 = footprint
        cv2.rectangle(img_walls, (fx1 * sf, fy1 * sf), (fx2 * sf, fy2 * sf), (255, 0, 255), 3)

    cv2.imwrite(os.path.join(debug_dir, "debug_walls.png"), img_walls)

    # Image 2: room polygons with labels
    img_rooms = image.copy()
    colors = [
        (76, 175, 80), (33, 150, 243), (255, 152, 0), (156, 39, 176),
        (244, 67, 54), (0, 188, 212), (255, 235, 59), (121, 85, 72),
    ]
    for i, r in enumerate(rooms):
        pts = np.array(r["polygon_px"], dtype=np.int32)
        color = colors[i % len(colors)]
        cv2.fillPoly(img_rooms, [pts], color, lineType=cv2.LINE_AA)
    img_rooms = cv2.addWeighted(image, 0.5, img_rooms, 0.5, 0)
    for i, r in enumerate(rooms):
        pts = np.array(r["polygon_px"], dtype=np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(img_rooms, [pts], True, color, 2)
        if len(pts) > 0:
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = r.get("name", "")
            cv2.putText(img_rooms, label, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, "debug_rooms.png"), img_rooms)

    # Image 3: raw wall mask for inspection
    cv2.imwrite(os.path.join(debug_dir, "debug_wallmask.png"), wall_mask)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_furnished_pipeline(
    image: np.ndarray,
    debug_dir: str | None = None,
    progress_cb=None,
    gemini_model: str | None = None,
) -> list[dict]:
    """Run the furnished floorplan pipeline: Gemini bbox + OpenCV rooms.

    Pipeline:
    1. Downscale 2x for better signal-to-noise
    2. Color filter (remove red/green/blue/orange annotation lines)
    3. Dual-pass wall extraction (thick + CLAHE-enhanced thin walls)
    4. Grid line removal (lines spanning >55% of image)
    5. Gemini building bbox (semantic detection of building footprint)
    6. Door arc detection + wall endpoint gap closing
    7. Multi-scale room detection (6 dilation levels, deduplicate)
    8. Wall snapping (snap polygon vertices to nearest wall pixels)
    9. Gemini labeling (names and types only)
    10. Upscale coordinates back to original resolution

    Args:
        image: Full floorplan image (BGR numpy array).
        debug_dir: Directory for debug images (None to skip).
        progress_cb: Optional callback(percent, message).
        gemini_model: Gemini model name for bbox detection (None = default Flash).

    Returns:
        List of room dicts with polygon_px, name, type, area_px, etc.
    """
    _logger.info("Starting furnished pipeline (Gemini bbox + OpenCV rooms)")

    def _progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    scale_factor = 2

    # Step 1: Downscale
    _progress(5, "Downscaling image...")
    small = _downscale(image, factor=scale_factor)
    _logger.info(f"Downscaled: {image.shape[:2]} -> {small.shape[:2]}")

    # Step 2: Color filter
    _progress(10, "Filtering colored annotations...")
    filtered = _filter_colors(small)

    # Step 3-4: Furniture erosion + wall extraction
    _progress(15, "Extracting wall lines...")
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    walls = _extract_walls(gray)
    wall_mask = walls["wall_mask"]

    # Step 5: Grid line removal
    _progress(25, "Removing grid lines...")
    wall_mask = _remove_grid_lines(wall_mask)

    # Step 6: Detect building footprint via Gemini (semantic understanding).
    # Gemini identifies the actual building vs title blocks, margins, annotations.
    # Falls back to CV density-based detection if Gemini fails.
    _progress(30, "Detecting building footprint (Gemini)...")
    from backend.pipeline.vision_ai import detect_building_bbox
    footprint = detect_building_bbox(image, model=gemini_model)
    # Convert full-image coords to downscaled coords
    bx1 = footprint[0] // scale_factor
    by1 = footprint[1] // scale_factor
    bx2 = footprint[2] // scale_factor
    by2 = footprint[3] // scale_factor
    _logger.info(f"Building footprint (downscaled): ({bx1},{by1})-({bx2},{by2})")
    # If Gemini returned full image (failure), fall back to CV density
    h_ds, w_ds = wall_mask.shape
    if bx1 == 0 and by1 == 0 and bx2 >= w_ds - 1 and by2 >= h_ds - 1:
        _logger.info("Gemini bbox covers full image, falling back to CV density detection")
        cv_footprint = _detect_building_footprint(wall_mask)
        bx1, by1, bx2, by2 = cv_footprint
    footprint = (bx1 * scale_factor, by1 * scale_factor,
                 bx2 * scale_factor, by2 * scale_factor)
    building_mask = np.zeros_like(wall_mask)
    building_mask[by1:by2, bx1:bx2] = wall_mask[by1:by2, bx1:bx2]
    wall_mask = building_mask

    # Step 6b: Add adaptive-threshold wall pass within footprint only.
    # Catches interior walls at grey 140-200 that grey-band passes miss.
    # Applied AFTER footprint detection to avoid margin text/annotations.
    adaptive_r = walls.get("adaptive_radius", 2)
    binary_fp = np.zeros_like(walls["binary"])
    binary_fp[by1:by2, bx1:bx2] = walls["binary"][by1:by2, bx1:bx2]
    dist_adapt = cv2.distanceTransform(binary_fp, cv2.DIST_L2, 5)
    adaptive_cores = (dist_adapt >= adaptive_r).astype(np.uint8) * 255
    dk_adapt = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    adaptive_walls = cv2.dilate(adaptive_cores, dk_adapt, iterations=1)
    p3_added = int(np.count_nonzero(cv2.bitwise_and(adaptive_walls, cv2.bitwise_not(wall_mask))))
    wall_mask = cv2.bitwise_or(wall_mask, adaptive_walls)
    _logger.info(f"Adaptive wall pass (r>={adaptive_r}): {p3_added} new wall pixels added within footprint")

    # Also clip the binary image to the building footprint for door detection
    binary_clipped = np.zeros_like(walls["binary"])
    binary_clipped[by1:by2, bx1:bx2] = walls["binary"][by1:by2, bx1:bx2]
    eroded_clipped = np.zeros_like(walls["eroded"])
    eroded_clipped[by1:by2, bx1:bx2] = walls["eroded"][by1:by2, bx1:bx2]

    # Step 7: Door arc detection
    _progress(35, "Detecting doors...")
    doors = _detect_door_arcs(binary_clipped, eroded_clipped)

    # Step 8: Close door gaps
    _progress(45, "Closing door gaps...")
    wall_mask = _close_door_gaps(wall_mask, doors)

    # Step 9: Room detection via sealed flood-fill
    # Seals building perimeter, bridges door gaps with directional closing,
    # then flood-fills from border to identify exterior. Each remaining
    # connected component is a room.
    _progress(55, "Detecting rooms...")
    ds_footprint = (bx1, by1, bx2, by2)
    raw_rooms = _detect_rooms_floodfill(wall_mask, ds_footprint)

    # Step 9b: Subdivide large merged rooms
    # Rooms >3% of building footprint are likely multiple rooms connected
    # through wide doorways. Re-run flood-fill within each one.
    raw_rooms = _subdivide_large_rooms(raw_rooms, wall_mask, ds_footprint)

    # Step 10: Snap polygon vertices to nearest wall pixels
    # Pulls each vertex onto actual wall lines, improving polygon fit
    # without the fragility of flood-fill refinement.
    _progress(65, "Snapping polygons to walls...")
    raw_rooms = _snap_rooms_to_walls(raw_rooms, wall_mask)

    _progress(70, f"Found {len(raw_rooms)} rooms, labeling...")

    # Step 11: Label rooms with Gemini
    # Build temporary room list with upscaled coordinates for labeling
    # (Gemini needs original-resolution image + centroids)
    upscaled_rooms = []
    for raw in raw_rooms:
        poly = raw["polygon"]
        coords = list(poly.exterior.coords)[:-1]
        upscaled_rooms.append({
            "centroid": (raw["centroid"][0] * scale_factor, raw["centroid"][1] * scale_factor),
            "polygon": poly,
        })

    from backend.pipeline.vision_ai import label_rooms
    try:
        labeled = label_rooms(image, upscaled_rooms)
    except Exception as e:
        _logger.warning(f"Gemini labeling failed: {e}, using defaults")
        labeled = [{"name": "Unnamed", "type": "unknown"}] * len(raw_rooms)

    # Step 12: Build result list with upscaled coordinates
    _progress(85, "Building results...")
    all_rooms = []
    sf = scale_factor
    for i, raw in enumerate(raw_rooms):
        label_info = labeled[i] if i < len(labeled) else {}
        poly = raw["polygon"]
        coords = list(poly.exterior.coords)[:-1]
        # Upscale coordinates back to original resolution
        polygon_px = [(int(round(x * sf)), int(round(y * sf))) for x, y in coords]

        # Upscale area and perimeter (area scales by sf², perimeter by sf)
        area_px = raw["area_px"] * (sf * sf)
        perimeter_px = raw["perimeter_px"] * sf
        centroid = (raw["centroid"][0] * sf, raw["centroid"][1] * sf)
        boundary_lengths_px = [l * sf for l in raw["boundary_lengths_px"]]

        all_rooms.append({
            "name": label_info.get("name", "Unnamed"),
            "type": label_info.get("type", "unknown"),
            "unit_name": None,
            "polygon_px": polygon_px,
            "printed_area_sqm": None,
            "area_px": area_px,
            "perimeter_px": perimeter_px,
            "centroid": centroid,
            "boundary_lengths_px": boundary_lengths_px,
            "area_divergence": False,
        })

    # Step 13: Debug images
    if debug_dir:
        _progress(92, "Saving debug images...")
        _save_debug_images(image, wall_mask, all_rooms, debug_dir,
                           doors=doors, scale_factor=scale_factor,
                           footprint=footprint)

    _progress(100, f"Done — {len(all_rooms)} rooms found")
    return all_rooms
