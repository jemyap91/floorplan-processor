"""Furnished floorplan analyzer — two-pass Gemini pipeline.

Pass 1: Identify apartment units and public spaces (bounding boxes).
Pass 2: For each residential unit, extract room polygons from a cropped image.
Public spaces skip Pass 2 and use their bounding box as a rectangle polygon.
"""
import json
import re
import logging
import os
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

_logger = logging.getLogger(__name__)

# Types that are considered public (skip Pass 2)
PUBLIC_TYPES = {"public", "lobby", "stairwell", "corridor", "utility", "mechanical", "elevator"}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | None:
    """Parse JSON from Gemini response, handling ```json``` fenced blocks."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    return None


def _repair_truncated_json(text: str, key: str) -> dict | None:
    """Repair a truncated JSON response by finding the last complete object in an array."""
    marker = f'{{"{key}"'
    json_start = text.find(marker)
    if json_start == -1:
        match = re.search(r"```(?:json)?\s*\n?", text)
        if match:
            json_start = match.end()
    if json_start == -1:
        return None
    fragment = text[json_start:].rstrip("`\n ")
    last_complete = fragment.rfind("},")
    if last_complete != -1:
        repaired = fragment[:last_complete + 1] + "]}"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    # Try closing with just ]}
    last_brace = fragment.rfind("}")
    if last_brace != -1:
        repaired = fragment[:last_brace + 1] + "]}"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Gemini call wrapper
# ---------------------------------------------------------------------------

def _call_gemini_with_model(image: np.ndarray, prompt: str, model: str | None = None) -> str:
    """Thin wrapper around vision_ai._call_gemini."""
    from backend.pipeline.vision_ai import _call_gemini
    return _call_gemini(image, prompt, model=model)


# ---------------------------------------------------------------------------
# Pass 1 — unit detection
# ---------------------------------------------------------------------------

def _build_pass1_prompt() -> str:
    """Build Gemini prompt for identifying apartment units and public spaces."""
    return """You are an expert architectural floor plan analyzer specialized in computer vision and spatial layout extraction.

Task: Identify ALL distinct apartment units and public/shared spaces in this floor plan.

Core Instructions:
- Structural Boundary Priority: Identify boundaries based strictly on structural wall lines (solid lines for full walls, dashed for pony walls/openings).
- Noise Filtering (Crucial): Ignore all non-structural elements. Do NOT be distracted by dimension lines, arrows, measurement text, furniture icons, appliance symbols, text labels, or hatching patterns.

For each unit/space, provide:
1. **name**: The unit label as written on the plan (e.g. "UNIT 20.01"). For public areas use the label (e.g. "Lobby", "Stair E").
2. **type**: One of: residential, public, lobby, stairwell, corridor, utility, mechanical, elevator
3. **box_2d**: Bounding box as [ymin, xmin, ymax, xmax] with values from 0 to 1000.

CRITICAL:
- The box must enclose the ENTIRE FLOOR AREA of the unit — all its rooms, walls, corridors, and balconies.
- Do NOT just box the unit's text label. Box the full physical space the unit occupies.
- Residential units are LARGE areas containing multiple rooms (bedrooms, kitchen, bathrooms, living areas).
- Do NOT include title blocks, legends, notes, or margins.
- Bboxes for adjacent units should NOT significantly overlap.

Return ONLY valid JSON:
```json
{
  "units": [
    {"name": "UNIT 20.01", "type": "residential", "box_2d": [100, 50, 500, 400]},
    {"name": "Lobby", "type": "lobby", "box_2d": [300, 450, 500, 600]}
  ]
}
```"""


def _parse_pass1_response(response_text: str, img_w: int, img_h: int) -> list[dict]:
    """Parse Pass 1 response, converting 0-1000 coords to pixel coords.

    Gemini returns [ymin, xmin, ymax, xmax] in 0-1000 scale.
    Returns list of dicts with keys: name, type, bbox_px (x, y, w, h in pixels), is_public.
    """
    result = _parse_json(response_text)
    if result is None:
        result = _repair_truncated_json(response_text, "units")
    if result is None or "units" not in result:
        return []

    units = []
    for u in result["units"]:
        name = u.get("name", "Unknown")
        utype = u.get("type", "unknown")
        box = u.get("box_2d", [0, 0, 0, 0])
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        # Convert 0-1000 to pixel coordinates
        x = int(round(xmin / 1000 * img_w))
        y = int(round(ymin / 1000 * img_h))
        w = int(round((xmax - xmin) / 1000 * img_w))
        h = int(round((ymax - ymin) / 1000 * img_h))
        if w <= 0 or h <= 0:
            continue
        is_public = utype.lower() in PUBLIC_TYPES
        units.append({
            "name": name,
            "type": utype,
            "bbox_px": (x, y, w, h),
            "is_public": is_public,
        })
    return units


def analyze_units(image: np.ndarray, model: str | None = None) -> list[dict]:
    """Pass 1: Call Gemini to identify apartment units and public spaces."""
    try:
        prompt = _build_pass1_prompt()
        response_text = _call_gemini_with_model(image, prompt, model=model)
        h, w = image.shape[:2]
        return _parse_pass1_response(response_text, img_w=w, img_h=h)
    except Exception as e:
        _logger.error(f"analyze_units failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Pass 2 — room polygons per unit
# ---------------------------------------------------------------------------

def _build_pass2_prompt(unit_name: str) -> str:
    """Build Gemini prompt for extracting room bounding boxes within a cropped unit image."""
    return f"""You are an expert architectural floor plan analyzer specialized in computer vision and spatial layout extraction.

Task: Perform semantic segmentation on this cropped floor plan section showing "{unit_name}" to identify all functional room areas.

Core Instructions:
- Structural Boundary Priority: Identify room boundaries based strictly on structural wall lines (solid lines for full walls, dashed for pony walls/openings).
- Noise Filtering (Crucial): Ignore all non-structural elements. Do NOT be distracted by dimension lines, arrows, measurement text numbers, furniture icons (beds, sofas, tables), appliance symbols, text labels, or hatching patterns.

For each room, provide:
1. **name**: The room label as written (e.g. "Master Bedroom", "Kitchen", "Bathroom 1")
2. **type**: One of: bedroom, bathroom, kitchen, living_room, dining_room, balcony, corridor, storage, utility, study, laundry, other
3. **box_2d**: Bounding box as [ymin, xmin, ymax, xmax] with values from 0 to 1000. The box must fit within the inner faces of the structural walls.
4. **area_sqm**: The printed area in square meters if visible on the plan, or null.

CRITICAL:
- Include ALL rooms: bedrooms, bathrooms, kitchen, living areas, balconies, corridors, storage
- Boxes must align with structural wall lines, NOT furniture edges
- For open-concept spaces (living/dining/kitchen with no separating wall), return ONE merged box
- Door openings separate rooms — each side of a doorway is a different room

Return ONLY valid JSON:
```json
{{
  "rooms": [
    {{"name": "Master Bedroom", "type": "bedroom", "box_2d": [50, 50, 450, 500], "area_sqm": 15.2}},
    {{"name": "Kitchen", "type": "kitchen", "box_2d": [0, 500, 400, 950], "area_sqm": null}}
  ]
}}
```"""


def _parse_pass2_response(response_text: str, crop_w: int, crop_h: int) -> list[dict]:
    """Parse Pass 2 response, converting 0-1000 box coords to crop-pixel coords.

    Gemini returns [ymin, xmin, ymax, xmax] in 0-1000 scale.
    Returns list of dicts with keys: name, type, polygon_px (4-point rect), printed_area_sqm.
    """
    result = _parse_json(response_text)
    if result is None:
        result = _repair_truncated_json(response_text, "rooms")
    if result is None or "rooms" not in result:
        return []

    rooms = []
    for r in result["rooms"]:
        box = r.get("box_2d", [])
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        # Convert 0-1000 to pixel coordinates
        x1 = int(round(xmin / 1000 * crop_w))
        y1 = int(round(ymin / 1000 * crop_h))
        x2 = int(round(xmax / 1000 * crop_w))
        y2 = int(round(ymax / 1000 * crop_h))
        if x2 <= x1 or y2 <= y1:
            continue
        # Convert bbox to 4-point polygon
        polygon_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rooms.append({
            "name": r.get("name", "Unknown"),
            "type": r.get("type", "other"),
            "polygon_px": polygon_px,
            "printed_area_sqm": r.get("area_sqm"),
        })
    return rooms


def analyze_rooms_in_unit(image: np.ndarray, unit: dict, model: str | None = None) -> list[dict]:
    """Pass 2: Crop unit from image, call Gemini for room polygons, transform coords back.

    Args:
        image: Full floorplan image (numpy array).
        unit: Dict with keys name, bbox_px (x, y, w, h), is_public.
        model: Optional Gemini model name.

    Returns:
        List of room dicts with polygon_px in full-image coordinates and unit_name set.
    """
    try:
        x, y, w, h = unit["bbox_px"]
        img_h, img_w = image.shape[:2]
        _logger.info(f"Pass 2: analyzing rooms in '{unit['name']}' bbox=({x},{y},{w},{h})")
        # Crop the unit region
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        crop = image[y:y2, x:x2]
        crop_h, crop_w = crop.shape[:2]
        if crop_w <= 0 or crop_h <= 0:
            return []

        prompt = _build_pass2_prompt(unit["name"])
        response_text = _call_gemini_with_model(crop, prompt, model=model)
        rooms = _parse_pass2_response(response_text, crop_w=crop_w, crop_h=crop_h)

        # Offset polygon coords from crop space to full-image space
        for room in rooms:
            room["polygon_px"] = [(px + x, py + y) for (px, py) in room["polygon_px"]]
            room["unit_name"] = unit["name"]

        return rooms
    except Exception as e:
        _logger.error(f"analyze_rooms_in_unit failed for {unit.get('name', '?')}: {e}")
        return []


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_room_geometry(polygon_px: list[tuple]) -> dict:
    """Compute area, perimeter, and centroid from a pixel-coordinate polygon using Shapely."""
    poly = ShapelyPolygon(polygon_px)
    centroid = poly.centroid
    return {
        "area_px": poly.area,
        "perimeter_px": poly.length,
        "centroid": (centroid.x, centroid.y),
    }


def _check_area_divergence(
    area_px: float,
    printed_area_sqm: float | None,
    px_per_meter: float | None,
    threshold: float = 0.3,
) -> bool:
    """Check if polygon area diverges from printed area by more than threshold.

    Returns True if divergence exceeds threshold, False otherwise.
    Returns False if printed_area_sqm or px_per_meter is None (can't compare).
    """
    if printed_area_sqm is None or px_per_meter is None:
        return False
    if px_per_meter <= 0 or printed_area_sqm <= 0:
        return False
    computed_sqm = area_px / (px_per_meter ** 2)
    divergence = abs(computed_sqm - printed_area_sqm) / printed_area_sqm
    return divergence > threshold


def _unit_bbox_to_polygon(unit: dict, img_w: int, img_h: int) -> list[tuple]:
    """Convert a unit's bbox_px (x, y, w, h) to a 4-point rectangle polygon."""
    x, y, w, h = unit["bbox_px"]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


# ---------------------------------------------------------------------------
# Debug images
# ---------------------------------------------------------------------------

def _save_debug_images(image: np.ndarray, units: list[dict], rooms: list[dict], debug_dir: str):
    """Save annotated debug images showing unit bboxes and room polygons."""
    import cv2

    os.makedirs(debug_dir, exist_ok=True)

    # Image 1: unit bounding boxes
    img_units = image.copy()
    for u in units:
        x, y, w, h = u["bbox_px"]
        color = (0, 255, 0) if not u.get("is_public") else (255, 165, 0)
        cv2.rectangle(img_units, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_units, u["name"], (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(os.path.join(debug_dir, "debug_units.png"), img_units)

    # Image 2: room polygons with labels
    img_rooms = image.copy()
    for r in rooms:
        pts = np.array(r["polygon_px"], dtype=np.int32)
        cv2.polylines(img_rooms, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        if len(pts) > 0:
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = f"{r.get('unit_name', '')}: {r['name']}"
            cv2.putText(img_rooms, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_dir, "debug_rooms.png"), img_rooms)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _match_gemini_rooms_to_cv_contours(
    gemini_rooms: list[dict],
    cv_rooms: list[dict],
    unit_offset: tuple[int, int],
) -> list[dict]:
    """Match Gemini-identified rooms to CV-detected contours by overlap.

    For each Gemini room (bounding box), find the CV contour whose centroid
    falls inside the Gemini box. If matched, use the CV contour's polygon
    (precise walls) but keep Gemini's name/type/area metadata.

    Args:
        gemini_rooms: Rooms from Pass 2 with polygon_px (4-point bbox in full-image coords).
        cv_rooms: Rooms from CV segmenter with polygon (Shapely), centroid, etc. in crop coords.
        unit_offset: (x, y) offset of the unit crop in the full image.

    Returns:
        Merged room list with CV polygons where matched, Gemini boxes as fallback.
    """
    from shapely.geometry import box as shapely_box, Point

    ox, oy = unit_offset
    matched_cv = set()
    result = []

    for gr in gemini_rooms:
        pts = gr["polygon_px"]  # 4-point bbox in full-image coords
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        gemini_box = shapely_box(min(xs), min(ys), max(xs), max(ys))

        best_cv_idx = None
        best_overlap = 0

        for ci, cv_room in enumerate(cv_rooms):
            if ci in matched_cv:
                continue
            # CV centroid is in crop coords — offset to full image
            cx = cv_room["centroid"][0] + ox
            cy = cv_room["centroid"][1] + oy
            if gemini_box.contains(Point(cx, cy)):
                # Use area as tiebreaker — prefer larger CV room
                overlap = cv_room["area_px"]
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cv_idx = ci

        if best_cv_idx is not None:
            matched_cv.add(best_cv_idx)
            cv_room = cv_rooms[best_cv_idx]
            # Use CV polygon (precise walls) but Gemini metadata
            cv_coords = list(cv_room["polygon"].exterior.coords)[:-1]
            polygon_px = [(int(round(x + ox)), int(round(y + oy))) for x, y in cv_coords]
            result.append({
                "name": gr.get("name", "Unknown"),
                "type": gr.get("type", "other"),
                "unit_name": gr.get("unit_name"),
                "polygon_px": polygon_px,
                "printed_area_sqm": gr.get("printed_area_sqm"),
            })
        else:
            # No CV match — use Gemini bbox as fallback
            result.append(gr)

    return result


def _cv_segment_unit(image: np.ndarray, unit: dict) -> list[dict]:
    """Run CV wall detection + room segmentation on a unit crop.

    Returns list of room dicts from the room segmenter (in crop coordinates).
    """
    import cv2

    x, y, w, h = unit["bbox_px"]
    img_h, img_w = image.shape[:2]
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)
    crop = image[y:y2, x:x2]

    if crop.size == 0:
        return []

    from backend.pipeline.preprocessor import preprocess_image
    from backend.pipeline.wall_detector import detect_walls
    from backend.pipeline.room_segmenter import segment_rooms

    prep = preprocess_image(crop)
    walls = detect_walls(prep["binary"])
    rooms = segment_rooms(
        walls["wall_mask"],
        min_area_px=200,
        min_area_ratio=0.005,
        min_compactness=0.02,
        close_gap_px=7,
    )
    return rooms


def run_furnished_pipeline(
    image: np.ndarray,
    flash_model: str = "gemini-2.5-flash",
    detail_model: str | None = None,
    px_per_meter: float | None = None,
    debug_dir: str | None = None,
    progress_cb=None,
) -> list[dict]:
    """Run the hybrid Gemini+CV furnished floorplan analysis pipeline.

    Gemini identifies rooms (names, types, approximate locations).
    CV detects precise wall-aligned contours.
    Rooms are matched by spatial overlap — CV polygons with Gemini labels.

    Args:
        image: Full floorplan image (BGR numpy array).
        flash_model: Gemini model for Pass 1 (unit detection).
        detail_model: Gemini model for Pass 2 (room identification). Defaults to flash_model.
        px_per_meter: Scale factor for area computation.
        debug_dir: Directory path for saving debug images (None to skip).
        progress_cb: Optional callback(percent, message) for progress reporting.

    Returns:
        List of room dicts, each with: name, type, unit_name, polygon_px,
        printed_area_sqm, area_px, perimeter_px, centroid, area_divergence.
    """
    if detail_model is None:
        detail_model = flash_model

    _logger.info(f"Starting furnished pipeline: flash={flash_model}, detail={detail_model}")

    def _progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    # Pass 1: detect units
    _progress(5, "Detecting apartment units...")
    units = analyze_units(image, model=flash_model)
    if not units:
        _progress(100, "No units detected")
        return []

    _progress(20, f"Found {len(units)} units")
    img_h, img_w = image.shape[:2]

    all_rooms = []
    residential_units = [u for u in units if not u["is_public"]]
    public_units = [u for u in units if u["is_public"]]
    total_work = len(residential_units) + 1  # +1 for pass1

    # Pass 2: for each residential unit, get Gemini rooms + CV contours, then match
    for i, unit in enumerate(residential_units):
        pct = 20 + int(60 * (i + 1) / total_work)
        _progress(pct, f"Analyzing {unit['name']}...")

        # Gemini: room names, types, approximate bounding boxes
        gemini_rooms = analyze_rooms_in_unit(image, unit, model=detail_model)

        # CV: precise wall-aligned contours
        cv_rooms = _cv_segment_unit(image, unit)
        _logger.info(f"  {unit['name']}: Gemini found {len(gemini_rooms)} rooms, CV found {len(cv_rooms)} contours")

        # Match: pair Gemini metadata with CV polygons
        unit_offset = (unit["bbox_px"][0], unit["bbox_px"][1])
        matched = _match_gemini_rooms_to_cv_contours(gemini_rooms, cv_rooms, unit_offset)
        all_rooms.extend(matched)

    # Public spaces: use bbox as rectangle polygon, skip Pass 2
    for unit in public_units:
        poly = _unit_bbox_to_polygon(unit, img_w, img_h)
        all_rooms.append({
            "name": unit["name"],
            "type": unit["type"],
            "unit_name": "Public Space",
            "polygon_px": poly,
            "printed_area_sqm": None,
        })

    # Compute geometry for all rooms
    _progress(85, "Computing room geometry...")
    for room in all_rooms:
        geom = _compute_room_geometry(room["polygon_px"])
        room["area_px"] = geom["area_px"]
        room["perimeter_px"] = geom["perimeter_px"]
        room["centroid"] = geom["centroid"]
        room["area_divergence"] = _check_area_divergence(
            geom["area_px"],
            room.get("printed_area_sqm"),
            px_per_meter,
        )

    # Debug images
    if debug_dir:
        _progress(95, "Saving debug images...")
        _save_debug_images(image, units, all_rooms, debug_dir)

    _progress(100, f"Done — {len(all_rooms)} rooms found")
    return all_rooms
