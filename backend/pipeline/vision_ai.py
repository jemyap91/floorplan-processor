"""Vision AI integration for region classification and room labeling using Google Gemini."""
import json
import re
import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# No fallback — Gemini 2 Flash and Flash Lite have 0 free-tier quota
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "")

import time as _time
import logging as _logging
_logger = _logging.getLogger(__name__)


def _call_gemini(image: np.ndarray, prompt: str, model: str | None = None) -> str:
    """Call Google Gemini Vision API with retry on rate limit.

    Uses the new google-genai SDK (replaces deprecated google.generativeai).
    """
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    pil_image = Image.fromarray(image)

    target_model = model or GEMINI_MODEL
    models_to_try = [target_model]
    if not model and GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != target_model:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, pil_image],
                )
                return response.text
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "ResourceExhausted" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(30 * (attempt + 1), 90)
                    _logger.warning(f"Rate limited on {model_name}, retrying in {wait}s...")
                    _time.sleep(wait)
                else:
                    _logger.warning(f"Error with {model_name}: {e}")
                    break  # non-rate-limit error, try next model
    raise last_error


def _call_vision(image: np.ndarray, prompt: str, model: str | None = None, **kwargs) -> str:
    """Call Gemini vision API."""
    return _call_gemini(image, prompt, model=model)

def _build_classification_prompt() -> str:
    return """You are analyzing an architectural floorplan drawing. These drawings typically have:
- A **floorplan area** showing the building layout (rooms, walls, corridors)
- **Non-floorplan areas** in the margins: title blocks, revision tables, legends, notes, key plans, logos, drawing borders, grid labels

Identify bounding boxes for ALL non-floorplan regions that should be EXCLUDED from room detection.
Be aggressive — exclude everything that is not part of the actual building floor layout.
Common regions to exclude:
- Right-side panel with title block, notes, legends, key plans
- Bottom strip with revision tables and drawing info
- Drawing border annotations and grid reference labels
- Any area with dense tabular grids, company logos, or blocks of text

Return JSON with **normalized coordinates** (0.0 to 1.0, where 0,0 is top-left and 1,1 is bottom-right):
```json
{
  "excluded_regions": [
    {"x": 0.7, "y": 0.0, "width": 0.3, "height": 1.0, "type": "title_block"},
    {"x": 0.0, "y": 0.95, "width": 1.0, "height": 0.05, "type": "border"}
  ]
}
```
The "type" field should be one of: "title_block", "legend", "notes", "table", "schedule", "border", "key_plan".
Return ONLY the JSON, no other text."""

def _build_labeling_prompt(room_count: int) -> str:
    return f"""This floorplan image has {room_count} rooms outlined with colored polygons and numbered labels.
For each numbered room, identify:
1. The **room name** (read from text labels inside or near the room boundary)
2. The **room type** (office, bathroom, corridor, meeting_room, kitchen, storage, lobby, elevator, stairwell, utility, or other)
3. Your **confidence** (0.0 to 1.0) in the identification
Return JSON in this exact format:
```json
{{
  "rooms": [
    {{"room_id": 0, "name": "Office 201", "type": "office", "confidence": 0.9}}
  ]
}}
```
Return ONLY the JSON, no other text."""

def _parse_json_response(text: str) -> dict | None:
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

def classify_regions(image: np.ndarray, provider: str | None = None) -> dict:
    h, w = image.shape[:2]
    try:
        prompt = _build_classification_prompt()
        response_text = _call_vision(image, prompt, provider=provider)
        result = _parse_json_response(response_text)
        if result and "excluded_regions" in result:
            # Convert normalized (0-1) coordinates to pixel coordinates
            pixel_regions = []
            for er in result["excluded_regions"]:
                rx = er.get("x", 0)
                ry = er.get("y", 0)
                rw = er.get("width", 0)
                rh = er.get("height", 0)
                # Detect whether coords are already in pixels or normalized
                if rx <= 1.0 and ry <= 1.0 and rw <= 1.0 and rh <= 1.0:
                    rx, ry, rw, rh = rx * w, ry * h, rw * w, rh * h
                pixel_regions.append({
                    "x": int(rx), "y": int(ry),
                    "width": int(rw), "height": int(rh),
                    "type": er.get("type", "unknown"),
                })
            return {
                "floorplan_regions": [{"x": 0, "y": 0, "width": w, "height": h}],
                "excluded_regions": pixel_regions,
            }
    except Exception as e:
        _logger.error(f"classify_regions failed: {e}")
    return {
        "floorplan_regions": [{"x": 0, "y": 0, "width": w, "height": h}],
        "excluded_regions": [],
    }

def _build_room_listing_prompt() -> str:
    return """You are an expert architectural floor plan analyst. List every distinct room and enclosed space visible on this floor plan.

For EACH room, provide:
1. **name**: The room label exactly as written on the plan (e.g. "LIFT 1", "PUBLIC LIFT LOBBY", "STAIR CORE 01"). If unlabeled, describe it (e.g. "Unlabeled corridor north of LIFT 3").
2. **type**: One of: office, bathroom, corridor, meeting_room, kitchen, storage, lobby, elevator, stairwell, utility, mechanical, parking, terrace, other

IMPORTANT:
- ONLY include rooms in the actual building floor plan
- EXCLUDE title blocks, legends, notes, revision tables, key plans, logos, borders
- Include ALL rooms: corridors, lift shafts, lobbies, stairwells, plant rooms, bathrooms, etc.
- Read the EXACT text labels from the drawing — do not paraphrase

Return ONLY valid JSON:
```json
{
  "rooms": [
    {"name": "PUBLIC LIFT LOBBY", "type": "lobby", "confidence": 0.95},
    {"name": "LIFT 1", "type": "elevator", "confidence": 0.95}
  ],
  "scale_text": "1:200 or whatever scale notation you can read, or null"
}
```"""


def extract_room_labels_with_gemini(image: np.ndarray, provider: str | None = None) -> dict:
    """Use vision AI to list all rooms with names/types (no boundaries needed)."""
    try:
        prompt = _build_room_listing_prompt()
        response_text = _call_vision(image, prompt, provider=provider)
        result = _parse_json_response(response_text)
        if result is None:
            result = _repair_truncated_json(response_text)
        if result and "rooms" in result:
            return result
    except Exception as e:
        _logger.error(f"extract_room_labels failed: {e}")
    return {"rooms": [], "scale_text": None}


def match_gemini_labels_to_cv_rooms(
    gemini_rooms: list[dict],
    cv_rooms: list[dict],
    image: np.ndarray,
    provider: str | None = None,
) -> list[dict]:
    """Match Gemini room labels to CV-detected room polygons.

    Since Gemini can't provide precise coordinates, we annotate the image
    with numbered CV room centroids, then ask Gemini to match room names
    to the numbers it sees.
    """
    import cv2

    if not cv_rooms:
        return [{"room_id": i, "name": "Unnamed", "type": "unknown", "confidence": 0.0}
                for i in range(len(cv_rooms))]

    # Draw numbered labels on CV room centroids
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font_scale = max(1.0, w / 3000)
    thickness = max(2, int(w / 2000))
    for i, room in enumerate(cv_rooms):
        cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
        label = str(i)
        # Draw background circle for visibility
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.circle(annotated, (cx, cy), max(text_size[0], text_size[1]) + 10, (255, 255, 0), -1)
        cv2.putText(annotated, label, (cx - text_size[0]//2, cy + text_size[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Ask Gemini to match numbers to room names
    match_prompt = f"""This architectural floor plan has {len(cv_rooms)} numbered yellow circles marking detected room regions (numbered 0 to {len(cv_rooms)-1}).

For each numbered region, identify:
1. The **room name** (read the text label from the drawing inside or near each numbered circle)
2. The **room type** (office, bathroom, corridor, meeting_room, kitchen, storage, lobby, elevator, stairwell, utility, mechanical, parking, terrace, other)
3. Your **confidence** (0.0 to 1.0)

If a numbered region is NOT a real room (e.g. it's an outer boundary, margin area, or annotation), set the name to "NOT_A_ROOM".

Return ONLY valid JSON:
```json
{{
  "rooms": [
    {{"room_id": 0, "name": "PUBLIC LIFT LOBBY", "type": "lobby", "confidence": 0.9}},
    {{"room_id": 1, "name": "NOT_A_ROOM", "type": "other", "confidence": 0.8}}
  ]
}}
```"""

    try:
        response_text = _call_vision(annotated, match_prompt, provider=provider)
        _logger.info(f"Gemini labeling response length: {len(response_text)}")
        result = _parse_json_response(response_text)
        if result is None:
            result = _repair_truncated_json(response_text)
        if result and "rooms" in result:
            return result["rooms"]
        _logger.warning(f"Could not parse Gemini labeling response: {response_text[:500]}")
    except Exception as e:
        _logger.error(f"match_gemini_labels_to_cv_rooms failed: {e}")

    return [{"room_id": i, "name": "Unnamed", "type": "unknown", "confidence": 0.0}
            for i in range(len(cv_rooms))]


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair a truncated JSON response from Gemini."""
    json_start = text.find('{"rooms"')
    if json_start == -1:
        match = re.search(r"```(?:json)?\s*\n?", text)
        if match:
            json_start = match.end()
    if json_start == -1:
        return None
    fragment = text[json_start:].rstrip("`\n ")
    last_complete = fragment.rfind("},")
    if last_complete != -1:
        repaired = fragment[: last_complete + 1] + "]}"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


def label_rooms(image: np.ndarray, rooms: list, provider: str | None = None) -> list[dict]:
    try:
        import cv2
        annotated = image.copy()
        for i, room in enumerate(rooms):
            cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
            cv2.putText(annotated, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        prompt = _build_labeling_prompt(room_count=len(rooms))
        response_text = _call_vision(annotated, prompt, provider=provider)
        result = _parse_json_response(response_text)
        if result and "rooms" in result:
            return result["rooms"]
    except Exception as e:
        _logger.error(f"label_rooms failed: {e}")
    return [{"room_id": i, "name": "Unnamed", "type": "unknown", "confidence": 0.0} for i in range(len(rooms))]
