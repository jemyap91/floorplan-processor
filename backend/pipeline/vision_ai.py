"""Gemini Flash integration for region classification and room labeling."""
import json
import re
import os
import numpy as np
from PIL import Image
import io

def _call_gemini(image: np.ndarray, prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    pil_image = Image.fromarray(image)
    response = model.generate_content([prompt, pil_image])
    return response.text

def _build_classification_prompt() -> str:
    return """Analyze this architectural floorplan image. Identify and return bounding boxes for:
1. **floorplan_regions**: Areas containing the actual building floor layout (rooms, corridors, walls).
2. **excluded_regions**: Areas containing tables, legends, schedules, title blocks, or any non-floorplan content.
Return JSON in this exact format:
```json
{
  "floorplan_regions": [{"x": 0, "y": 0, "width": 100, "height": 100}],
  "excluded_regions": [{"x": 200, "y": 0, "width": 50, "height": 50, "type": "table"}]
}
```
Coordinates are in pixels from the top-left corner. The "type" field should be one of: "table", "legend", "title_block", "schedule".
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

def classify_regions(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    try:
        prompt = _build_classification_prompt()
        response_text = _call_gemini(image, prompt)
        result = _parse_json_response(response_text)
        if result and "floorplan_regions" in result:
            return result
    except Exception:
        pass
    return {
        "floorplan_regions": [{"x": 0, "y": 0, "width": w, "height": h}],
        "excluded_regions": [],
    }

def label_rooms(image: np.ndarray, rooms: list) -> list[dict]:
    try:
        import cv2
        annotated = image.copy()
        for i, room in enumerate(rooms):
            cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
            cv2.putText(annotated, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        prompt = _build_labeling_prompt(room_count=len(rooms))
        response_text = _call_gemini(annotated, prompt)
        result = _parse_json_response(response_text)
        if result and "rooms" in result:
            return result["rooms"]
    except Exception:
        pass
    return [{"room_id": i, "name": "Unnamed", "type": "unknown", "confidence": 0.0} for i in range(len(rooms))]
