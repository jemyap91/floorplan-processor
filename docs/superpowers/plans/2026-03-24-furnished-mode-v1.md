# Furnished Processing Mode (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Furnished" processing mode that uses a two-pass Gemini approach to detect individual rooms in furnished residential floorplans, with unit tagging and area validation.

**Architecture:** Pass 1 (Gemini Flash) identifies apartment units and public spaces with bounding boxes. Pass 2 (Gemini Flash or Pro) crops each unit and extracts detailed room polygons. Rooms are tagged with their parent unit name. v1 delivers raw Gemini polygons without wall snapping.

**Tech Stack:** Python 3.13 (anaconda), FastAPI, google-genai SDK, React + TypeScript, TailwindCSS, Fabric.js v6, SQLite

**PRD:** `docs/prd-furnished-mode.md`

---

## File Structure

### New Files
- `backend/pipeline/furnished_analyzer.py` — Two-pass Gemini orchestrator (prompts, parsing, coordinate transform)
- `backend/tests/pipeline/test_furnished_analyzer.py` — Tests for furnished analyzer (mocked Gemini)

### Modified Files
- `backend/models/room.py` — Add `unit_name`, `printed_area_sqm`, `area_divergence_flag` fields
- `backend/database.py` — Add 3 new columns to rooms table, update save/get
- `backend/pipeline/vision_ai.py` — Add `model` param to `_call_gemini()`
- `backend/main.py` — Add `_process_furnished_mode()`, accept `gemini_model` param
- `frontend/src/api.ts` — Add `'furnished'` to ProcessMode, add `geminiModel` param
- `frontend/src/App.tsx` — Add Furnished mode button, model selector dropdown
- `frontend/src/components/RoomSidebar.tsx` — Show `unit_name` under room name
- `frontend/src/components/RoomDetail.tsx` — Show `printed_area_sqm`, divergence flag, unit name

---

### Task 1: Add new fields to RoomData model

**Files:**
- Modify: `backend/models/room.py:7-22`
- Test: `backend/tests/test_models.py`

- [ ] **Step 1: Write failing tests for new fields**

Add to `backend/tests/test_models.py`:

```python
def test_room_unit_name_field(self):
    room = RoomData(
        id="room-1", name="Master Bedroom", room_type="bedroom",
        boundary_polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
        area_px=5000.0, perimeter_px=300.0, centroid=(50.0, 25.0),
        boundary_lengths_px=[100.0, 50.0, 100.0, 50.0],
        unit_name="UNIT 20.01",
        printed_area_sqm=15.5,
        area_divergence_flag=True,
    )
    assert room.unit_name == "UNIT 20.01"
    assert room.printed_area_sqm == 15.5
    assert room.area_divergence_flag is True
    d = room.model_dump()
    assert d["unit_name"] == "UNIT 20.01"
    assert d["printed_area_sqm"] == 15.5
    assert d["area_divergence_flag"] is True

def test_room_new_fields_default_values(self):
    room = RoomData(id="room-2", name="Test")
    assert room.unit_name is None
    assert room.printed_area_sqm is None
    assert room.area_divergence_flag is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_models.py -v`
Expected: FAIL — `unit_name`, `printed_area_sqm`, `area_divergence_flag` not defined on RoomData

- [ ] **Step 3: Add new fields to RoomData**

In `backend/models/room.py`, add three fields after line 22 (after `confidence`):

```python
    unit_name: Optional[str] = None
    printed_area_sqm: Optional[float] = None
    area_divergence_flag: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_models.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/models/room.py backend/tests/test_models.py
git commit -m "feat: add unit_name, printed_area_sqm, area_divergence_flag to RoomData"
```

---

### Task 2: Update database schema for new fields

**Files:**
- Modify: `backend/database.py:12-49` (table creation), `backend/database.py:72-101` (save/get rooms)
- Test: `backend/tests/test_database.py`

- [ ] **Step 1: Write failing test for new columns**

Add to `backend/tests/test_database.py`:

```python
def test_save_and_get_room_with_unit_fields(self):
    from backend.models.room import RoomData, ProjectData
    db = Database(":memory:")
    project = ProjectData(id="p1", name="Test")
    db.save_project(project)
    room = RoomData(
        id="r1", project_id="p1", name="Bedroom",
        boundary_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        area_px=100.0, perimeter_px=40.0, centroid=(5.0, 5.0),
        boundary_lengths_px=[10.0, 10.0, 10.0, 10.0],
        unit_name="UNIT 20.01",
        printed_area_sqm=12.5,
        area_divergence_flag=True,
    )
    db.save_room(room)
    rooms = db.get_rooms("p1")
    assert len(rooms) == 1
    assert rooms[0].unit_name == "UNIT 20.01"
    assert rooms[0].printed_area_sqm == 12.5
    assert rooms[0].area_divergence_flag is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_database.py::test_save_and_get_room_with_unit_fields -v`
Expected: FAIL — columns don't exist

- [ ] **Step 3: Update database schema and save/get methods**

In `backend/database.py`:

1. Add migration block in `_create_tables()` (after the existing `fill_color_rgb` migration, around line 19):

```python
for col, coltype in [("unit_name", "TEXT"), ("printed_area_sqm", "REAL"), ("area_divergence_flag", "INTEGER")]:
    try:
        self.conn.execute(f"SELECT {col} FROM rooms LIMIT 1")
    except sqlite3.OperationalError:
        try:
            self.conn.execute(f"ALTER TABLE rooms ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass
```

2. Update the `CREATE TABLE rooms` statement to include the new columns:

```sql
unit_name TEXT, printed_area_sqm REAL, area_divergence_flag INTEGER
```

3. Replace the entire `save_room()` method with the updated version (19 columns, 19 placeholders):

```python
def save_room(self, room: RoomData):
    self.conn.execute(
        """INSERT OR REPLACE INTO rooms
           (id, project_id, name, room_type, boundary_polygon,
            area_px, perimeter_px, area_sqm, perimeter_m,
            boundary_lengths_px, boundary_lengths_m,
            centroid_x, centroid_y, fill_color_rgb, source, confidence,
            unit_name, printed_area_sqm, area_divergence_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (room.id, room.project_id, room.name, room.room_type,
         json.dumps(room.boundary_polygon), room.area_px, room.perimeter_px,
         room.area_sqm, room.perimeter_m,
         json.dumps(room.boundary_lengths_px),
         json.dumps(room.boundary_lengths_m) if room.boundary_lengths_m else None,
         room.centroid[0], room.centroid[1],
         json.dumps(room.fill_color_rgb) if room.fill_color_rgb else None,
         room.source, room.confidence,
         room.unit_name, room.printed_area_sqm, int(room.area_divergence_flag)))
    self.conn.commit()
```

4. Update `get_rooms()` — read the 3 new columns:

```python
unit_name=r["unit_name"],
printed_area_sqm=r["printed_area_sqm"],
area_divergence_flag=bool(r["area_divergence_flag"]) if r["area_divergence_flag"] else False,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_database.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/database.py backend/tests/test_database.py
git commit -m "feat: add unit_name, printed_area_sqm, area_divergence_flag columns to rooms table"
```

---

### Task 3: Add model parameter to Gemini call function

**Files:**
- Modify: `backend/pipeline/vision_ai.py:21-58`

- [ ] **Step 1: Update `_call_gemini` to accept an explicit model parameter**

In `backend/pipeline/vision_ai.py`, change the signature of `_call_gemini` (line 21):

```python
def _call_gemini(image: np.ndarray, prompt: str, model: str | None = None) -> str:
```

At the top of the function body, resolve the model name:

```python
    target_model = model or GEMINI_MODEL
    models_to_try = [target_model]
    if not model and GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != target_model:
        models_to_try.append(GEMINI_FALLBACK_MODEL)
```

Remove the existing `models_to_try` construction (lines 35-37) since it's replaced above.

- [ ] **Step 2: Update `_call_vision` to pass model through**

```python
def _call_vision(image: np.ndarray, prompt: str, model: str | None = None, **kwargs) -> str:
    return _call_gemini(image, prompt, model=model)
```

- [ ] **Step 3: Run existing vision_ai tests to verify no regression**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_vision_ai.py -v`
Expected: ALL PASS (no behavior change for callers that don't pass `model`)

- [ ] **Step 4: Commit**

```bash
git add backend/pipeline/vision_ai.py
git commit -m "feat: add explicit model parameter to _call_gemini for Flash/Pro selection"
```

---

### Task 4: Build furnished analyzer module — Pass 1 (unit detection)

**Files:**
- Create: `backend/pipeline/furnished_analyzer.py`
- Create: `backend/tests/pipeline/test_furnished_analyzer.py`

- [ ] **Step 1: Write failing tests for Pass 1**

Create `backend/tests/pipeline/test_furnished_analyzer.py`:

```python
import json
import numpy as np
import pytest
from unittest.mock import patch

from backend.pipeline.furnished_analyzer import (
    _build_pass1_prompt,
    _parse_pass1_response,
    analyze_units,
)


class TestPass1Prompt:
    def test_prompt_is_string_with_key_terms(self):
        prompt = _build_pass1_prompt()
        assert isinstance(prompt, str)
        assert "unit" in prompt.lower()
        assert "bounding box" in prompt.lower() or "bbox" in prompt.lower()
        assert "public" in prompt.lower()


class TestPass1Parsing:
    def test_parse_valid_response(self):
        response = json.dumps({
            "units": [
                {
                    "name": "UNIT 20.01",
                    "type": "residential",
                    "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.4},
                    "printed_area_sqm": 86.0,
                },
                {
                    "name": "PUBLIC LIFT LOBBY",
                    "type": "public",
                    "bbox": {"x": 0.05, "y": 0.3, "width": 0.1, "height": 0.15},
                    "printed_area_sqm": None,
                },
            ]
        })
        result = _parse_pass1_response(response, img_w=9934, img_h=7016)
        assert len(result) == 2
        assert result[0]["name"] == "UNIT 20.01"
        assert result[0]["is_public"] is False
        assert result[0]["printed_area_sqm"] == 86.0
        # bbox should be in pixel coordinates
        assert result[0]["bbox_px"][0] == int(0.1 * 9934)  # x
        assert result[1]["name"] == "PUBLIC LIFT LOBBY"
        assert result[1]["is_public"] is True

    def test_parse_truncated_response(self):
        response = '```json\n{"units": [{"name": "UNIT 1", "type": "residential", "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.4}, "printed_area_sqm": 50},'
        result = _parse_pass1_response(response, img_w=1000, img_h=1000)
        assert len(result) >= 1

    def test_parse_invalid_response(self):
        result = _parse_pass1_response("not json", img_w=1000, img_h=1000)
        assert result == []


class TestAnalyzeUnits:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_returns_units(self, mock_call):
        mock_call.return_value = json.dumps({
            "units": [
                {
                    "name": "UNIT 20.01",
                    "type": "residential",
                    "bbox": {"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.4},
                    "printed_area_sqm": 86.0,
                }
            ]
        })
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = analyze_units(img)
        assert len(result) == 1
        assert result[0]["name"] == "UNIT 20.01"
        mock_call.assert_called_once()

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_api_failure_returns_empty(self, mock_call):
        mock_call.side_effect = Exception("API error")
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = analyze_units(img)
        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement Pass 1 in furnished_analyzer.py**

Create `backend/pipeline/furnished_analyzer.py`:

```python
"""Furnished residential floorplan analyzer using two-pass Gemini approach.

Pass 1: Identify apartment units and public spaces with bounding boxes.
Pass 2: For each unit, extract detailed room polygons.
"""
import json
import logging
import re
import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------

def _call_gemini_with_model(image: np.ndarray, prompt: str, model: str | None = None) -> str:
    """Call Gemini vision API with optional model override."""
    from backend.pipeline.vision_ai import _call_gemini
    return _call_gemini(image, prompt, model=model)


def _parse_json(text: str) -> dict | None:
    """Parse JSON from a Gemini response (handles ```json``` blocks)."""
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


def _repair_truncated_json(text: str, key: str = "units") -> dict | None:
    """Attempt to repair truncated JSON array responses."""
    json_start = text.find(f'"{key}"')
    if json_start == -1:
        match = re.search(r"```(?:json)?\s*\n?", text)
        if match:
            json_start = match.end()
    if json_start == -1:
        return None
    # Find the start of the outer object
    obj_start = text.rfind("{", 0, json_start)
    if obj_start == -1:
        obj_start = 0
    fragment = text[obj_start:].rstrip("`\n ")
    last_complete = fragment.rfind("},")
    if last_complete != -1:
        repaired = fragment[: last_complete + 1] + "]}"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Pass 1: Unit detection
# ---------------------------------------------------------------------------

PUBLIC_TYPES = {"public", "lobby", "stairwell", "corridor", "utility", "mechanical", "elevator"}


def _build_pass1_prompt() -> str:
    return """You are analyzing an architectural floor plan of a residential building. This floor plan shows multiple apartment units and shared public spaces on one floor.

Identify EVERY distinct zone on this floor plan:
1. **Apartment units** — labeled areas like "UNIT 20.01", "UNIT 20.02", etc. These contain bedrooms, bathrooms, kitchens, living areas.
2. **Public spaces** — shared areas like lift lobbies, stairwells, corridors, fire hydrant rooms, utility rooms.

For each zone, provide:
- **name**: The label exactly as written on the plan (e.g., "UNIT 20.01", "PUBLIC LIFT LOBBY", "STAIRWELL E")
- **type**: "residential" for apartment units, "public" for shared spaces
- **bbox**: Normalized bounding box (0.0 to 1.0, where 0,0 is top-left). This should tightly enclose the zone's walls.
- **printed_area_sqm**: The area in m² printed on the drawing for this zone (read from text labels like "86 m²"). null if not visible.

IMPORTANT:
- EXCLUDE title blocks, revision tables, legends, notes, borders, key plans — only include actual building spaces
- Include ALL units and public spaces visible on the plan
- Read area values EXACTLY as printed (e.g., 86, not 86.0)
- The bbox should be tight around each zone, not overlapping with neighbors

Return ONLY valid JSON:
```json
{
  "units": [
    {"name": "UNIT 20.01", "type": "residential", "bbox": {"x": 0.1, "y": 0.15, "width": 0.25, "height": 0.35}, "printed_area_sqm": 86},
    {"name": "PUBLIC LIFT LOBBY", "type": "public", "bbox": {"x": 0.05, "y": 0.3, "width": 0.08, "height": 0.12}, "printed_area_sqm": null}
  ]
}
```"""


def _parse_pass1_response(response_text: str, img_w: int, img_h: int) -> list[dict]:
    """Parse Pass 1 response into list of unit dicts with pixel bboxes."""
    result = _parse_json(response_text)
    if result is None:
        result = _repair_truncated_json(response_text, "units")
    if not result or "units" not in result:
        return []

    units = []
    for u in result["units"]:
        bbox = u.get("bbox", {})
        bx = bbox.get("x", 0)
        by = bbox.get("y", 0)
        bw = bbox.get("width", 0)
        bh = bbox.get("height", 0)

        # Convert normalized to pixel if values are <= 1.0
        if bx <= 1.0 and by <= 1.0 and bw <= 1.0 and bh <= 1.0:
            bx, by, bw, bh = bx * img_w, by * img_h, bw * img_w, bh * img_h

        unit_type = u.get("type", "residential").lower()
        is_public = unit_type in PUBLIC_TYPES

        units.append({
            "name": u.get("name", "Unknown"),
            "type": unit_type,
            "is_public": is_public,
            "bbox_px": (int(bx), int(by), int(bw), int(bh)),
            "printed_area_sqm": u.get("printed_area_sqm"),
        })

    return units


def analyze_units(image: np.ndarray, model: str | None = None) -> list[dict]:
    """Pass 1: Identify apartment units and public spaces."""
    try:
        prompt = _build_pass1_prompt()
        response = _call_gemini_with_model(image, prompt, model=model)
        h, w = image.shape[:2]
        return _parse_pass1_response(response, img_w=w, img_h=h)
    except Exception as e:
        _logger.error(f"analyze_units failed: {e}")
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/furnished_analyzer.py backend/tests/pipeline/test_furnished_analyzer.py
git commit -m "feat: add furnished analyzer Pass 1 — unit detection with Gemini"
```

---

### Task 5: Build furnished analyzer — Pass 2 (room polygon extraction)

**Files:**
- Modify: `backend/pipeline/furnished_analyzer.py`
- Modify: `backend/tests/pipeline/test_furnished_analyzer.py`

- [ ] **Step 1: Write failing tests for Pass 2**

Add to `backend/tests/pipeline/test_furnished_analyzer.py`:

```python
from backend.pipeline.furnished_analyzer import (
    _build_pass2_prompt,
    _parse_pass2_response,
    analyze_rooms_in_unit,
)


class TestPass2Prompt:
    def test_prompt_includes_unit_name(self):
        prompt = _build_pass2_prompt("UNIT 20.01")
        assert "UNIT 20.01" in prompt
        assert "polygon" in prompt.lower()


class TestPass2Parsing:
    def test_parse_valid_room_polygons(self):
        response = json.dumps({
            "rooms": [
                {
                    "name": "Master Bedroom",
                    "type": "bedroom",
                    "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]],
                    "printed_area_sqm": 15.0,
                    "confidence": 0.9,
                },
                {
                    "name": "Kitchen",
                    "type": "kitchen",
                    "polygon": [[0.5, 0.1], [0.9, 0.1], [0.9, 0.4], [0.5, 0.4]],
                    "printed_area_sqm": None,
                    "confidence": 0.8,
                },
            ]
        })
        result = _parse_pass2_response(response, crop_w=500, crop_h=400)
        assert len(result) == 2
        assert result[0]["name"] == "Master Bedroom"
        assert result[0]["type"] == "bedroom"
        # Polygon should be in pixel coords of the crop
        assert result[0]["polygon_px"][0] == (int(0.1 * 500), int(0.1 * 400))
        assert result[0]["printed_area_sqm"] == 15.0

    def test_parse_empty_rooms(self):
        response = json.dumps({"rooms": []})
        result = _parse_pass2_response(response, crop_w=500, crop_h=400)
        assert result == []

    def test_parse_invalid_polygon_skipped(self):
        response = json.dumps({
            "rooms": [
                {"name": "Bad", "type": "other", "polygon": [[0.1, 0.1]], "confidence": 0.5},
                {"name": "Good", "type": "bedroom", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], "confidence": 0.8},
            ]
        })
        result = _parse_pass2_response(response, crop_w=1000, crop_h=1000)
        assert len(result) == 1
        assert result[0]["name"] == "Good"


class TestAnalyzeRoomsInUnit:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_returns_rooms_with_offset(self, mock_call):
        mock_call.return_value = json.dumps({
            "rooms": [
                {
                    "name": "Bedroom",
                    "type": "bedroom",
                    "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    "confidence": 0.9,
                }
            ]
        })
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        unit = {
            "name": "UNIT 20.01",
            "bbox_px": (100, 200, 500, 400),
            "is_public": False,
        }
        result = analyze_rooms_in_unit(img, unit)
        assert len(result) == 1
        assert result[0]["name"] == "Bedroom"
        assert result[0]["unit_name"] == "UNIT 20.01"
        # Polygon should be offset to full-image coordinates
        # crop is img[200:600, 100:600] => 500x400 crop
        # polygon point (0.1*500, 0.1*400) = (50, 40) in crop
        # offset: (50+100, 40+200) = (150, 240) in full image
        assert result[0]["polygon_px"][0] == (150, 240)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py::TestPass2Prompt -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement Pass 2**

Add to `backend/pipeline/furnished_analyzer.py`:

```python
# ---------------------------------------------------------------------------
# Pass 2: Room polygon extraction per unit
# ---------------------------------------------------------------------------

def _build_pass2_prompt(unit_name: str) -> str:
    return f"""You are analyzing a CROPPED section of an architectural floor plan showing the interior of "{unit_name}".

Identify EVERY distinct room and enclosed space within this area. This includes:
- Bedrooms, bathrooms, en-suites, walk-in closets
- Kitchen, living room, dining room (even if open-concept with no separating wall)
- Balconies, terraces, service yards
- Corridors, hallways, foyers within the unit
- Store rooms, utility spaces

For open-concept areas (e.g., living/dining/kitchen with no walls between them):
- If they are clearly distinct functional zones, return separate polygons for each
- If they are one continuous undivided space, return a single polygon and name it descriptively (e.g., "Living/Dining/Kitchen")

Use door swing arcs as boundary indicators between rooms — where you see a door, that marks where one room ends and another begins.

For each room, return:
- **name**: Room label as written on the plan, or a descriptive name if unlabeled
- **type**: One of: bedroom, bathroom, kitchen, living_room, dining_room, balcony, corridor, storage, utility, study, laundry, foyer, walk_in_closet, terrace, other
- **polygon**: Array of [x, y] vertices as NORMALIZED coordinates (0.0 to 1.0 within this cropped image). Trace the room boundary following walls. Use at least 4 vertices. Go clockwise.
- **printed_area_sqm**: Area in m² printed on the plan for this specific room (null if not visible)
- **confidence**: 0.0 to 1.0

Return ONLY valid JSON:
```json
{{
  "rooms": [
    {{
      "name": "Master Bedroom",
      "type": "bedroom",
      "polygon": [[0.05, 0.1], [0.45, 0.1], [0.45, 0.55], [0.05, 0.55]],
      "printed_area_sqm": 15,
      "confidence": 0.9
    }}
  ]
}}
```"""


def _parse_pass2_response(response_text: str, crop_w: int, crop_h: int) -> list[dict]:
    """Parse Pass 2 response into room dicts with pixel-coordinate polygons."""
    result = _parse_json(response_text)
    if result is None:
        result = _repair_truncated_json(response_text, "rooms")
    if not result or "rooms" not in result:
        return []

    rooms = []
    for r in result["rooms"]:
        raw_polygon = r.get("polygon", [])
        if len(raw_polygon) < 3:
            continue

        # Convert normalized coords to pixel coords within the crop
        polygon_px = []
        for pt in raw_polygon:
            if len(pt) >= 2:
                px = pt[0] * crop_w if pt[0] <= 1.0 else pt[0]
                py = pt[1] * crop_h if pt[1] <= 1.0 else pt[1]
                polygon_px.append((int(px), int(py)))

        if len(polygon_px) < 3:
            continue

        rooms.append({
            "name": r.get("name", "Unnamed"),
            "type": r.get("type", "other"),
            "polygon_px": polygon_px,
            "printed_area_sqm": r.get("printed_area_sqm"),
            "confidence": float(r.get("confidence", 0.5)),
        })

    return rooms


def analyze_rooms_in_unit(
    image: np.ndarray,
    unit: dict,
    model: str | None = None,
) -> list[dict]:
    """Pass 2: Extract room polygons for a single unit by cropping and querying Gemini."""
    bx, by, bw, bh = unit["bbox_px"]
    h, w = image.shape[:2]

    # Clamp bbox to image bounds
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(w, bx + bw)
    y2 = min(h, by + bh)

    if x2 <= x1 or y2 <= y1:
        _logger.warning(f"Invalid bbox for unit {unit['name']}: {unit['bbox_px']}")
        return []

    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    try:
        prompt = _build_pass2_prompt(unit["name"])
        response = _call_gemini_with_model(crop, prompt, model=model)
        rooms = _parse_pass2_response(response, crop_w=crop_w, crop_h=crop_h)
    except Exception as e:
        _logger.error(f"analyze_rooms_in_unit failed for {unit['name']}: {e}")
        return []

    # Transform polygon coordinates from crop-space to full-image-space
    for room in rooms:
        room["polygon_px"] = [(px + x1, py + y1) for px, py in room["polygon_px"]]
        room["unit_name"] = unit["name"]

    return rooms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/furnished_analyzer.py backend/tests/pipeline/test_furnished_analyzer.py
git commit -m "feat: add furnished analyzer Pass 2 — room polygon extraction per unit"
```

---

### Task 6: Build furnished analyzer — orchestrator and debug images

**Files:**
- Modify: `backend/pipeline/furnished_analyzer.py`
- Modify: `backend/tests/pipeline/test_furnished_analyzer.py`

- [ ] **Step 1: Write failing test for orchestrator**

Add to `backend/tests/pipeline/test_furnished_analyzer.py`:

```python
from backend.pipeline.furnished_analyzer import run_furnished_pipeline, _check_area_divergence


class TestAreaDivergence:
    def test_no_divergence_when_close(self):
        # 10000 px area at 100 px/m => 1.0 sqm, printed is 1.0 => no divergence
        assert _check_area_divergence(10000.0, 1.0, 100.0) is False

    def test_divergence_when_far(self):
        # 10000 px area at 100 px/m => 1.0 sqm, printed is 2.0 => 50% off
        assert _check_area_divergence(10000.0, 2.0, 100.0) is True

    def test_no_divergence_when_no_printed_area(self):
        assert _check_area_divergence(10000.0, None, 100.0) is False

    def test_no_divergence_when_no_scale(self):
        assert _check_area_divergence(10000.0, 1.0, None) is False


class TestRunFurnishedPipeline:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_full_pipeline_combines_passes(self, mock_call):
        # First call = Pass 1 (unit listing)
        # Second call = Pass 2 (room detail for the one residential unit)
        # Public space should NOT trigger a Pass 2 call
        mock_call.side_effect = [
            # Pass 1 response
            json.dumps({
                "units": [
                    {"name": "UNIT 1", "type": "residential", "bbox": {"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.4}, "printed_area_sqm": 86},
                    {"name": "LOBBY", "type": "public", "bbox": {"x": 0.0, "y": 0.5, "width": 0.1, "height": 0.1}, "printed_area_sqm": None},
                ]
            }),
            # Pass 2 response for UNIT 1
            json.dumps({
                "rooms": [
                    {"name": "Bedroom", "type": "bedroom", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], "confidence": 0.9},
                ]
            }),
        ]
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        rooms = run_furnished_pipeline(img)

        # Should have 2 rooms: 1 from Pass 2 (bedroom in UNIT 1) + 1 public space (LOBBY)
        assert len(rooms) == 2
        bedroom = next(r for r in rooms if r["name"] == "Bedroom")
        assert bedroom["unit_name"] == "UNIT 1"
        lobby = next(r for r in rooms if r["name"] == "LOBBY")
        assert lobby["unit_name"] == "Public Space"

        # Pass 2 should only be called once (for UNIT 1, not LOBBY)
        assert mock_call.call_count == 2

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_pipeline_handles_pass1_failure(self, mock_call):
        mock_call.side_effect = Exception("API down")
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        rooms = run_furnished_pipeline(img)
        assert rooms == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py::TestRunFurnishedPipeline -v`
Expected: FAIL — `run_furnished_pipeline` not defined

- [ ] **Step 3: Implement orchestrator and debug image generation**

Add to `backend/pipeline/furnished_analyzer.py`:

```python
import cv2
import os
from shapely.geometry import Polygon
from shapely.validation import make_valid


def _unit_bbox_to_polygon(unit: dict, img_w: int, img_h: int) -> list[tuple[int, int]]:
    """Convert a unit's bbox into a 4-point polygon for public spaces."""
    bx, by, bw, bh = unit["bbox_px"]
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(img_w, bx + bw)
    y2 = min(img_h, by + bh)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _compute_room_geometry(polygon_px: list[tuple[int, int]]) -> dict:
    """Compute area, perimeter, centroid, boundary lengths from pixel polygon."""
    try:
        poly = Polygon(polygon_px)
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type != "Polygon":
                return {}
    except Exception:
        return {}

    centroid = poly.centroid
    coords = list(poly.exterior.coords)
    boundary_lengths = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        boundary_lengths.append(float((dx**2 + dy**2) ** 0.5))

    return {
        "polygon": poly,
        "area_px": float(poly.area),
        "perimeter_px": float(poly.length),
        "centroid": (float(centroid.x), float(centroid.y)),
        "boundary_lengths_px": boundary_lengths,
        "boundary_polygon": [[float(x), float(y)] for x, y in coords],
    }


def _check_area_divergence(
    area_px: float,
    printed_area_sqm: float | None,
    px_per_meter: float | None,
    threshold: float = 0.3,
) -> bool:
    """Return True if polygon area diverges >threshold from printed area."""
    if printed_area_sqm is None or px_per_meter is None or px_per_meter <= 0:
        return False
    computed_sqm = area_px / (px_per_meter ** 2)
    if printed_area_sqm <= 0:
        return False
    ratio = abs(computed_sqm - printed_area_sqm) / printed_area_sqm
    return ratio > threshold


def run_furnished_pipeline(
    image: np.ndarray,
    flash_model: str = "gemini-2.5-flash",
    detail_model: str | None = None,
    px_per_meter: float | None = None,
    debug_dir: str | None = None,
    progress_cb: callable | None = None,
) -> list[dict]:
    """Run the full two-pass furnished pipeline.

    Args:
        image: RGB image array.
        flash_model: Model for Pass 1 (always Flash).
        detail_model: Model for Pass 2 (user-selected, defaults to flash_model).
        px_per_meter: Scale factor for area divergence checking.
        debug_dir: If set, save annotated debug images to this directory.
        progress_cb: Optional callback(percent, message) for progress updates.

    Returns:
        List of room dicts ready for RoomData construction.
    """
    def _progress(pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    if detail_model is None:
        detail_model = flash_model

    h, w = image.shape[:2]

    # --- Pass 1: Identify units ---
    _progress(5, "Pass 1: identifying units...")
    _logger.info("Furnished pipeline Pass 1: identifying units...")
    units = analyze_units(image, model=flash_model)
    if not units:
        _logger.warning("Pass 1 found no units")
        return []

    _logger.info(f"Pass 1 found {len(units)} units/spaces")

    all_rooms: list[dict] = []

    _progress(20, f"Pass 1 found {len(units)} units. Starting Pass 2...")

    # --- Pass 2: Extract rooms per unit ---
    residential_units = [u for u in units if not u["is_public"]]
    for idx, unit in enumerate(units):
        if not unit["is_public"]:
            res_idx = residential_units.index(unit)
            pct = 20 + int((res_idx + 1) / max(len(residential_units), 1) * 60)
            _progress(pct, f"Pass 2: analyzing {unit['name']} ({res_idx+1}/{len(residential_units)})...")
        if unit["is_public"]:
            # Public spaces: use bbox as polygon, skip Pass 2
            polygon_px = _unit_bbox_to_polygon(unit, w, h)
            geom = _compute_room_geometry(polygon_px)
            if not geom:
                continue
            room = {
                "name": unit["name"],
                "type": unit["type"],
                "unit_name": "Public Space",
                "printed_area_sqm": unit.get("printed_area_sqm"),
                "confidence": 0.7,
                **geom,
                "area_divergence_flag": _check_area_divergence(
                    geom["area_px"], unit.get("printed_area_sqm"), px_per_meter
                ),
            }
            all_rooms.append(room)
        else:
            # Residential units: run Pass 2
            _logger.info(f"Pass 2: analyzing rooms in {unit['name']}...")
            unit_rooms = analyze_rooms_in_unit(image, unit, model=detail_model)
            for ur in unit_rooms:
                geom = _compute_room_geometry(ur["polygon_px"])
                if not geom:
                    continue
                room = {
                    "name": ur["name"],
                    "type": ur["type"],
                    "unit_name": ur["unit_name"],
                    "printed_area_sqm": ur.get("printed_area_sqm"),
                    "confidence": ur.get("confidence", 0.5),
                    **geom,
                    "area_divergence_flag": _check_area_divergence(
                        geom["area_px"], ur.get("printed_area_sqm"), px_per_meter
                    ),
                }
                all_rooms.append(room)

    # --- Debug images ---
    if debug_dir:
        _save_debug_images(image, units, all_rooms, debug_dir)

    _logger.info(f"Furnished pipeline complete: {len(all_rooms)} rooms found")
    return all_rooms


def _save_debug_images(
    image: np.ndarray,
    units: list[dict],
    rooms: list[dict],
    debug_dir: str,
) -> None:
    """Save annotated debug images showing detected units and rooms."""
    os.makedirs(debug_dir, exist_ok=True)

    # Debug image 1: Unit bounding boxes
    unit_img = image.copy()
    for i, unit in enumerate(units):
        bx, by, bw, bh = unit["bbox_px"]
        color = (0, 255, 0) if not unit["is_public"] else (255, 165, 0)
        cv2.rectangle(unit_img, (bx, by), (bx + bw, by + bh), color, 3)
        cv2.putText(unit_img, unit["name"], (bx + 5, by + 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    bgr = cv2.cvtColor(unit_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir, "debug_units.jpg"), bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Debug image 2: Room polygons
    room_img = image.copy()
    colors = [
        (76, 175, 80), (33, 150, 243), (255, 152, 0), (156, 39, 176),
        (244, 67, 54), (0, 188, 212), (255, 235, 59), (121, 85, 72),
    ]
    for i, room in enumerate(rooms):
        if "boundary_polygon" not in room:
            continue
        pts = np.array([[int(x), int(y)] for x, y in room["boundary_polygon"][:-1]], dtype=np.int32)
        color = colors[i % len(colors)]
        cv2.polylines(room_img, [pts], True, color, 2)
        cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
        label = f"{room['name']} ({room.get('unit_name', '?')})"
        cv2.putText(room_img, label, (cx - 50, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    bgr = cv2.cvtColor(room_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir, "debug_rooms.jpg"), bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    _logger.info(f"Debug images saved to {debug_dir}")
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/pipeline/test_furnished_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/furnished_analyzer.py backend/tests/pipeline/test_furnished_analyzer.py
git commit -m "feat: add furnished pipeline orchestrator with debug image output"
```

---

### Task 7: Wire up furnished mode in the backend API

**Files:**
- Modify: `backend/main.py:127-135` (process endpoint params), `backend/main.py:199-205` (mode routing)

- [ ] **Step 1: Add `gemini_model` parameter to process endpoint**

In `backend/main.py`, add a new Form parameter to `process_pdf()` (after line 134):

```python
    gemini_model: str = Form("flash"),
```

- [ ] **Step 2: Add furnished mode routing**

In `backend/main.py`, add the new mode branch (around line 199, in the mode routing block):

```python
        elif mode == "furnished":
            rooms_out = _process_furnished_mode(image, project, px_per_meter_val, db, _update, gemini_model)
```

- [ ] **Step 3: Implement `_process_furnished_mode()`**

Add this function to `backend/main.py` (after `_process_linedraw_mode`):

```python
def _process_furnished_mode(
    image: np.ndarray,
    project: ProjectData,
    px_per_meter: Optional[float],
    db: Database,
    _update=lambda p, m: None,
    gemini_model: str = "flash",
) -> list[dict]:
    """Furnished residential pipeline — Gemini two-pass room detection."""
    from backend.pipeline.furnished_analyzer import run_furnished_pipeline

    model_map = {
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }
    detail_model = model_map.get(gemini_model, "gemini-2.5-flash")
    flash_model = "gemini-2.5-flash"

    _update(10, "Analyzing floor plan (Pass 1 — identifying units)...")

    debug_dir = os.path.join(os.path.dirname(project.pdf_path or "."), "debug")

    def _pipeline_progress(percent: int, message: str):
        # Map pipeline 0-100 to overall 10-80 range
        mapped = 10 + int(percent * 0.7)
        _update(mapped, message)

    raw_rooms = run_furnished_pipeline(
        image,
        flash_model=flash_model,
        detail_model=detail_model,
        px_per_meter=px_per_meter,
        debug_dir=debug_dir,
        progress_cb=_pipeline_progress,
    )

    _update(80, f"Saving {len(raw_rooms)} rooms...")

    rooms_out = []
    for raw in raw_rooms:
        polygon_coords = raw.get("boundary_polygon", [])
        if not polygon_coords:
            continue

        real = {}
        if px_per_meter:
            real = to_real_measurements(
                raw["area_px"],
                raw["perimeter_px"],
                raw["boundary_lengths_px"],
                px_per_meter,
            )

        fill_color = _sample_fill_color(image, raw["centroid"], raw.get("polygon"))

        room = RoomData(
            project_id=project.id,
            name=raw["name"],
            room_type=raw.get("type", "unknown"),
            boundary_polygon=polygon_coords,
            area_px=raw["area_px"],
            perimeter_px=raw["perimeter_px"],
            centroid=raw["centroid"],
            boundary_lengths_px=raw["boundary_lengths_px"],
            area_sqm=real.get("area_sqm"),
            perimeter_m=real.get("perimeter_m"),
            boundary_lengths_m=real.get("boundary_lengths_m"),
            fill_color_rgb=fill_color,
            source="furnished",
            confidence=float(raw.get("confidence", 0.5)),
            unit_name=raw.get("unit_name"),
            printed_area_sqm=raw.get("printed_area_sqm"),
            area_divergence_flag=raw.get("area_divergence_flag", False),
        )
        db.save_room(room)
        rooms_out.append(room.model_dump())

    _update(100, "Done!")
    return rooms_out
```

- [ ] **Step 4: Run existing API tests to verify no regression**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_api.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "feat: wire up furnished processing mode in backend API"
```

---

### Task 8: Update frontend — API layer and types

**Files:**
- Modify: `frontend/src/api.ts:5-19` (Room interface), `frontend/src/api.ts:44` (ProcessMode), `frontend/src/api.ts:46-57` (processFloorplan)

- [ ] **Step 1: Add new fields to Room interface**

In `frontend/src/api.ts`, add after line 18 (after `confidence`):

```typescript
  unit_name: string | null;
  printed_area_sqm: number | null;
  area_divergence_flag: boolean;
```

- [ ] **Step 2: Add 'furnished' to ProcessMode**

Change line 44:

```typescript
export type ProcessMode = 'hybrid' | 'gemini' | 'linedraw' | 'furnished';
```

- [ ] **Step 3: Add geminiModel parameter to processFloorplan**

Update the function signature and body (lines 46-57):

```typescript
export async function processFloorplan(
  file: File, pageNum = 0, mode: ProcessMode = 'hybrid',
  jobId?: string, filterColors = true, geminiModel = 'flash',
): Promise<ProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', mode);
  formData.append('gemini_model', geminiModel);
  if (jobId) formData.append('job_id', jobId);
  if (mode === 'linedraw') formData.append('filter_colors', filterColors ? 'true' : 'false');
  const { data } = await api.post(`/process?page_num=${pageNum}`, formData);
  return data;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat: add furnished mode and gemini model selection to frontend API"
```

---

### Task 9: Update frontend — App.tsx mode selector and model picker

**Files:**
- Modify: `frontend/src/App.tsx:14-18` (MODE_LABELS), `frontend/src/App.tsx:45` (state), `frontend/src/App.tsx:92` (doUpload call), `frontend/src/App.tsx:267-300` (mode selector UI)

- [ ] **Step 1: Add state and mode label for furnished**

In `frontend/src/App.tsx`:

Add to `MODE_LABELS` (line 14-18):

```typescript
  furnished: { label: 'Furnished', color: 'bg-emerald-600/30 text-emerald-400' },
```

Add state for model selection (after line 58, after `filterColors`):

```typescript
  const [geminiModel, setGeminiModel] = useState<'flash' | 'pro'>('flash');
```

- [ ] **Step 2: Pass geminiModel to processFloorplan call**

Update the `doUpload` call (line 92):

```typescript
      const result = await processFloorplan(file, 0, mode, jobId, filterColors, geminiModel);
```

Update the `useCallback` dependency array for `doUpload` to include `geminiModel`:

```typescript
  }, [mode, geminiModel]);
```

- [ ] **Step 3: Add Furnished button and model selector to the mode picker UI**

In the mode selector array (lines 268-272), add the furnished option:

```typescript
              { value: 'furnished' as ProcessMode, label: 'Furnished', color: 'bg-emerald-600' },
```

Add a mode description for furnished (around line 289):

```typescript
            {mode === 'furnished' && 'Two-pass Gemini AI for furnished residential floorplans'}
```

Add model selector that shows for gemini-dependent modes (after the mode description, around line 290):

```tsx
          {(mode === 'gemini' || mode === 'furnished') && (
            <div className="flex items-center justify-center gap-2">
              <span className="text-neutral-500 text-xs">Model:</span>
              {(['flash', 'pro'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setGeminiModel(m)}
                  className={`px-2.5 py-1 text-xs rounded-full transition-colors ${
                    geminiModel === m
                      ? 'bg-neutral-600 text-white'
                      : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                  }`}
                >
                  {m === 'flash' ? 'Flash (free)' : 'Pro (paid)'}
                </button>
              ))}
            </div>
          )}
```

- [ ] **Step 4: Verify frontend builds**

Run: `source ~/.nvm/nvm.sh && nvm use 22 && cd frontend && npx vite build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add Furnished mode button and Flash/Pro model selector to UI"
```

---

### Task 10: Update frontend — RoomSidebar and RoomDetail for unit display

**Files:**
- Modify: `frontend/src/components/RoomSidebar.tsx:9-13` (MODE_LABELS), `frontend/src/components/RoomSidebar.tsx:79-83` (room item)
- Modify: `frontend/src/components/RoomDetail.tsx:95-104` (area section)

- [ ] **Step 1: Add furnished to RoomSidebar MODE_LABELS and show unit_name**

In `frontend/src/components/RoomSidebar.tsx`:

Add to `MODE_LABELS` (line 9-13):

```typescript
  furnished: { label: 'Furnished', color: 'bg-emerald-600/30 text-emerald-400' },
```

In the room list item (after line 79, after the room name div):

```tsx
            {room.unit_name && (
              <div className="text-xs text-neutral-600">{room.unit_name}</div>
            )}
```

- [ ] **Step 2: Add unit name, printed area, and divergence flag to RoomDetail**

In `frontend/src/components/RoomDetail.tsx`:

After the type selector section (around line 93), add unit name display:

```tsx
        {room.unit_name && (
          <div>
            <label className="block text-neutral-500 text-xs mb-1">Unit</label>
            <div className="text-neutral-300 text-sm">{room.unit_name}</div>
          </div>
        )}
```

In the area section (around line 99), after the area_sqm display, add printed area and divergence:

```tsx
          {room.printed_area_sqm != null && (
            <div className="text-neutral-500 text-xs mt-0.5">
              Printed: {room.printed_area_sqm} m²
              {room.area_divergence_flag && (
                <span className="text-amber-400 ml-1" title="Polygon area differs significantly from printed area">
                  ⚠ divergence
                </span>
              )}
            </div>
          )}
```

Add `"furnished"` source display in the source section (around line 168):

```typescript
             room.source === 'furnished' ? 'Furnished (Gemini)' :
```

And the corresponding color:

```typescript
            room.source === 'furnished' ? 'text-emerald-400' :
```

- [ ] **Step 3: Verify frontend builds**

Run: `source ~/.nvm/nvm.sh && nvm use 22 && cd frontend && npx vite build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/RoomSidebar.tsx frontend/src/components/RoomDetail.tsx
git commit -m "feat: display unit name, printed area, and divergence flag in sidebar and detail"
```

---

### Task 11: Update export endpoint to include new fields

**Files:**
- Modify: `backend/main.py:754-821` (export endpoint)

- [ ] **Step 1: Add new fields to CSV export**

In `backend/main.py`, in the `export_project` function, update the CSV `fieldnames` list to include the new columns:

```python
fieldnames=[
    "id", "name", "room_type",
    "area_px", "perimeter_px", "area_sqm", "perimeter_m",
    "centroid_x", "centroid_y", "source", "confidence",
    "unit_name", "printed_area_sqm", "area_divergence_flag",
],
```

And update the `writer.writerow()` dict to include them:

```python
"unit_name": room.unit_name,
"printed_area_sqm": room.printed_area_sqm,
"area_divergence_flag": room.area_divergence_flag,
```

The JSON export already uses `r.model_dump()` which will automatically include the new fields — no change needed there.

- [ ] **Step 2: Run existing tests**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/test_api.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: include unit_name, printed_area, divergence_flag in CSV export"
```

---

### Task 12: Run all tests and verify end-to-end

**Files:** None (verification only)

- [ ] **Step 1: Run full backend test suite**

Run: `/opt/anaconda3/bin/python3 -m pytest backend/tests/ -v`
Expected: ALL PASS (55 existing + ~12 new = ~67 tests)

- [ ] **Step 2: Verify frontend builds cleanly**

Run: `source ~/.nvm/nvm.sh && nvm use 22 && cd frontend && npx vite build 2>&1 | tail -10`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 3: Commit any final fixes if needed**

---

### Task 13: Manual integration test with input_sample_3.pdf

**Files:** None (manual verification)

- [ ] **Step 1: Start backend**

Run: `GOOGLE_API_KEY=<key> /opt/anaconda3/bin/python3 -m uvicorn backend.main:app --reload --port 8000`

- [ ] **Step 2: Start frontend**

Run: `source ~/.nvm/nvm.sh && nvm use 22 && cd frontend && npm run dev`

- [ ] **Step 3: Test in browser**

1. Open http://localhost:5173
2. Select "Furnished" mode
3. Select "Flash (free)" model
4. Upload `input_sample_3.pdf`
5. Verify: rooms appear in sidebar with unit tags
6. Verify: clicking rooms selects polygons on canvas
7. Verify: detail panel shows unit name, printed area, divergence flag
8. Check `debug/debug_units.jpg` and `debug/debug_rooms.jpg` for visual verification

- [ ] **Step 4: Commit any tuning changes**

```bash
git add -A
git commit -m "feat: furnished mode v1 — Gemini two-pass room detection for residential floorplans"
```
