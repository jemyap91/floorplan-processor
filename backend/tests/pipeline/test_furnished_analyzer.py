"""Tests for furnished_analyzer module — two-pass Gemini pipeline."""
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Pass 1 — unit detection
# ---------------------------------------------------------------------------

class TestPass1Prompt:
    def test_prompt_is_string(self):
        from backend.pipeline.furnished_analyzer import _build_pass1_prompt
        prompt = _build_pass1_prompt()
        assert isinstance(prompt, str)

    def test_prompt_contains_key_terms(self):
        from backend.pipeline.furnished_analyzer import _build_pass1_prompt
        prompt = _build_pass1_prompt().lower()
        assert "unit" in prompt
        assert "box_2d" in prompt or "bounding box" in prompt
        assert "structural" in prompt


class TestPass1Parsing:
    def test_valid_response(self):
        from backend.pipeline.furnished_analyzer import _parse_pass1_response
        # box_2d: [ymin, xmin, ymax, xmax] in 0-1000
        response = json.dumps({
            "units": [
                {"name": "Unit A", "type": "residential",
                 "box_2d": [200, 100, 600, 600]},
                {"name": "Lobby", "type": "public",
                 "box_2d": [0, 600, 300, 1000]},
            ]
        })
        units = _parse_pass1_response(response, img_w=1000, img_h=800)
        assert len(units) == 2
        u0 = units[0]
        assert u0["name"] == "Unit A"
        assert u0["type"] == "residential"
        # bbox_px = (x=100/1000*1000, y=200/1000*800, w=(600-100)/1000*1000, h=(600-200)/1000*800)
        assert u0["bbox_px"] == (100, 160, 500, 320)
        assert u0["is_public"] is False
        assert units[1]["is_public"] is True

    def test_truncated_response(self):
        from backend.pipeline.furnished_analyzer import _parse_pass1_response
        response = '```json\n{"units": [{"name": "Unit A", "type": "residential", "box_2d": [200, 100, 600, 600]},'
        units = _parse_pass1_response(response, img_w=1000, img_h=800)
        assert len(units) == 1
        assert units[0]["name"] == "Unit A"

    def test_invalid_response_returns_empty(self):
        from backend.pipeline.furnished_analyzer import _parse_pass1_response
        units = _parse_pass1_response("not json at all", img_w=1000, img_h=800)
        assert units == []

    def test_public_types_detection(self):
        from backend.pipeline.furnished_analyzer import _parse_pass1_response
        for ptype in ["lobby", "stairwell", "corridor", "elevator", "utility", "mechanical"]:
            response = json.dumps({
                "units": [{"name": "X", "type": ptype, "box_2d": [0, 0, 1000, 1000]}]
            })
            units = _parse_pass1_response(response, img_w=100, img_h=100)
            assert units[0]["is_public"] is True, f"{ptype} should be public"

    def test_zero_size_box_skipped(self):
        from backend.pipeline.furnished_analyzer import _parse_pass1_response
        response = json.dumps({
            "units": [
                {"name": "Bad", "type": "residential", "box_2d": [500, 500, 500, 500]},  # zero size
                {"name": "Good", "type": "residential", "box_2d": [0, 0, 500, 500]},
            ]
        })
        units = _parse_pass1_response(response, img_w=1000, img_h=1000)
        assert len(units) == 1
        assert units[0]["name"] == "Good"


class TestAnalyzeUnits:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_returns_units(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import analyze_units
        mock_gemini.return_value = json.dumps({
            "units": [
                {"name": "Unit 1", "type": "residential", "box_2d": [0, 0, 500, 500]},
            ]
        })
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        units = analyze_units(img, model="gemini-2.5-flash")
        assert len(units) == 1
        assert units[0]["name"] == "Unit 1"
        mock_gemini.assert_called_once()

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_api_failure_returns_empty(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import analyze_units
        mock_gemini.side_effect = Exception("API error")
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        units = analyze_units(img)
        assert units == []


# ---------------------------------------------------------------------------
# Pass 2 — room bounding boxes per unit
# ---------------------------------------------------------------------------

class TestPass2Prompt:
    def test_prompt_includes_unit_name(self):
        from backend.pipeline.furnished_analyzer import _build_pass2_prompt
        prompt = _build_pass2_prompt("Unit A")
        assert "Unit A" in prompt
        assert "box_2d" in prompt or "bounding box" in prompt.lower()
        assert "structural" in prompt.lower()


class TestPass2Parsing:
    def test_valid_room_boxes(self):
        from backend.pipeline.furnished_analyzer import _parse_pass2_response
        # box_2d: [ymin, xmin, ymax, xmax] in 0-1000
        response = json.dumps({
            "rooms": [
                {"name": "Bedroom", "type": "bedroom",
                 "box_2d": [100, 100, 900, 900],
                 "area_sqm": 12.5},
                {"name": "Kitchen", "type": "kitchen",
                 "box_2d": [0, 0, 500, 500]},
            ]
        })
        rooms = _parse_pass2_response(response, crop_w=500, crop_h=400)
        assert len(rooms) == 2
        r0 = rooms[0]
        assert r0["name"] == "Bedroom"
        # ymin=100, xmin=100 -> x1=100/1000*500=50, y1=100/1000*400=40
        # ymax=900, xmax=900 -> x2=900/1000*500=450, y2=900/1000*400=360
        assert r0["polygon_px"][0] == (50, 40)   # top-left
        assert r0["polygon_px"][1] == (450, 40)   # top-right
        assert r0["polygon_px"][2] == (450, 360)  # bottom-right
        assert r0["polygon_px"][3] == (50, 360)   # bottom-left
        assert r0["printed_area_sqm"] == 12.5

    def test_empty_rooms(self):
        from backend.pipeline.furnished_analyzer import _parse_pass2_response
        response = json.dumps({"rooms": []})
        rooms = _parse_pass2_response(response, crop_w=500, crop_h=400)
        assert rooms == []

    def test_invalid_box_skipped(self):
        from backend.pipeline.furnished_analyzer import _parse_pass2_response
        response = json.dumps({
            "rooms": [
                {"name": "Bad", "type": "other",
                 "box_2d": [500, 500, 500, 500]},  # zero size
                {"name": "Good", "type": "bedroom",
                 "box_2d": [0, 0, 1000, 1000]},
            ]
        })
        rooms = _parse_pass2_response(response, crop_w=100, crop_h=100)
        assert len(rooms) == 1
        assert rooms[0]["name"] == "Good"


class TestAnalyzeRoomsInUnit:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_rooms_offset_correctly(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import analyze_rooms_in_unit
        # box_2d: [ymin, xmin, ymax, xmax] in 0-1000
        mock_gemini.return_value = json.dumps({
            "rooms": [
                {"name": "Bedroom", "type": "bedroom",
                 "box_2d": [100, 100, 900, 900]}
            ]
        })
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        unit = {
            "name": "Unit A",
            "bbox_px": (100, 200, 500, 400),  # x, y, w, h
            "is_public": False,
        }
        rooms = analyze_rooms_in_unit(img, unit, model="gemini-2.5-flash")
        assert len(rooms) == 1
        r = rooms[0]
        # crop is 500x400
        # box_2d [100,100,900,900] -> x1=100/1000*500=50, y1=100/1000*400=40
        #                           -> x2=900/1000*500=450, y2=900/1000*400=360
        # offset by unit bbox (100, 200): (150, 240) and (550, 560)
        assert r["polygon_px"][0] == (150, 240)  # top-left
        assert r["polygon_px"][2] == (550, 560)  # bottom-right
        assert r["unit_name"] == "Unit A"

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_api_failure_returns_empty(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import analyze_rooms_in_unit
        mock_gemini.side_effect = Exception("API error")
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        unit = {"name": "Unit B", "bbox_px": (0, 0, 100, 100), "is_public": False}
        rooms = analyze_rooms_in_unit(img, unit)
        assert rooms == []


# ---------------------------------------------------------------------------
# Area divergence check
# ---------------------------------------------------------------------------

class TestAreaDivergence:
    def test_close_returns_false(self):
        from backend.pipeline.furnished_analyzer import _check_area_divergence
        assert _check_area_divergence(100.0, 1.0, 10.0) is False

    def test_far_returns_true(self):
        from backend.pipeline.furnished_analyzer import _check_area_divergence
        assert _check_area_divergence(200.0, 1.0, 10.0) is True

    def test_no_printed_area_returns_false(self):
        from backend.pipeline.furnished_analyzer import _check_area_divergence
        assert _check_area_divergence(100.0, None, 10.0) is False

    def test_no_scale_returns_false(self):
        from backend.pipeline.furnished_analyzer import _check_area_divergence
        assert _check_area_divergence(100.0, 1.0, None) is False


# ---------------------------------------------------------------------------
# Compute room geometry
# ---------------------------------------------------------------------------

class TestComputeRoomGeometry:
    def test_square_polygon(self):
        from backend.pipeline.furnished_analyzer import _compute_room_geometry
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        geom = _compute_room_geometry(polygon)
        assert abs(geom["area_px"] - 10000.0) < 1.0
        assert abs(geom["perimeter_px"] - 400.0) < 1.0
        assert abs(geom["centroid"][0] - 50.0) < 1.0
        assert abs(geom["centroid"][1] - 50.0) < 1.0


# ---------------------------------------------------------------------------
# Unit bbox to polygon
# ---------------------------------------------------------------------------

class TestUnitBboxToPolygon:
    def test_conversion(self):
        from backend.pipeline.furnished_analyzer import _unit_bbox_to_polygon
        unit = {"bbox_px": (10, 20, 100, 50)}
        poly = _unit_bbox_to_polygon(unit, img_w=500, img_h=400)
        assert poly == [(10, 20), (110, 20), (110, 70), (10, 70)]


# ---------------------------------------------------------------------------
# Orchestrator — run_furnished_pipeline
# ---------------------------------------------------------------------------

class TestRunFurnishedPipeline:
    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_combines_pass1_and_pass2(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        pass1_response = json.dumps({
            "units": [
                {"name": "Unit 1", "type": "residential",
                 "box_2d": [0, 0, 500, 500]},
                {"name": "Lobby", "type": "lobby",
                 "box_2d": [0, 500, 500, 1000]},
            ]
        })
        pass2_response = json.dumps({
            "rooms": [
                {"name": "Bedroom", "type": "bedroom",
                 "box_2d": [0, 0, 1000, 1000]}
            ]
        })
        mock_gemini.side_effect = [pass1_response, pass2_response]

        img = np.zeros((200, 400, 3), dtype=np.uint8)
        rooms = run_furnished_pipeline(img)

        assert mock_gemini.call_count == 2
        residential_rooms = [r for r in rooms if r["unit_name"] == "Unit 1"]
        public_rooms = [r for r in rooms if r["unit_name"] == "Public Space"]
        assert len(residential_rooms) == 1
        assert residential_rooms[0]["name"] == "Bedroom"
        assert len(public_rooms) == 1
        assert public_rooms[0]["name"] == "Lobby"

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_pass1_failure_returns_empty(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline
        mock_gemini.side_effect = Exception("API error")
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        rooms = run_furnished_pipeline(img)
        assert rooms == []

    @patch("backend.pipeline.furnished_analyzer._call_gemini_with_model")
    def test_progress_callback(self, mock_gemini):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        pass1_response = json.dumps({
            "units": [
                {"name": "Unit 1", "type": "residential",
                 "box_2d": [0, 0, 500, 500]},
            ]
        })
        pass2_response = json.dumps({
            "rooms": [
                {"name": "Bedroom", "type": "bedroom",
                 "box_2d": [0, 0, 1000, 1000]}
            ]
        })
        mock_gemini.side_effect = [pass1_response, pass2_response]

        progress_calls = []
        def cb(pct, msg):
            progress_calls.append((pct, msg))

        img = np.zeros((200, 400, 3), dtype=np.uint8)
        run_furnished_pipeline(img, progress_cb=cb)
        assert len(progress_calls) >= 2


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_fenced_json(self):
        from backend.pipeline.furnished_analyzer import _parse_json
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json(text) == {"key": "value"}

    def test_raw_json(self):
        from backend.pipeline.furnished_analyzer import _parse_json
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_invalid_returns_none(self):
        from backend.pipeline.furnished_analyzer import _parse_json
        assert _parse_json("not json") is None


class TestRepairTruncatedJson:
    def test_repair_truncated_units(self):
        from backend.pipeline.furnished_analyzer import _repair_truncated_json
        text = '{"units": [{"name": "A", "type": "residential", "box_2d": [0,0,500,500]},'
        result = _repair_truncated_json(text, "units")
        assert result is not None
        assert len(result["units"]) == 1

    def test_unrepairable_returns_none(self):
        from backend.pipeline.furnished_analyzer import _repair_truncated_json
        assert _repair_truncated_json("garbage", "units") is None
