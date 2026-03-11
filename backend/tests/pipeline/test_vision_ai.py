import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from backend.pipeline.vision_ai import (
    classify_regions, label_rooms,
    _build_classification_prompt, _build_labeling_prompt, _parse_json_response,
)

class TestBuildPrompts:
    def test_classification_prompt_is_string(self):
        prompt = _build_classification_prompt()
        assert isinstance(prompt, str)
        assert "floor" in prompt.lower()
        assert "table" in prompt.lower()

    def test_labeling_prompt_includes_room_count(self):
        prompt = _build_labeling_prompt(room_count=5)
        assert "5" in prompt

class TestParseJsonResponse:
    def test_parses_valid_json(self):
        text = '```json\n{"regions": [{"type": "floorplan"}]}\n```'
        result = _parse_json_response(text)
        assert result is not None
        assert "regions" in result

    def test_parses_raw_json(self):
        text = '{"regions": []}'
        result = _parse_json_response(text)
        assert result is not None

    def test_invalid_json_returns_none(self):
        result = _parse_json_response("not json at all")
        assert result is None

class TestClassifyRegions:
    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_returns_regions(self, mock_call):
        mock_call.return_value = json.dumps({
            "floorplan_regions": [{"x": 0, "y": 0, "width": 100, "height": 100}],
            "excluded_regions": [{"x": 200, "y": 0, "width": 50, "height": 50, "type": "table"}],
        })
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = classify_regions(img)
        assert "floorplan_regions" in result
        assert "excluded_regions" in result

    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_api_failure_returns_fallback(self, mock_call):
        mock_call.side_effect = Exception("API error")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = classify_regions(img)
        assert "floorplan_regions" in result
        assert len(result["floorplan_regions"]) == 1

class TestLabelRooms:
    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_returns_labels(self, mock_call):
        mock_call.return_value = json.dumps({
            "rooms": [{"room_id": 0, "name": "Office 201", "type": "office", "confidence": 0.9}]
        })
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        rooms = [{"centroid": (50, 50), "polygon": None}]
        result = label_rooms(img, rooms)
        assert len(result) == 1
        assert result[0]["name"] == "Office 201"

    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_api_failure_returns_unnamed(self, mock_call):
        mock_call.side_effect = Exception("API error")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        rooms = [{"centroid": (50, 50), "polygon": None}]
        result = label_rooms(img, rooms)
        assert result[0]["name"] == "Unnamed"
