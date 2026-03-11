import pytest
from backend.pipeline.scale_detector import detect_scale, parse_scale_text

class TestParseScaleText:
    def test_parses_px_to_meter(self):
        result = parse_scale_text("Scale: 1px = 0.0169m")
        assert result is not None
        assert abs(result["px_per_meter"] - (1 / 0.0169)) < 0.1

    def test_parses_ratio_scale(self):
        result = parse_scale_text("1:100")
        assert result is not None
        assert result["scale_ratio"] == 100

    def test_parses_ratio_with_prefix(self):
        result = parse_scale_text("Scale 1:200")
        assert result is not None
        assert result["scale_ratio"] == 200

    def test_no_scale_returns_none(self):
        result = parse_scale_text("No scale here")
        assert result is None

    def test_empty_text_returns_none(self):
        result = parse_scale_text("")
        assert result is None

class TestDetectScale:
    def test_detects_from_text(self):
        result = detect_scale(text="GPU-Accelerated Analysis | Scale: 1px = 0.0169m | Rooms: 359")
        assert result is not None
        assert "px_per_meter" in result
        assert result["source"] == "auto"

    def test_manual_override(self):
        result = detect_scale(text="no scale", manual_px_per_meter=59.17)
        assert result is not None
        assert abs(result["px_per_meter"] - 59.17) < 0.01
        assert result["source"] == "manual"

    def test_no_scale_no_manual_returns_none(self):
        result = detect_scale(text="no scale text")
        assert result is None
