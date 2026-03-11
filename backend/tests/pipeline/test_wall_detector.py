import numpy as np
import pytest
from backend.pipeline.wall_detector import detect_walls

class TestDetectWalls:
    def _make_wall_image(self, w=400, h=400):
        img = np.zeros((h, w), dtype=np.uint8)
        img[100:105, 50:350] = 255
        img[300:305, 50:350] = 255
        img[100:305, 50:55] = 255
        img[100:305, 345:350] = 255
        return img

    def test_returns_wall_segments(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        assert "segments" in result
        assert len(result["segments"]) > 0

    def test_segments_have_endpoints(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        for seg in result["segments"]:
            assert "x1" in seg and "y1" in seg
            assert "x2" in seg and "y2" in seg

    def test_returns_wall_mask(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        assert "wall_mask" in result
        assert result["wall_mask"].shape == img.shape

    def test_detects_horizontal_walls(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        horizontal = [s for s in result["segments"] if s["orientation"] == "horizontal"]
        assert len(horizontal) >= 2

    def test_detects_vertical_walls(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        vertical = [s for s in result["segments"] if s["orientation"] == "vertical"]
        assert len(vertical) >= 2

    def test_empty_image_returns_no_walls(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        result = detect_walls(img)
        assert len(result["segments"]) == 0
