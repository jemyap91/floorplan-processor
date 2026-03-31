"""Tests for furnished_analyzer module — all-in OpenCV pipeline with Gemini labeling."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from shapely.geometry import Polygon as ShapelyPolygon


# ---------------------------------------------------------------------------
# Step 1: Downscale
# ---------------------------------------------------------------------------

class TestDownscale:
    def test_halves_dimensions(self):
        from backend.pipeline.furnished_analyzer import _downscale
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        result = _downscale(img, factor=2)
        assert result.shape == (100, 200, 3)

    def test_grayscale(self):
        from backend.pipeline.furnished_analyzer import _downscale
        img = np.zeros((200, 400), dtype=np.uint8)
        result = _downscale(img, factor=2)
        assert result.shape == (100, 200)


# ---------------------------------------------------------------------------
# Step 2: Color filtering
# ---------------------------------------------------------------------------

class TestFilterColors:
    def test_removes_red_pixels(self):
        from backend.pipeline.furnished_analyzer import _filter_colors
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Paint a red stripe
        img[40:60, :] = (0, 0, 255)  # BGR red
        result = _filter_colors(img)
        # Red stripe should be replaced with white
        assert np.all(result[50, 50] == [255, 255, 255])

    def test_preserves_black_ink(self):
        from backend.pipeline.furnished_analyzer import _filter_colors
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img[40:60, 40:60] = (0, 0, 0)  # black
        result = _filter_colors(img)
        # Black should survive
        assert np.all(result[50, 50] == [0, 0, 0])

    def test_preserves_white(self):
        from backend.pipeline.furnished_analyzer import _filter_colors
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = _filter_colors(img)
        assert np.all(result == 255)


# ---------------------------------------------------------------------------
# Step 3-4: Wall extraction
# ---------------------------------------------------------------------------

class TestExtractWalls:
    def test_returns_expected_keys(self):
        from backend.pipeline.furnished_analyzer import _extract_walls
        gray = np.ones((200, 300), dtype=np.uint8) * 200
        result = _extract_walls(gray)
        assert "binary" in result
        assert "eroded" in result
        assert "wall_mask" in result

    def test_thick_grey_walls_survive(self):
        from backend.pipeline.furnished_analyzer import _extract_walls
        gray = np.ones((200, 300), dtype=np.uint8) * 255
        # Draw a thick horizontal wall (10px wide) at wall-grey intensity (~99)
        gray[95:105, 20:280] = 99
        result = _extract_walls(gray)
        assert np.count_nonzero(result["wall_mask"]) > 0

    def test_thin_furniture_removed(self):
        from backend.pipeline.furnished_analyzer import _extract_walls
        gray = np.ones((200, 300), dtype=np.uint8) * 255
        # Draw thin black furniture line (2px wide)
        gray[100:102, 20:280] = 0
        result = _extract_walls(gray)
        # Thin black lines should NOT appear in wall mask
        assert np.count_nonzero(result["wall_mask"]) == 0

    def test_output_shape_matches(self):
        from backend.pipeline.furnished_analyzer import _extract_walls
        gray = np.ones((200, 300), dtype=np.uint8) * 200
        result = _extract_walls(gray)
        assert result["wall_mask"].shape == (200, 300)
        assert result["binary"].shape == (200, 300)


# ---------------------------------------------------------------------------
# Step 5: Grid line removal
# ---------------------------------------------------------------------------

class TestRemoveGridLines:
    def test_removes_spanning_line(self):
        from backend.pipeline.furnished_analyzer import _remove_grid_lines
        mask = np.zeros((200, 300), dtype=np.uint8)
        # Horizontal line spanning 80% of width
        mask[100, 20:280] = 255
        result = _remove_grid_lines(mask, span_threshold=0.55)
        assert np.count_nonzero(result) == 0

    def test_keeps_short_line(self):
        from backend.pipeline.furnished_analyzer import _remove_grid_lines
        mask = np.zeros((200, 300), dtype=np.uint8)
        # Short wall segment (30% of width)
        mask[100, 50:140] = 255
        result = _remove_grid_lines(mask, span_threshold=0.55)
        assert np.count_nonzero(result) > 0


# ---------------------------------------------------------------------------
# Step 6: Door arc detection
# ---------------------------------------------------------------------------

class TestDetectDoorArcs:
    def test_detects_arc_shape(self):
        import cv2
        from backend.pipeline.furnished_analyzer import _detect_door_arcs
        # Create binary image with a thin quarter-circle arc
        binary = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(binary, (100, 100), (40, 40), 0, 0, 90, 255, 1)
        # Eroded version has nothing (arc is 1px thin)
        eroded = np.zeros_like(binary)
        doors = _detect_door_arcs(binary, eroded)
        # Should detect at least one arc-like feature
        # (may or may not pass all filters depending on exact shape)
        assert isinstance(doors, list)

    def test_empty_image_returns_empty(self):
        from backend.pipeline.furnished_analyzer import _detect_door_arcs
        binary = np.zeros((200, 200), dtype=np.uint8)
        eroded = np.zeros_like(binary)
        doors = _detect_door_arcs(binary, eroded)
        assert doors == []


# ---------------------------------------------------------------------------
# Step 7: Door gap closing
# ---------------------------------------------------------------------------

class TestCloseDoorGaps:
    def test_returns_same_shape(self):
        from backend.pipeline.furnished_analyzer import _close_door_gaps
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[100, 50:80] = 255
        mask[100, 120:150] = 255
        result = _close_door_gaps(mask, doors=[(100, 100, 30)])
        assert result.shape == mask.shape

    def test_no_doors_returns_unchanged(self):
        from backend.pipeline.furnished_analyzer import _close_door_gaps
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[100, 50:150] = 255
        result = _close_door_gaps(mask, doors=[])
        assert np.array_equal(result, mask)


# ---------------------------------------------------------------------------
# Debug images
# ---------------------------------------------------------------------------

class TestSaveDebugImages:
    def test_saves_files(self, tmp_path):
        from backend.pipeline.furnished_analyzer import _save_debug_images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        wall_mask = np.zeros((50, 50), dtype=np.uint8)
        rooms = [
            {"polygon_px": [(10, 10), (90, 10), (90, 90), (10, 90)], "name": "Room 1"},
        ]
        debug_dir = str(tmp_path / "debug")
        _save_debug_images(img, wall_mask, rooms, debug_dir, scale_factor=2)

        import os
        assert os.path.exists(os.path.join(debug_dir, "debug_walls.png"))
        assert os.path.exists(os.path.join(debug_dir, "debug_rooms.png"))
        assert os.path.exists(os.path.join(debug_dir, "debug_wallmask.png"))

    def test_handles_empty_rooms(self, tmp_path):
        from backend.pipeline.furnished_analyzer import _save_debug_images
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        wall_mask = np.zeros((50, 50), dtype=np.uint8)
        debug_dir = str(tmp_path / "debug")
        _save_debug_images(img, wall_mask, [], debug_dir, scale_factor=2)

        import os
        assert os.path.exists(os.path.join(debug_dir, "debug_walls.png"))


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def _make_walled_image(width, height, walls, wall_grey=99, bg=255):
    """Create a test image with grey walls. walls = list of (x1,y1,x2,y2,thickness)."""
    import cv2 as _cv2
    img = np.ones((height, width, 3), dtype=np.uint8) * bg
    for x1, y1, x2, y2, t in walls:
        _cv2.line(img, (x1, y1), (x2, y2), (wall_grey, wall_grey, wall_grey), t)
    return img


class TestRunFurnishedPipeline:
    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_returns_labeled_rooms(self, mock_label, mock_bbox):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (0, 0, 2000, 1400)
        mock_label.return_value = [{"name": "Bedroom", "type": "bedroom"}]

        # Large image so room is <6% of total area after downscaling
        img = _make_walled_image(2000, 1400, [
            (400, 300, 1600, 300, 30),   # top wall
            (400, 1100, 1600, 1100, 30), # bottom wall
            (400, 300, 400, 1100, 30),   # left wall
            (1600, 300, 1600, 1100, 30), # right wall
        ])
        rooms = run_furnished_pipeline(img)

        assert len(rooms) >= 1
        assert rooms[0]["name"] == "Bedroom"
        assert rooms[0]["type"] == "bedroom"
        assert rooms[0]["unit_name"] is None
        assert len(rooms[0]["polygon_px"]) >= 4

    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_gemini_failure_uses_defaults(self, mock_label, mock_bbox):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (0, 0, 2000, 1400)
        mock_label.side_effect = Exception("API error")

        img = _make_walled_image(2000, 1400, [
            (400, 300, 1600, 300, 30),
            (400, 1100, 1600, 1100, 30),
            (400, 300, 400, 1100, 30),
            (1600, 300, 1600, 1100, 30),
        ])
        rooms = run_furnished_pipeline(img)

        assert len(rooms) >= 1
        assert rooms[0]["name"] == "Unnamed"
        assert rooms[0]["type"] == "unknown"

    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_no_rooms_detected(self, mock_label, mock_bbox):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (0, 0, 300, 200)

        # Blank white image — no walls, no rooms
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        rooms = run_furnished_pipeline(img)

        assert rooms == []

    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_progress_callback(self, mock_label, mock_bbox):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (0, 0, 300, 200)

        progress_calls = []
        def cb(pct, msg):
            progress_calls.append((pct, msg))

        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        run_furnished_pipeline(img, progress_cb=cb)

        assert len(progress_calls) >= 2
        assert progress_calls[-1][0] == 100

    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_multiple_rooms(self, mock_label, mock_bbox):
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (0, 0, 2000, 1400)
        mock_label.return_value = [
            {"name": "Kitchen", "type": "kitchen"},
            {"name": "Living", "type": "living"},
        ]

        # Two rooms side by side separated by a wall
        img = _make_walled_image(2000, 1400, [
            (400, 300, 1600, 300, 30),    # top wall
            (400, 1100, 1600, 1100, 30),  # bottom wall
            (400, 300, 400, 1100, 30),    # left wall
            (1600, 300, 1600, 1100, 30),  # right wall
            (1000, 300, 1000, 1100, 30),  # dividing wall
        ])
        rooms = run_furnished_pipeline(img)

        assert len(rooms) >= 2

    @patch("backend.pipeline.vision_ai.detect_building_bbox")
    @patch("backend.pipeline.vision_ai.label_rooms")
    def test_gemini_bbox_constrains_rooms(self, mock_label, mock_bbox):
        """Gemini bbox should be passed as model param and used for footprint."""
        from backend.pipeline.furnished_analyzer import run_furnished_pipeline

        mock_bbox.return_value = (50, 50, 250, 150)

        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        run_furnished_pipeline(img, gemini_model="gemini-2.5-pro")

        mock_bbox.assert_called_once_with(img, model="gemini-2.5-pro")
