"""Tests for color-based room segmentation."""
import numpy as np
import pytest
from backend.pipeline.color_segmenter import (
    segment_rooms_by_color,
    merge_room_lists,
)


def _make_colored_floorplan(w=400, h=300):
    """Create a test image with colored room zones separated by dark walls."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    # Room 1: pink rectangle (50,50)-(150,130)
    img[50:130, 50:150] = [240, 200, 200]

    # Room 2: yellow rectangle (50,150)-(130,280)
    img[50:130, 150:280] = [240, 240, 200]

    # Room 3: green rectangle (160,50)-(250,200)
    img[160:250, 50:200] = [200, 240, 200]

    # Dark wall lines between rooms
    img[128:132, 40:290] = [30, 30, 30]  # horizontal wall
    img[40:260, 148:152] = [30, 30, 30]  # vertical wall

    return img


class TestSegmentRoomsByColor:
    def test_detects_colored_rooms(self):
        img = _make_colored_floorplan()
        rooms = segment_rooms_by_color(img, min_area_ratio=0.001)
        assert len(rooms) >= 2

    def test_rooms_have_required_fields(self):
        img = _make_colored_floorplan()
        rooms = segment_rooms_by_color(img, min_area_ratio=0.001)
        assert len(rooms) > 0
        for room in rooms:
            assert "polygon" in room
            assert "area_px" in room
            assert "perimeter_px" in room
            assert "centroid" in room
            assert "boundary_lengths_px" in room
            assert room["area_px"] > 0

    def test_excludes_regions(self):
        img = _make_colored_floorplan()
        # Exclude the entire right half
        excluded = [(200, 0, 200, 300)]
        rooms = segment_rooms_by_color(img, min_area_ratio=0.001, excluded_regions=excluded)
        # Should only find rooms on the left side
        for room in rooms:
            cx = room["centroid"][0]
            assert cx < 200

    def test_empty_image_returns_empty(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        rooms = segment_rooms_by_color(img)
        assert rooms == []

    def test_filters_small_regions(self):
        img = _make_colored_floorplan()
        # With very high min_area, nothing should pass
        rooms = segment_rooms_by_color(img, min_area_ratio=0.5)
        assert len(rooms) == 0

    def test_wall_lines_split_same_color_rooms(self):
        """Two same-color rectangles separated by a wall should be 2 rooms."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        img[30:170, 30:130] = [240, 200, 200]  # pink left
        img[30:170, 170:270] = [240, 200, 200]  # pink right (same color)
        img[20:180, 148:152] = [20, 20, 20]     # wall between them
        rooms = segment_rooms_by_color(img, min_area_ratio=0.001)
        assert len(rooms) >= 2


class TestMergeRoomLists:
    def test_no_overlap_keeps_all(self):
        from shapely.geometry import Polygon as SPoly
        r1 = {"polygon": SPoly([(0,0),(10,0),(10,10),(0,10)]), "area_px": 100}
        r2 = {"polygon": SPoly([(20,20),(30,20),(30,30),(20,30)]), "area_px": 100}
        merged = merge_room_lists([r1], [r2])
        assert len(merged) == 2

    def test_overlapping_wall_room_removed(self):
        from shapely.geometry import Polygon as SPoly
        color_room = {"polygon": SPoly([(0,0),(10,0),(10,10),(0,10)]), "area_px": 100}
        wall_room = {"polygon": SPoly([(1,1),(9,1),(9,9),(1,9)]), "area_px": 64}
        merged = merge_room_lists([color_room], [wall_room])
        assert len(merged) == 1  # wall room is inside color room
