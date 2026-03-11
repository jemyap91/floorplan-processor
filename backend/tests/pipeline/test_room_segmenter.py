import numpy as np
import pytest
from shapely.geometry import Polygon
from backend.pipeline.room_segmenter import segment_rooms

class TestSegmentRooms:
    def _make_rooms_image(self, w=400, h=400):
        img = np.zeros((h, w), dtype=np.uint8)
        img[50:55, 50:350] = 255
        img[250:255, 50:350] = 255
        img[50:255, 50:55] = 255
        img[50:255, 345:350] = 255
        img[50:255, 195:200] = 255
        return img

    def test_returns_room_list(self):
        img = self._make_rooms_image()
        rooms = segment_rooms(img)
        assert isinstance(rooms, list)
        assert len(rooms) >= 2

    def test_rooms_have_polygons(self):
        img = self._make_rooms_image()
        rooms = segment_rooms(img)
        for room in rooms:
            assert "polygon" in room
            assert isinstance(room["polygon"], Polygon)
            assert room["polygon"].is_valid

    def test_rooms_have_area(self):
        img = self._make_rooms_image()
        rooms = segment_rooms(img)
        for room in rooms:
            assert "area_px" in room
            assert room["area_px"] > 0

    def test_rooms_have_centroid(self):
        img = self._make_rooms_image()
        rooms = segment_rooms(img)
        for room in rooms:
            assert "centroid" in room
            assert len(room["centroid"]) == 2

    def test_filters_small_regions(self):
        img = self._make_rooms_image()
        rooms = segment_rooms(img, min_area_px=100)
        for room in rooms:
            assert room["area_px"] >= 100

    def test_empty_image_returns_empty(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        rooms = segment_rooms(img)
        assert rooms == []
