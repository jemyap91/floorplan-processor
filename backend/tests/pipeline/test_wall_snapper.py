"""Tests for wall snapper — snap polygon vertices to nearest dark pixels."""
import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from backend.pipeline.wall_snapper import snap_polygon_to_walls, _find_nearest_dark_pixel


class TestFindNearestDarkPixel:
    """Test the per-vertex dark pixel search."""

    def test_finds_dark_pixel_nearby(self):
        """Vertex near a dark pixel cluster should snap to it."""
        img = np.ones((100, 100), dtype=np.uint8) * 255  # all white
        # Draw a dark wall line at x=50
        img[20:80, 48:53] = 0
        result = _find_nearest_dark_pixel(img, (45, 50), radius=20)
        # Should snap to somewhere in the dark region (x ~48-52)
        assert result is not None
        rx, ry = result
        assert 48 <= rx <= 52
        assert ry == 50

    def test_no_dark_pixel_returns_none(self):
        """If no dark pixels within radius, return None."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = _find_nearest_dark_pixel(img, (50, 50), radius=10)
        assert result is None

    def test_respects_radius(self):
        """Dark pixels outside radius should not be found."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[50, 80] = 0  # dark pixel far away
        result = _find_nearest_dark_pixel(img, (50, 50), radius=10)
        assert result is None

    def test_snaps_to_nearest(self):
        """Should snap to the closest dark pixel, not just any dark pixel."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[50, 55] = 0   # closer dark pixel
        img[50, 65] = 0   # farther dark pixel
        result = _find_nearest_dark_pixel(img, (50, 50), radius=20)
        assert result is not None
        rx, ry = result
        assert rx == 55
        assert ry == 50

    def test_edge_of_image(self):
        """Vertex near image edge should not crash."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[0:5, 0:5] = 0
        result = _find_nearest_dark_pixel(img, (2, 2), radius=10)
        assert result is not None

    def test_dark_threshold(self):
        """Pixels above the dark threshold should not count as dark."""
        img = np.ones((100, 100), dtype=np.uint8) * 200  # gray, not dark
        result = _find_nearest_dark_pixel(img, (50, 50), radius=20, dark_threshold=128)
        assert result is None


class TestSnapPolygonToWalls:
    """Test the full polygon snapping function."""

    def test_snaps_vertices_to_walls(self):
        """Polygon vertices near wall lines should snap to them."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        # Draw walls forming a rectangle at (50,50)-(150,150)
        img[50, 50:150] = 0    # top wall
        img[150, 50:150] = 0   # bottom wall
        img[50:150, 50] = 0    # left wall
        img[50:150, 150] = 0   # right wall

        # Input polygon slightly offset from walls
        polygon = [(55, 55), (145, 55), (145, 145), (55, 145)]
        result = snap_polygon_to_walls(polygon, img, radius=20)

        assert len(result) == 4
        # Each vertex should be closer to the wall than the original
        for (rx, ry), (ox, oy) in zip(result, polygon):
            # Snapped points should be on or very near the walls
            assert abs(rx - 50) <= 5 or abs(rx - 150) <= 5
            assert abs(ry - 50) <= 5 or abs(ry - 150) <= 5

    def test_no_walls_returns_original(self):
        """White image (no walls) should return the original polygon."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        polygon = [(50, 50), (150, 50), (150, 150), (50, 150)]
        result = snap_polygon_to_walls(polygon, img, radius=20)
        assert result == polygon

    def test_preserves_polygon_validity(self):
        """Snapped polygon must not self-intersect."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        # Draw walls that could cause crossing if naively snapped
        img[50, 40:60] = 0
        img[50, 140:160] = 0
        img[150, 40:60] = 0
        img[150, 140:160] = 0

        polygon = [(45, 55), (145, 55), (145, 145), (45, 145)]
        result = snap_polygon_to_walls(polygon, img, radius=20)

        shapely_poly = ShapelyPolygon(result)
        assert shapely_poly.is_valid

    def test_minimum_polygon_size(self):
        """Polygon with fewer than 3 points should be returned as-is."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        polygon = [(10, 10), (20, 20)]
        result = snap_polygon_to_walls(polygon, img, radius=10)
        assert result == polygon

    def test_custom_radius(self):
        """Larger radius should find more distant walls."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        img[50, 80] = 0  # dark pixel 30px away from vertex

        polygon = [(50, 50), (150, 50), (150, 150), (50, 150)]

        # Small radius: won't find it
        result_small = snap_polygon_to_walls(polygon, img, radius=10)
        assert result_small[0] == (50, 50)

        # Large radius: should find it
        result_large = snap_polygon_to_walls(polygon, img, radius=40)
        assert result_large[0] != (50, 50)

    def test_snapping_reverted_if_self_intersecting(self):
        """If snapping creates a self-intersection, revert to original polygon."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        # Create dark pixels that would pull vertices into a bowtie
        img[20, 60] = 0  # would pull vertex 0 right
        img[20, 40] = 0  # would pull vertex 1 left
        img[80, 60] = 0  # would pull vertex 2 right
        img[80, 40] = 0  # would pull vertex 3 left

        polygon = [(40, 20), (60, 20), (60, 80), (40, 80)]
        result = snap_polygon_to_walls(polygon, img, radius=25)

        # Result must be valid regardless
        shapely_poly = ShapelyPolygon(result)
        assert shapely_poly.is_valid
