import pytest
from shapely.geometry import Polygon
from backend.models.room import RoomData, ProjectData, to_real_measurements

class TestRoomData:
    def test_create_room(self):
        room = RoomData(
            id="room-1", name="Office 201", room_type="office",
            boundary_polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
            area_px=5000.0, perimeter_px=300.0, centroid=(50.0, 25.0),
            boundary_lengths_px=[100.0, 50.0, 100.0, 50.0], source="cv", confidence=0.9,
        )
        assert room.name == "Office 201"
        assert room.area_px == 5000.0

    def test_room_to_dict(self):
        room = RoomData(
            id="room-1", name="Test", room_type="office",
            boundary_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            area_px=100.0, perimeter_px=40.0, centroid=(5.0, 5.0),
            boundary_lengths_px=[10.0, 10.0, 10.0, 10.0],
        )
        d = room.model_dump()
        assert d["name"] == "Test"
        assert d["boundary_polygon"] == [[0, 0], [10, 0], [10, 10], [0, 10]]

class TestRealMeasurements:
    def test_convert_area(self):
        result = to_real_measurements(
            area_px=10000.0, perimeter_px=400.0,
            boundary_lengths_px=[100.0, 100.0, 100.0, 100.0], px_per_meter=100.0,
        )
        assert abs(result["area_sqm"] - 1.0) < 0.01
        assert abs(result["perimeter_m"] - 4.0) < 0.01
        assert len(result["boundary_lengths_m"]) == 4
        assert abs(result["boundary_lengths_m"][0] - 1.0) < 0.01

class TestProjectData:
    def test_create_project(self):
        proj = ProjectData(
            id="proj-1", name="Test Building", pdf_path="/path/to/file.pdf",
            scale_px_per_meter=59.17, scale_source="auto",
        )
        assert proj.name == "Test Building"
