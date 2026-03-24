import os
import pytest
from backend.database import Database
from backend.models.room import RoomData, ProjectData, ExcludedRegion

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    yield database
    database.close()

class TestDatabase:
    def test_create_project(self, db):
        proj = ProjectData(name="Test", pdf_path="/test.pdf", scale_px_per_meter=59.17, scale_source="auto")
        db.save_project(proj)
        loaded = db.get_project(proj.id)
        assert loaded is not None
        assert loaded.name == "Test"

    def test_save_and_get_rooms(self, db):
        proj = ProjectData(name="Test", pdf_path="/test.pdf")
        db.save_project(proj)
        room = RoomData(
            project_id=proj.id, name="Office", room_type="office",
            boundary_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            area_px=100.0, perimeter_px=40.0, centroid=(5.0, 5.0),
            boundary_lengths_px=[10.0, 10.0, 10.0, 10.0],
        )
        db.save_room(room)
        rooms = db.get_rooms(proj.id)
        assert len(rooms) == 1
        assert rooms[0].name == "Office"

    def test_update_room(self, db):
        proj = ProjectData(name="Test", pdf_path="/test.pdf")
        db.save_project(proj)
        room = RoomData(project_id=proj.id, name="Old Name")
        db.save_room(room)
        room.name = "New Name"
        db.update_room(room)
        rooms = db.get_rooms(proj.id)
        assert rooms[0].name == "New Name"

    def test_delete_room(self, db):
        proj = ProjectData(name="Test", pdf_path="/test.pdf")
        db.save_project(proj)
        room = RoomData(project_id=proj.id, name="ToDelete")
        db.save_room(room)
        db.delete_room(room.id)
        rooms = db.get_rooms(proj.id)
        assert len(rooms) == 0

    def test_save_excluded_region(self, db):
        proj = ProjectData(name="Test", pdf_path="/test.pdf")
        db.save_project(proj)
        region = ExcludedRegion(project_id=proj.id, region_type="table", bbox=[0, 0, 100, 50])
        db.save_excluded_region(region)
        regions = db.get_excluded_regions(proj.id)
        assert len(regions) == 1
        assert regions[0].region_type == "table"

    def test_list_projects(self, db):
        db.save_project(ProjectData(name="P1", pdf_path="/a.pdf"))
        db.save_project(ProjectData(name="P2", pdf_path="/b.pdf"))
        projects = db.list_projects()
        assert len(projects) == 2

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
