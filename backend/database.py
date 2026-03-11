"""SQLite database for storing floorplan processing results."""
import json
import sqlite3
from backend.models.room import RoomData, ProjectData, ExcludedRegion

class Database:
    def __init__(self, db_path: str = "floorplan.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT, created_at TEXT,
                pdf_path TEXT, scale_px_per_meter REAL, scale_source TEXT
            );
            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id),
                name TEXT, room_type TEXT, boundary_polygon TEXT,
                area_px REAL, perimeter_px REAL, area_sqm REAL, perimeter_m REAL,
                boundary_lengths_px TEXT, boundary_lengths_m TEXT,
                centroid_x REAL, centroid_y REAL, source TEXT, confidence REAL
            );
            CREATE TABLE IF NOT EXISTS excluded_regions (
                id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id),
                region_type TEXT, bbox TEXT
            );
        """)

    def save_project(self, project: ProjectData):
        self.conn.execute(
            "INSERT OR REPLACE INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            (project.id, project.name, project.created_at.isoformat(),
             project.pdf_path, project.scale_px_per_meter, project.scale_source))
        self.conn.commit()

    def get_project(self, project_id: str) -> ProjectData | None:
        row = self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row: return None
        return ProjectData(id=row["id"], name=row["name"], pdf_path=row["pdf_path"],
            scale_px_per_meter=row["scale_px_per_meter"], scale_source=row["scale_source"] or "manual")

    def list_projects(self) -> list[ProjectData]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [ProjectData(id=r["id"], name=r["name"], pdf_path=r["pdf_path"],
            scale_px_per_meter=r["scale_px_per_meter"], scale_source=r["scale_source"] or "manual") for r in rows]

    def save_room(self, room: RoomData):
        self.conn.execute(
            "INSERT OR REPLACE INTO rooms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (room.id, room.project_id, room.name, room.room_type,
             json.dumps(room.boundary_polygon), room.area_px, room.perimeter_px,
             room.area_sqm, room.perimeter_m,
             json.dumps(room.boundary_lengths_px),
             json.dumps(room.boundary_lengths_m) if room.boundary_lengths_m else None,
             room.centroid[0], room.centroid[1], room.source, room.confidence))
        self.conn.commit()

    def get_rooms(self, project_id: str) -> list[RoomData]:
        rows = self.conn.execute("SELECT * FROM rooms WHERE project_id = ?", (project_id,)).fetchall()
        return [RoomData(
            id=r["id"], project_id=r["project_id"], name=r["name"], room_type=r["room_type"],
            boundary_polygon=json.loads(r["boundary_polygon"]) if r["boundary_polygon"] else [],
            area_px=r["area_px"] or 0, perimeter_px=r["perimeter_px"] or 0,
            area_sqm=r["area_sqm"], perimeter_m=r["perimeter_m"],
            boundary_lengths_px=json.loads(r["boundary_lengths_px"]) if r["boundary_lengths_px"] else [],
            boundary_lengths_m=json.loads(r["boundary_lengths_m"]) if r["boundary_lengths_m"] else None,
            centroid=(r["centroid_x"] or 0, r["centroid_y"] or 0),
            source=r["source"] or "cv", confidence=r["confidence"] or 0) for r in rows]

    def update_room(self, room: RoomData):
        self.save_room(room)

    def delete_room(self, room_id: str):
        self.conn.execute("DELETE FROM rooms WHERE id = ?", (room_id,))
        self.conn.commit()

    def save_excluded_region(self, region: ExcludedRegion):
        self.conn.execute("INSERT OR REPLACE INTO excluded_regions VALUES (?, ?, ?, ?)",
            (region.id, region.project_id, region.region_type, json.dumps(region.bbox)))
        self.conn.commit()

    def get_excluded_regions(self, project_id: str) -> list[ExcludedRegion]:
        rows = self.conn.execute("SELECT * FROM excluded_regions WHERE project_id = ?", (project_id,)).fetchall()
        return [ExcludedRegion(id=r["id"], project_id=r["project_id"],
            region_type=r["region_type"], bbox=json.loads(r["bbox"]) if r["bbox"] else []) for r in rows]

    def close(self):
        self.conn.close()
