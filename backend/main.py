"""FastAPI backend for the floorplan processor pipeline."""
import io
import os
import tempfile
import csv
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon
from shapely.validation import make_valid

from backend.database import Database
from backend.models.room import ExcludedRegion, ProjectData, RoomData, to_real_measurements
from backend.pipeline.extractor import extract_floorplan
from backend.pipeline.preprocessor import preprocess_image
from backend.pipeline.room_segmenter import segment_rooms
from backend.pipeline.scale_detector import detect_scale
from backend.pipeline.vision_ai import classify_regions, label_rooms
from backend.pipeline.wall_detector import detect_walls

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Floorplan Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory image cache: project_id -> np.ndarray (RGB)
_image_cache: dict[str, np.ndarray] = {}


def get_db() -> Database:
    db_path = os.environ.get("DB_PATH", "floorplan.db")
    return Database(db_path)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RoomUpdateRequest(BaseModel):
    name: Optional[str] = None
    room_type: Optional[str] = None
    polygon: Optional[list[list[float]]] = None


class ScaleUpdateRequest(BaseModel):
    px_per_meter: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Projects ---------------------------------------------------------------


@app.get("/api/projects")
def list_projects():
    db = get_db()
    projects = db.list_projects()
    return [p.model_dump() for p in projects]


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()


# --- Process ----------------------------------------------------------------


@app.post("/api/process")
async def process_pdf(
    file: UploadFile = File(...),
    page_num: int = Form(0),
    manual_px_per_meter: Optional[float] = Form(None),
):
    """Upload a PDF, run the full pipeline, persist results, and return them."""
    # Save upload to temp file
    suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1. Extract
        extraction = extract_floorplan(tmp_path, page_num=page_num)
        image: np.ndarray = extraction["image"]

        # 2. Detect scale
        scale_result = detect_scale(
            text=extraction["text"],
            image=image,
            manual_px_per_meter=manual_px_per_meter,
        )
        px_per_meter: Optional[float] = None
        scale_source = "manual"
        if scale_result:
            px_per_meter = scale_result.get("px_per_meter")
            scale_source = scale_result.get("source", "auto")

        # 3. Classify regions (Gemini — may fall back to full-image)
        region_result = classify_regions(image)
        excluded_regions_raw = region_result.get("excluded_regions", [])

        # 4. Preprocess
        prep = preprocess_image(image)
        binary: np.ndarray = prep["binary"]

        # 5. Mask excluded regions from binary BEFORE wall detection
        excluded_bboxes = []
        for er in excluded_regions_raw:
            rx = int(er.get("x", 0))
            ry = int(er.get("y", 0))
            rw = int(er.get("width", 0))
            rh = int(er.get("height", 0))
            binary[ry : ry + rh, rx : rx + rw] = 0
            excluded_bboxes.append((rx, ry, rw, rh))

        # 6. Detect walls
        wall_result = detect_walls(binary)

        # 7. Segment rooms
        raw_rooms = segment_rooms(
            wall_result["wall_mask"], excluded_regions=excluded_bboxes
        )

        # 8. Label rooms with vision AI
        labeled = label_rooms(image, raw_rooms)

        # 9. Build project
        project_name = os.path.splitext(os.path.basename(file.filename or "upload.pdf"))[0]
        project = ProjectData(
            name=project_name,
            pdf_path=tmp_path,
            scale_px_per_meter=px_per_meter,
            scale_source=scale_source,
        )

        db = get_db()
        db.save_project(project)

        # 10. Save excluded regions
        for er, bbox in zip(excluded_regions_raw, excluded_bboxes):
            excl = ExcludedRegion(
                project_id=project.id,
                region_type=er.get("type", "table"),
                bbox=list(bbox),
            )
            db.save_excluded_region(excl)

        # 11. Build and save rooms
        rooms_out = []
        for i, raw in enumerate(raw_rooms):
            label_info = labeled[i] if i < len(labeled) else {}

            # Convert Shapely polygon exterior coords to [[x,y], ...]
            poly: Polygon = raw["polygon"]
            polygon_coords = [[x, y] for x, y in list(poly.exterior.coords)]

            real = {}
            if px_per_meter:
                real = to_real_measurements(
                    raw["area_px"],
                    raw["perimeter_px"],
                    raw["boundary_lengths_px"],
                    px_per_meter,
                )

            room = RoomData(
                project_id=project.id,
                name=label_info.get("name", "Unnamed"),
                room_type=label_info.get("type", "unknown"),
                boundary_polygon=polygon_coords,
                area_px=raw["area_px"],
                perimeter_px=raw["perimeter_px"],
                centroid=raw["centroid"],
                boundary_lengths_px=raw["boundary_lengths_px"],
                area_sqm=real.get("area_sqm"),
                perimeter_m=real.get("perimeter_m"),
                boundary_lengths_m=real.get("boundary_lengths_m"),
                source="cv",
                confidence=float(label_info.get("confidence", 0.0)),
            )
            db.save_room(room)
            rooms_out.append(room.model_dump())

        # Cache the extracted image for later serving
        _image_cache[project.id] = image

        return {
            "project_id": project.id,
            "rooms": rooms_out,
            "scale": scale_result,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# --- Rooms ------------------------------------------------------------------


@app.get("/api/projects/{project_id}/rooms")
def get_rooms(project_id: str):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    rooms = db.get_rooms(project_id)
    return [r.model_dump() for r in rooms]


@app.put("/api/rooms/{room_id}")
def update_room(room_id: str, body: RoomUpdateRequest):
    db = get_db()
    # Find the room across all projects (get by id via a small helper)
    room = _get_room_by_id(db, room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    if body.name is not None:
        room.name = body.name
    if body.room_type is not None:
        room.room_type = body.room_type

    if body.polygon is not None:
        # Recalculate geometry from the new polygon
        points = [(float(p[0]), float(p[1])) for p in body.polygon]
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid polygon: {exc}") from exc

        coords = list(poly.exterior.coords)
        boundary_lengths = []
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            boundary_lengths.append(float(np.sqrt(dx**2 + dy**2)))

        centroid = poly.centroid
        room.boundary_polygon = [[x, y] for x, y in coords]
        room.area_px = float(poly.area)
        room.perimeter_px = float(poly.length)
        room.centroid = (float(centroid.x), float(centroid.y))
        room.boundary_lengths_px = boundary_lengths
        room.source = "corrected"

        # Recalculate real measurements if scale is available
        project = db.get_project(room.project_id)
        if project and project.scale_px_per_meter:
            real = to_real_measurements(
                room.area_px,
                room.perimeter_px,
                room.boundary_lengths_px,
                project.scale_px_per_meter,
            )
            room.area_sqm = real["area_sqm"]
            room.perimeter_m = real["perimeter_m"]
            room.boundary_lengths_m = real["boundary_lengths_m"]

    db.update_room(room)
    return room.model_dump()


@app.delete("/api/rooms/{room_id}")
def delete_room(room_id: str):
    db = get_db()
    room = _get_room_by_id(db, room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    db.delete_room(room_id)
    return {"deleted": room_id}


# --- Scale ------------------------------------------------------------------


@app.put("/api/projects/{project_id}/scale")
def update_scale(project_id: str, body: ScaleUpdateRequest):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    project.scale_px_per_meter = body.px_per_meter
    project.scale_source = "manual"
    db.save_project(project)

    # Recalculate all room measurements
    rooms = db.get_rooms(project_id)
    for room in rooms:
        real = to_real_measurements(
            room.area_px,
            room.perimeter_px,
            room.boundary_lengths_px,
            body.px_per_meter,
        )
        room.area_sqm = real["area_sqm"]
        room.perimeter_m = real["perimeter_m"]
        room.boundary_lengths_m = real["boundary_lengths_m"]
        db.update_room(room)

    return {
        "project_id": project_id,
        "px_per_meter": body.px_per_meter,
        "rooms_updated": len(rooms),
    }


# --- Image ------------------------------------------------------------------


@app.get("/api/projects/{project_id}/image")
def get_image(project_id: str):
    if project_id not in _image_cache:
        raise HTTPException(status_code=404, detail="Image not available for this project")

    import cv2

    image = _image_cache[project_id]
    # Convert RGB -> BGR for OpenCV encoding
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buf.tobytes(), media_type="image/png")


# --- Export -----------------------------------------------------------------


@app.get("/api/export/{project_id}")
def export_project(project_id: str, format: str = "json"):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = db.get_rooms(project_id)

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id",
                "name",
                "room_type",
                "area_px",
                "perimeter_px",
                "area_sqm",
                "perimeter_m",
                "centroid_x",
                "centroid_y",
                "source",
                "confidence",
            ],
        )
        writer.writeheader()
        for room in rooms:
            writer.writerow(
                {
                    "id": room.id,
                    "name": room.name,
                    "room_type": room.room_type,
                    "area_px": room.area_px,
                    "perimeter_px": room.perimeter_px,
                    "area_sqm": room.area_sqm,
                    "perimeter_m": room.perimeter_m,
                    "centroid_x": room.centroid[0],
                    "centroid_y": room.centroid[1],
                    "source": room.source,
                    "confidence": room.confidence,
                }
            )
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{project_id}.csv"'
            },
        )

    # Default: JSON
    return {
        "project": project.model_dump(),
        "rooms": [r.model_dump() for r in rooms],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_room_by_id(db: Database, room_id: str) -> Optional[RoomData]:
    """Retrieve a single room by its ID across all projects."""
    projects = db.list_projects()
    for project in projects:
        rooms = db.get_rooms(project.id)
        for room in rooms:
            if room.id == room_id:
                return room
    return None
