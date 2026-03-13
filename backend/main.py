"""FastAPI backend for the floorplan processor pipeline."""
import asyncio
import io
import os
import tempfile
import csv
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon
from shapely.validation import make_valid

from backend.database import Database
from backend.models.room import ExcludedRegion, ProjectData, RoomData, to_real_measurements
from backend.pipeline.extractor import extract_floorplan, extract_from_image, IMAGE_EXTENSIONS
from backend.pipeline.preprocessor import preprocess_image, detect_margin_regions
from backend.pipeline.color_segmenter import segment_rooms_by_color, merge_room_lists
from backend.pipeline.room_segmenter import segment_rooms
from backend.pipeline.scale_detector import detect_scale
from backend.pipeline.vision_ai import classify_regions, match_gemini_labels_to_cv_rooms, label_rooms
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

# In-memory image cache (fallback when DB image not available)
_image_cache: dict[str, np.ndarray] = {}

# Processing progress: job_id -> {step, percent, message}
_progress: dict[str, dict] = {}


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
    result = []
    for p in projects:
        d = p.model_dump()
        d["room_count"] = len(db.get_rooms(p.id))
        result.append(d)
    return result


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    d = project.model_dump()
    d["room_count"] = len(db.get_rooms(project_id))
    return d


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete_project(project_id)
    return {"deleted": project_id}


# --- Progress ---------------------------------------------------------------


@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    return _progress.get(job_id, {"step": 0, "percent": 0, "message": "Waiting..."})


# --- Process ----------------------------------------------------------------


@app.post("/api/process")
async def process_pdf(
    file: UploadFile = File(...),
    page_num: int = Form(0),
    manual_px_per_meter: Optional[float] = Form(None),
    mode: str = Form("hybrid"),
    job_id: str = Form(""),
):
    """Upload a PDF or image, run the pipeline, persist results, and return them.

    mode: "hybrid" (CV + Gemini labelling) or "gemini" (pure Gemini extraction)
    Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, WebP
    """
    # Save upload to temp file
    suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def _update(percent: int, message: str):
        if job_id:
            _progress[job_id] = {"percent": percent, "message": message}

    def _run_pipeline():
        """Run the full processing pipeline (CPU-bound, runs in a thread)."""
        # 1. Extract image from PDF or image file
        _update(5, "Extracting image from file...")
        if suffix.lower() in IMAGE_EXTENSIONS:
            extraction = extract_from_image(tmp_path)
        else:
            extraction = extract_floorplan(tmp_path, page_num=page_num)
        image: np.ndarray = extraction["image"]
        img_h, img_w = image.shape[:2]

        # Cap max dimension to 8000px — balances detail vs processing time
        MAX_DIM = 8000
        if max(img_h, img_w) > MAX_DIM:
            _update(10, f"Resizing image ({img_w}x{img_h} -> {MAX_DIM}px)...")
            scale_factor = MAX_DIM / max(img_h, img_w)
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_h, img_w = new_h, new_w

        # 2. Detect scale from text
        _update(15, "Detecting scale...")
        scale_result = detect_scale(
            text=extraction["text"],
            image=image,
            manual_px_per_meter=manual_px_per_meter,
        )
        px_per_meter_val: float | None = None
        scale_source = "manual"
        if scale_result:
            px_per_meter_val = scale_result.get("px_per_meter")
            scale_source = scale_result.get("source", "auto")

        # Build project
        _update(20, "Setting up project...")
        project_name = os.path.splitext(os.path.basename(file.filename or "upload.pdf"))[0]
        project = ProjectData(
            name=project_name,
            pdf_path=tmp_path,
            scale_px_per_meter=px_per_meter_val,
            scale_source=scale_source,
        )
        db = get_db()
        db.save_project(project)

        if mode == "gemini":
            rooms_out = _process_gemini_mode(image, project, px_per_meter_val, db, _update)
        else:
            rooms_out = _process_hybrid_mode(image, project, px_per_meter_val, db, _update)

        # Persist image for later serving (DB + in-memory cache)
        _update(95, "Saving image...")
        _image_cache[project.id] = image
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, img_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        db.save_image(project.id, img_buf.tobytes())

        _update(100, "Done!")
        return {
            "project_id": project.id,
            "rooms": rooms_out,
            "scale": scale_result,
        }

    try:
        # Run in a thread so the event loop stays free for progress polling
        result = await asyncio.to_thread(_run_pipeline)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        # Clean up progress after a delay (let frontend read 100%)
        if job_id:
            import threading
            threading.Timer(10, lambda: _progress.pop(job_id, None)).start()


def _process_gemini_mode(
    image: np.ndarray,
    project: ProjectData,
    px_per_meter: Optional[float],
    db: Database,
    _update=lambda p, m: None,
) -> list[dict]:
    """Gemini-enhanced pipeline — CV detects boundaries, Gemini labels rooms."""
    # 1. Classify regions to exclude title block etc.
    _update(25, "Classifying regions (Gemini)...")
    region_result = classify_regions(image)
    excluded_regions_raw = region_result.get("excluded_regions", [])

    # 2. Build excluded bboxes
    excluded_bboxes = []
    for er in excluded_regions_raw:
        rx = int(er.get("x", 0))
        ry = int(er.get("y", 0))
        rw = int(er.get("width", 0))
        rh = int(er.get("height", 0))
        excluded_bboxes.append((rx, ry, rw, rh))

    # 3. Color-based room detection (primary — finds colored zone rooms)
    _update(35, "Detecting colored room zones...")
    color_rooms = segment_rooms_by_color(image, excluded_regions=excluded_bboxes)

    # 4. Wall-based detection for unfilled rooms (corridors, lift shafts)
    _update(45, "Detecting walls...")
    prep = preprocess_image(image)
    binary: np.ndarray = prep["binary"]
    for rx, ry, rw, rh in excluded_bboxes:
        binary[ry : ry + rh, rx : rx + rw] = 0
    margin_regions = detect_margin_regions(binary)
    for rx, ry, rw, rh in margin_regions:
        binary[ry : ry + rh, rx : rx + rw] = 0
        excluded_bboxes.append((rx, ry, rw, rh))

    # Mask out color-detected rooms so wall detector focuses on unfilled gaps
    for cr in color_rooms:
        poly = cr["polygon"]
        pts = np.array(list(poly.exterior.coords), dtype=np.int32)
        cv2.fillPoly(binary, [pts], 0)

    wall_result = detect_walls(binary)
    wall_rooms = segment_rooms(
        wall_result["wall_mask"], excluded_regions=excluded_bboxes
    )

    # 5. Merge color + wall rooms
    _update(55, "Merging detected rooms...")
    raw_rooms = merge_room_lists(color_rooms, wall_rooms)

    # 6. Label rooms by annotating image with numbered centroids
    _update(60, f"Labelling {len(raw_rooms)} rooms (Gemini)...")
    labeled = match_gemini_labels_to_cv_rooms([], raw_rooms, image)

    # 7. Save excluded regions
    _update(80, "Saving results...")
    for er, bbox in zip(excluded_regions_raw, excluded_bboxes):
        excl = ExcludedRegion(
            project_id=project.id,
            region_type=er.get("type", "table"),
            bbox=list(bbox),
        )
        db.save_excluded_region(excl)

    # 7. Build and save rooms, filtering out NOT_A_ROOM entries
    rooms_out = []
    for i, raw in enumerate(raw_rooms):
        label_info = labeled[i] if i < len(labeled) else {}
        name = label_info.get("name", "Unnamed")

        # Skip regions Gemini identified as not real rooms
        if name == "NOT_A_ROOM":
            continue

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

        # Sample fill color from image at centroid
        fill_color = _sample_fill_color(image, raw["centroid"], raw["polygon"])

        room = RoomData(
            project_id=project.id,
            name=name,
            room_type=label_info.get("type", "unknown"),
            boundary_polygon=polygon_coords,
            area_px=raw["area_px"],
            perimeter_px=raw["perimeter_px"],
            centroid=raw["centroid"],
            boundary_lengths_px=raw["boundary_lengths_px"],
            area_sqm=real.get("area_sqm"),
            perimeter_m=real.get("perimeter_m"),
            boundary_lengths_m=real.get("boundary_lengths_m"),
            fill_color_rgb=fill_color,
            source="gemini",
            confidence=float(label_info.get("confidence", 0.0)),
        )
        db.save_room(room)
        rooms_out.append(room.model_dump())

    return rooms_out


def _process_hybrid_mode(
    image: np.ndarray,
    project: ProjectData,
    px_per_meter: Optional[float],
    db: Database,
    _update=lambda p, m: None,
) -> list[dict]:
    """Hybrid CV + Gemini pipeline — walls detected by CV, rooms labelled by Gemini."""
    # 3. Classify regions (Gemini — may fall back to full-image)
    _update(25, "Classifying regions (Gemini)...")
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

    # 5b. CV-based margin detection (catches dense edge regions)
    margin_regions = detect_margin_regions(binary)
    for rx, ry, rw, rh in margin_regions:
        binary[ry : ry + rh, rx : rx + rw] = 0
        excluded_bboxes.append((rx, ry, rw, rh))

    # 6. Detect walls
    _update(40, "Detecting walls...")
    wall_result = detect_walls(binary)

    # 7. Segment rooms
    _update(50, "Segmenting rooms...")
    raw_rooms = segment_rooms(
        wall_result["wall_mask"], excluded_regions=excluded_bboxes
    )

    # 8. Label rooms with vision AI
    _update(60, f"Labelling {len(raw_rooms)} rooms (Gemini)...")
    labeled = label_rooms(image, raw_rooms)

    # 9. Save excluded regions
    _update(80, "Saving results...")
    for er, bbox in zip(excluded_regions_raw, excluded_bboxes):
        excl = ExcludedRegion(
            project_id=project.id,
            region_type=er.get("type", "table"),
            bbox=list(bbox),
        )
        db.save_excluded_region(excl)

    # 10. Build and save rooms
    rooms_out = []
    for i, raw in enumerate(raw_rooms):
        label_info = labeled[i] if i < len(labeled) else {}

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

        fill_color = _sample_fill_color(image, raw["centroid"], raw["polygon"])

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
            fill_color_rgb=fill_color,
            source="cv",
            confidence=float(label_info.get("confidence", 0.0)),
        )
        db.save_room(room)
        rooms_out.append(room.model_dump())

    return rooms_out


# --- Rooms ------------------------------------------------------------------


@app.get("/api/projects/{project_id}/rooms")
def get_rooms(project_id: str):
    db = get_db()
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    rooms = db.get_rooms(project_id)
    return [r.model_dump() for r in rooms]


class CreateRoomRequest(BaseModel):
    project_id: str
    name: str = "Unnamed"
    room_type: str = "unknown"
    boundary_polygon: list[list[float]]


@app.post("/api/rooms")
def create_room(body: CreateRoomRequest):
    db = get_db()
    project = db.get_project(body.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    points = [(float(p[0]), float(p[1])) for p in body.boundary_polygon]
    if len(points) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 vertices")

    poly = Polygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)

    coords = list(poly.exterior.coords)
    boundary_lengths = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        boundary_lengths.append(float(np.sqrt(dx**2 + dy**2)))

    centroid = poly.centroid
    polygon_coords = [[x, y] for x, y in coords]

    real = {}
    if project.scale_px_per_meter:
        real = to_real_measurements(
            float(poly.area), float(poly.length),
            boundary_lengths, project.scale_px_per_meter,
        )

    room = RoomData(
        project_id=body.project_id,
        name=body.name,
        room_type=body.room_type,
        boundary_polygon=polygon_coords,
        area_px=float(poly.area),
        perimeter_px=float(poly.length),
        centroid=(float(centroid.x), float(centroid.y)),
        boundary_lengths_px=boundary_lengths,
        area_sqm=real.get("area_sqm"),
        perimeter_m=real.get("perimeter_m"),
        boundary_lengths_m=real.get("boundary_lengths_m"),
        source="user",
        confidence=1.0,
    )
    db.save_room(room)
    return room.model_dump()


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
        room.source = "user"

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
    # Try in-memory cache first
    if project_id in _image_cache:
        image = _image_cache[project_id]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            return Response(content=buf.tobytes(), media_type="image/jpeg")

    # Fall back to database
    db = get_db()
    image_bytes = db.get_image(project_id)
    if image_bytes:
        return Response(content=image_bytes, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Image not available for this project")


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

    if format == "xlsx":
        from backend.export import build_excel

        xlsx_bytes = build_excel(project, rooms)
        filename = (project.name or project_id).replace(" ", "_") + ".xlsx"
        return Response(
            content=xlsx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Default: JSON
    return {
        "project": project.model_dump(),
        "rooms": [r.model_dump() for r in rooms],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_fill_color(
    image: np.ndarray,
    centroid: tuple[float, float],
    polygon,
) -> list[int] | None:
    """Sample the median fill color from the image within a room polygon."""
    h, w = image.shape[:2]
    cx, cy = int(centroid[0]), int(centroid[1])
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    # Sample a small region around the centroid (faster than masking full polygon)
    radius = 15
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius)
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius)
    patch = image[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    # Filter to non-dark, non-white pixels (the fill color)
    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    brightness = (r.astype(int) + g.astype(int) + b.astype(int)) // 3
    mask = (brightness > 50) & (brightness < 245)
    colored_pixels = patch[mask]

    if len(colored_pixels) == 0:
        return [int(image[cy, cx, 0]), int(image[cy, cx, 1]), int(image[cy, cx, 2])]

    median = np.median(colored_pixels, axis=0).astype(int)
    return [int(median[0]), int(median[1]), int(median[2])]


def _get_room_by_id(db: Database, room_id: str) -> Optional[RoomData]:
    """Retrieve a single room by its ID across all projects."""
    projects = db.list_projects()
    for project in projects:
        rooms = db.get_rooms(project.id)
        for room in rooms:
            if room.id == room_id:
                return room
    return None
