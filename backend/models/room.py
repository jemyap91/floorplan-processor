"""Data models for floorplan processing results."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

class RoomData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    name: str = "Unnamed"
    room_type: str = "unknown"
    boundary_polygon: list[list[float]] = []
    area_px: float = 0.0
    perimeter_px: float = 0.0
    area_sqm: Optional[float] = None
    perimeter_m: Optional[float] = None
    boundary_lengths_px: list[float] = []
    boundary_lengths_m: Optional[list[float]] = None
    centroid: tuple[float, float] = (0.0, 0.0)
    fill_color_rgb: list[int] | None = None
    source: str = "cv"
    confidence: float = 0.0

class ProjectData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    pdf_path: str = ""
    scale_px_per_meter: Optional[float] = None
    scale_source: str = "manual"

class ExcludedRegion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    region_type: str = "table"
    bbox: list[float] = []

def to_real_measurements(area_px: float, perimeter_px: float, boundary_lengths_px: list[float], px_per_meter: float) -> dict:
    scale_sq = px_per_meter ** 2
    return {
        "area_sqm": area_px / scale_sq,
        "perimeter_m": perimeter_px / px_per_meter,
        "boundary_lengths_m": [l / px_per_meter for l in boundary_lengths_px],
    }
