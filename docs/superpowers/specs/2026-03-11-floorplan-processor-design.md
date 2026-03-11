# Floorplan Processor — Design Spec

## Problem

Architectural floorplan PDFs contain raster images of building layouts alongside schedule tables, legend blocks, and title blocks. We need a tool that:

1. Extracts the floorplan image from the PDF
2. Differentiates the actual floor layout from tables/legends/title blocks
3. Detects room boundaries as vector polygons
4. Reads room names and classifies room types
5. Calculates room areas, perimeters, and boundary segment lengths using a detected or user-provided scale
6. Provides a GUI for reviewing and correcting results
7. Exports structured room data to a database

## Input Characteristics

- Always raster images embedded in PDFs (not vector CAD)
- Varying formats and sources (not a single consistent template)
- May contain schedule/legend tables and title blocks alongside the floor layout
- Scale references may be embedded as text, scale bars, or dimension lines

## Approach: Hybrid CV + Vision AI

OpenCV handles precise geometry (wall detection, room segmentation, polygon extraction). Gemini Flash (free tier) handles semantics (region classification, room naming, table detection).

## Architecture

### Pipeline

1. **PDF Extraction** — PyMuPDF extracts raster image + metadata
2. **Pre-processing** — Denoise, adaptive threshold binarization, morphological closing, deskew
3. **Vision AI Pass 1** — Gemini Flash classifies regions: floor layout vs tables/legends/title blocks (bounding boxes)
4. **Wall Detection (OpenCV)** — Morphological line extraction (H/V kernels) + Hough transform for angled walls → wall segment merging → intersection graph
5. **Room Segmentation (OpenCV)** — Binary wall mask → flood-fill enclosed regions → filter by area → contour extraction → Douglas-Peucker simplification → Shapely polygons
6. **Scale Detection** — Auto-detect: OCR for scale text, detect scale bar graphics, find dimension lines. Fallback: user provides scale manually via GUI
7. **Vision AI Pass 2** — Send room crops/overlays to Gemini Flash → read room names, classify room types, flag suspect polygons
8. **GUI Review** — User reviews, corrects boundaries/names, confirms results
9. **Export** — Structured JSON/CSV, annotated PDF, SQLite database, GeoJSON

### Tech Stack

**Backend (Python):**
- PyMuPDF (fitz) — PDF parsing & image extraction
- OpenCV — wall detection, room segmentation
- NumPy / Shapely — geometry & polygon operations
- Google Generative AI SDK — Gemini 2.0 Flash (free tier: 15 RPM, 1M tokens/min)
- FastAPI — backend API server
- SQLite — local database

**Frontend (Web):**
- React + TypeScript — UI framework
- Fabric.js — canvas rendering with pan/zoom + polygon editing
- Electron — desktop wrapper
- TailwindCSS — styling

### Project Structure

```
floorplan-processor/
├── backend/
│   ├── main.py                 # FastAPI app entry
│   ├── pipeline/
│   │   ├── extractor.py        # PDF → image extraction
│   │   ├── preprocessor.py     # Image cleanup, binarization
│   │   ├── wall_detector.py    # OpenCV wall detection
│   │   ├── room_segmenter.py   # Flood-fill → room polygons
│   │   ├── scale_detector.py   # Auto-detect scale bars
│   │   └── vision_ai.py        # Gemini Flash integration
│   ├── models/
│   │   └── room.py             # Room data model
│   └── database.py             # SQLite operations
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FloorplanCanvas.tsx  # Main canvas with pan/zoom
│   │   │   ├── RoomEditor.tsx       # Boundary editing tools
│   │   │   ├── RoomTable.tsx        # Room data table/list
│   │   │   └── Toolbar.tsx          # Controls & actions
│   │   └── App.tsx
│   └── electron/
│       └── main.ts             # Electron entry
└── docs/
```

## Data Model (SQLite)

### projects
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| name | TEXT | Project name |
| created_at | TIMESTAMP | Creation time |
| pdf_path | TEXT | Path to source PDF |
| scale_px_per_meter | REAL | Pixels per meter |
| scale_source | TEXT | 'auto' or 'manual' |

### rooms
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| project_id | TEXT FK | References projects |
| name | TEXT | e.g. "Office 201" |
| room_type | TEXT | e.g. "office", "bathroom" |
| boundary_polygon | JSON | [[x1,y1],[x2,y2],...] |
| area_sqm | REAL | Area in square meters |
| perimeter_m | REAL | Perimeter in meters |
| boundary_lengths | JSON | [3.2, 5.1, 3.2, 5.1] per segment |
| centroid_x | REAL | Center X coordinate |
| centroid_y | REAL | Center Y coordinate |
| source | TEXT | 'cv', 'manual', or 'corrected' |
| confidence | REAL | 0.0–1.0 from AI |

### excluded_regions
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | UUID |
| project_id | TEXT FK | References projects |
| region_type | TEXT | 'table', 'legend', 'title_block' |
| bbox | JSON | [x, y, width, height] |

## GUI Design

Three-panel layout:

- **Left sidebar:** Room list with color-coded entries showing name, area, perimeter. Rooms with missing labels flagged with warning. Summary stats at bottom.
- **Center canvas:** Zoomable/pannable floorplan with colored polygon overlays. Toolbar for zoom, edit mode, add room, delete, undo. Scroll to zoom, drag to pan, click room to select.
- **Right panel:** Selected room detail editor — editable name/type fields, computed area/perimeter/segment lengths, vertex count, detection source indicator.

### Key Interactions
- **Import:** Open/drop PDF → progress bar through pipeline stages → results appear
- **Edit:** Click to select → drag vertices → double-click to add/remove vertices → right-click for merge/split/delete
- **Scale calibration:** Auto-detect first → if not found, user draws reference line and enters real length → all measurements recalculate
- **Export:** CSV/JSON, annotated PDF, SQLite DB, GeoJSON

## Gemini Flash Integration

**Pass 1 — Region Classification:**
Send full floorplan image. Prompt returns structured JSON with bounding boxes for: floor layout regions, table/legend regions, title blocks, scale bar locations.

**Pass 2 — Room Labeling:**
After CV extracts room polygons, send annotated image (or room crops). Prompt returns JSON mapping room_id → {name, type, confidence}. Flags suspect polygons.

Free tier limits: 15 requests/minute, 1M tokens/minute — sufficient for normal use (1-2 calls per floorplan).

## Error Handling

- **Gemini API failure:** If Pass 1 fails, skip region classification and run CV on the full image (may produce false rooms from tables — user corrects in GUI). If Pass 2 fails, rooms are left unnamed — user labels manually.
- **Wall detection failure:** If too few or too many walls detected, surface a warning in GUI with option to adjust sensitivity parameters and re-run.
- **Multi-page PDFs:** Process first page by default. If multiple pages detected, prompt user to select which page(s) to process.
- **Poor scan quality:** Pre-processing applies adaptive thresholding and denoising. If result is still poor, warn user and allow manual scale/boundary input.

## Configurable Parameters

All thresholds are user-adjustable via the GUI with sensible defaults:

- **Min room area:** 2.0 m² (filters noise polygons)
- **Max room area:** 5000 m² (filters exterior/background)
- **Wall thickness range:** 3–30 px (morphological kernel sizes)
- **Douglas-Peucker tolerance:** 2.0 px (polygon simplification)
- **Binarization block size:** 51 px (adaptive threshold)

## Polygon Validation

- Polygons must be valid (no self-intersections) — use `Shapely.is_valid` with `make_valid()` fix-up
- Overlapping polygons flagged in GUI for user review
- Area and perimeter auto-recalculate on any vertex edit
- Boundary segment lengths update in sync with polygon changes

## OCR for Scale Detection

- Use Tesseract (pytesseract) for text extraction — free, local, no API dependency
- Search for patterns: "1:100", "Scale 1:200", "1px = Xm", dimension annotations with numbers + unit markers
- Scale bar detection: find isolated horizontal lines near numeric text clusters
