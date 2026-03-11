# Floorplan Processor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a desktop app that extracts room boundaries, names, areas, and perimeters from raster floorplan PDFs using OpenCV + Gemini Flash, with a GUI for review and correction.

**Architecture:** Python FastAPI backend runs the CV + AI pipeline and serves results via REST API. React + Fabric.js frontend renders the floorplan with interactive polygon overlays. Electron wraps the frontend as a desktop app. SQLite stores project/room data.

**Tech Stack:** Python 3.13, OpenCV, Shapely, PyMuPDF, pytesseract, google-generativeai, FastAPI, SQLite | React, TypeScript, Fabric.js, TailwindCSS, Electron

**Spec:** `docs/superpowers/specs/2026-03-11-floorplan-processor-design.md`

**Sample files:** `input sample.pdf` (raster floorplan), `output sample.pdf` (previous attempt with poor polygon overlays)

---

## Chunk 1: Project Setup & Core Pipeline

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pyproject.toml`
- Create: `backend/__init__.py`
- Create: `backend/pipeline/__init__.py`
- Create: `backend/models/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create backend directory structure**

```bash
mkdir -p backend/pipeline backend/models backend/tests/pipeline
touch backend/__init__.py backend/pipeline/__init__.py backend/models/__init__.py backend/tests/__init__.py backend/tests/pipeline/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
# backend/requirements.txt
PyMuPDF>=1.24.0
opencv-python>=4.9.0
numpy>=2.0.0
shapely>=2.0.0
pytesseract>=0.3.10
google-generativeai>=0.8.0
fastapi>=0.115.0
uvicorn>=0.30.0
python-multipart>=0.0.9
pydantic>=2.0.0
Pillow>=10.0.0
```

- [ ] **Step 3: Create .gitignore**

```
# .gitignore
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
node_modules/
.superpowers/
venv/
*.db
```

- [ ] **Step 4: Install dependencies**

```bash
cd backend && pip install -r requirements.txt
```

Verify with: `python -c "import fitz, cv2, shapely, google.generativeai; print('All imports OK')"`

- [ ] **Step 5: Install Tesseract OCR binary**

Note: `brew` may not be available. Check for alternative install methods:
```bash
# Try brew first
brew install tesseract 2>/dev/null || echo "brew not available, try conda"
# Fallback: conda
conda install -c conda-forge tesseract 2>/dev/null || echo "Install tesseract manually"
```

Verify: `tesseract --version`

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/__init__.py backend/pipeline/__init__.py backend/models/__init__.py backend/tests/ .gitignore
git commit -m "chore: project setup with backend structure and dependencies"
```

---

### Task 2: PDF Extractor

**Files:**
- Create: `backend/pipeline/extractor.py`
- Create: `backend/tests/pipeline/test_extractor.py`

The extractor pulls the raster image out of a PDF page and returns it as a NumPy array (for OpenCV) plus metadata (page dimensions, any embedded text).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_extractor.py
import os
import numpy as np
import pytest
from backend.pipeline.extractor import extract_floorplan

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "input sample.pdf")

class TestExtractFloorplan:
    def test_extracts_image_from_pdf(self):
        result = extract_floorplan(SAMPLE_PDF)
        assert result is not None
        assert "image" in result
        assert isinstance(result["image"], np.ndarray)
        assert len(result["image"].shape) == 3  # H x W x C
        assert result["image"].shape[2] == 3    # RGB

    def test_extracts_page_dimensions(self):
        result = extract_floorplan(SAMPLE_PDF)
        assert "page_width" in result
        assert "page_height" in result
        assert result["page_width"] > 0
        assert result["page_height"] > 0

    def test_extracts_embedded_text(self):
        result = extract_floorplan(SAMPLE_PDF)
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_returns_page_count(self):
        result = extract_floorplan(SAMPLE_PDF)
        assert "page_count" in result
        assert result["page_count"] >= 1

    def test_specific_page_selection(self):
        result = extract_floorplan(SAMPLE_PDF, page_num=0)
        assert result["image"] is not None

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_floorplan("nonexistent.pdf")

    def test_invalid_page_raises(self):
        with pytest.raises(ValueError):
            extract_floorplan(SAMPLE_PDF, page_num=999)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jemyap/Projects/Meinhardt-Group-Work/floorplan-processor
python -m pytest backend/tests/pipeline/test_extractor.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/extractor.py
"""Extract raster floorplan images from PDF files using PyMuPDF."""

import fitz
import numpy as np
from pathlib import Path


def extract_floorplan(pdf_path: str, page_num: int = 0) -> dict:
    """Extract the floorplan image and metadata from a PDF page.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Zero-based page index to extract from.

    Returns:
        dict with keys:
            image: np.ndarray (H x W x 3, RGB)
            page_width: float (PDF points)
            page_height: float (PDF points)
            text: str (all text on the page)
            page_count: int (total pages in PDF)
            image_width: int (pixels)
            image_height: int (pixels)
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))

    if page_num < 0 or page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (0-{len(doc)-1})")

    page = doc[page_num]

    # Try to extract embedded image first (higher resolution than rendering)
    images = page.get_images(full=True)
    if images:
        xref = images[0][0]
        pix = fitz.Pixmap(doc, xref)
        # Convert CMYK or grayscale to RGB
        if pix.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
    else:
        # Fallback: render the page at high resolution
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for detail
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

    text = page.get_text()
    rect = page.rect

    result = {
        "image": img_array.copy(),
        "page_width": rect.width,
        "page_height": rect.height,
        "text": text,
        "page_count": len(doc),
        "image_width": img_array.shape[1],
        "image_height": img_array.shape[0],
    }

    doc.close()
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/pipeline/test_extractor.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/extractor.py backend/tests/pipeline/test_extractor.py
git commit -m "feat: PDF floorplan image extractor"
```

---

### Task 3: Image Pre-processor

**Files:**
- Create: `backend/pipeline/preprocessor.py`
- Create: `backend/tests/pipeline/test_preprocessor.py`

Pre-processes the extracted image for wall detection: grayscale conversion, adaptive thresholding, morphological closing to fill gaps, and optional deskew.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_preprocessor.py
import numpy as np
import pytest
from backend.pipeline.preprocessor import preprocess_image


class TestPreprocessImage:
    def _make_test_image(self, w=200, h=200):
        """Create a synthetic floorplan-like image with walls."""
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Draw some "walls" as black lines
        img[50, 20:180] = 0      # horizontal wall
        img[150, 20:180] = 0     # horizontal wall
        img[50:150, 20] = 0      # vertical wall
        img[50:150, 180] = 0     # vertical wall
        return img

    def test_returns_binary_image(self):
        img = self._make_test_image()
        result = preprocess_image(img)
        assert "binary" in result
        assert result["binary"].dtype == np.uint8
        assert len(result["binary"].shape) == 2  # grayscale
        unique = np.unique(result["binary"])
        assert all(v in [0, 255] for v in unique)

    def test_returns_grayscale(self):
        img = self._make_test_image()
        result = preprocess_image(img)
        assert "gray" in result
        assert len(result["gray"].shape) == 2

    def test_preserves_dimensions(self):
        img = self._make_test_image(300, 200)
        result = preprocess_image(img)
        assert result["binary"].shape == (200, 300)

    def test_walls_are_white_in_binary(self):
        """In output binary, walls (dark lines) should be white (255) for morphological ops."""
        img = self._make_test_image()
        result = preprocess_image(img)
        # After inversion, wall pixels should be 255
        assert result["binary"][50, 100] == 255  # horizontal wall midpoint

    def test_custom_block_size(self):
        img = self._make_test_image()
        result = preprocess_image(img, block_size=31)
        assert result["binary"] is not None

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            preprocess_image(np.array([]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/pipeline/test_preprocessor.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/preprocessor.py
"""Pre-process floorplan images for wall detection."""

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    block_size: int = 51,
    closing_kernel_size: int = 3,
) -> dict:
    """Convert floorplan image to binary for wall detection.

    Args:
        image: RGB image as np.ndarray (H x W x 3).
        block_size: Block size for adaptive thresholding (must be odd).
        closing_kernel_size: Kernel size for morphological closing.

    Returns:
        dict with keys:
            gray: Grayscale image (H x W)
            binary: Binary image with walls as white (255) on black (0)
    """
    if image.size == 0:
        raise ValueError("Empty image provided")

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Adaptive thresholding handles uneven lighting from scans
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )

    # Morphological closing fills small gaps in wall lines
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_component_area = 50
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            binary[labels == i] = 0

    return {
        "gray": gray,
        "binary": binary,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest backend/tests/pipeline/test_preprocessor.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/preprocessor.py backend/tests/pipeline/test_preprocessor.py
git commit -m "feat: image pre-processor with adaptive thresholding"
```

---

### Task 4: Wall Detector

**Files:**
- Create: `backend/pipeline/wall_detector.py`
- Create: `backend/tests/pipeline/test_wall_detector.py`

Detects wall segments from the binary image using morphological line extraction and Hough transform.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_wall_detector.py
import numpy as np
import pytest
from backend.pipeline.wall_detector import detect_walls


class TestDetectWalls:
    def _make_wall_image(self, w=400, h=400):
        """Create binary image with clear wall segments (walls=255)."""
        img = np.zeros((h, w), dtype=np.uint8)
        # Thick horizontal walls
        img[100:105, 50:350] = 255
        img[300:305, 50:350] = 255
        # Thick vertical walls
        img[100:305, 50:55] = 255
        img[100:305, 345:350] = 255
        return img

    def test_returns_wall_segments(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        assert "segments" in result
        assert len(result["segments"]) > 0

    def test_segments_have_endpoints(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        for seg in result["segments"]:
            assert "x1" in seg and "y1" in seg
            assert "x2" in seg and "y2" in seg

    def test_returns_wall_mask(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        assert "wall_mask" in result
        assert result["wall_mask"].shape == img.shape

    def test_detects_horizontal_walls(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        horizontal = [s for s in result["segments"] if s["orientation"] == "horizontal"]
        assert len(horizontal) >= 2

    def test_detects_vertical_walls(self):
        img = self._make_wall_image()
        result = detect_walls(img)
        vertical = [s for s in result["segments"] if s["orientation"] == "vertical"]
        assert len(vertical) >= 2

    def test_empty_image_returns_no_walls(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        result = detect_walls(img)
        assert len(result["segments"]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/pipeline/test_wall_detector.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/wall_detector.py
"""Detect wall segments from binary floorplan images."""

import cv2
import numpy as np


def detect_walls(
    binary: np.ndarray,
    min_wall_length: int = 30,
    wall_thickness_range: tuple = (3, 30),
) -> dict:
    """Detect wall segments using morphological operations and Hough transform.

    Args:
        binary: Binary image with walls as white (255).
        min_wall_length: Minimum wall length in pixels.
        wall_thickness_range: (min, max) wall thickness in pixels.

    Returns:
        dict with keys:
            segments: list of dicts with x1, y1, x2, y2, orientation, length
            wall_mask: binary mask of detected walls
    """
    h, w = binary.shape
    wall_mask = np.zeros_like(binary)

    # Morphological line detection for horizontal walls
    h_kernel_len = max(min_wall_length, w // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Morphological line detection for vertical walls
    v_kernel_len = max(min_wall_length, h // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    wall_mask = cv2.bitwise_or(h_lines, v_lines)

    # Extract line segments using Hough transform on the wall mask
    segments = []

    lines = cv2.HoughLinesP(
        wall_mask, rho=1, theta=np.pi / 180,
        threshold=min_wall_length,
        minLineLength=min_wall_length,
        maxLineGap=10,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Classify orientation
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                orientation = "horizontal"
            elif 70 < angle < 110:
                orientation = "vertical"
            else:
                orientation = "diagonal"

            segments.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "orientation": orientation,
                "length": float(length),
            })

    # Merge nearby collinear segments
    segments = _merge_segments(segments, merge_threshold=15)

    return {
        "segments": segments,
        "wall_mask": wall_mask,
    }


def _merge_segments(segments: list, merge_threshold: int = 15) -> list:
    """Merge nearby collinear wall segments."""
    if not segments:
        return segments

    merged = []
    used = set()

    for i, s1 in enumerate(segments):
        if i in used:
            continue
        group = [s1]
        used.add(i)

        for j, s2 in enumerate(segments):
            if j in used or j <= i:
                continue
            if s1["orientation"] != s2["orientation"]:
                continue

            # Check if segments are close and collinear
            if s1["orientation"] == "horizontal":
                y_dist = abs((s1["y1"] + s1["y2"]) / 2 - (s2["y1"] + s2["y2"]) / 2)
                if y_dist < merge_threshold:
                    group.append(s2)
                    used.add(j)
            elif s1["orientation"] == "vertical":
                x_dist = abs((s1["x1"] + s1["x2"]) / 2 - (s2["x1"] + s2["x2"]) / 2)
                if x_dist < merge_threshold:
                    group.append(s2)
                    used.add(j)

        # Merge group into single segment
        all_x = [s["x1"] for s in group] + [s["x2"] for s in group]
        all_y = [s["y1"] for s in group] + [s["y2"] for s in group]

        if group[0]["orientation"] == "horizontal":
            merged_seg = {
                "x1": min(all_x), "y1": int(np.mean([s["y1"] for s in group])),
                "x2": max(all_x), "y2": int(np.mean([s["y2"] for s in group])),
                "orientation": "horizontal",
            }
        elif group[0]["orientation"] == "vertical":
            merged_seg = {
                "x1": int(np.mean([s["x1"] for s in group])), "y1": min(all_y),
                "x2": int(np.mean([s["x2"] for s in group])), "y2": max(all_y),
                "orientation": "vertical",
            }
        else:
            merged_seg = group[0]

        merged_seg["length"] = float(np.sqrt(
            (merged_seg["x2"] - merged_seg["x1"]) ** 2 +
            (merged_seg["y2"] - merged_seg["y1"]) ** 2
        ))
        merged.append(merged_seg)

    return merged
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/pipeline/test_wall_detector.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/wall_detector.py backend/tests/pipeline/test_wall_detector.py
git commit -m "feat: wall detector with morphological line extraction"
```

---

### Task 5: Room Segmenter

**Files:**
- Create: `backend/pipeline/room_segmenter.py`
- Create: `backend/tests/pipeline/test_room_segmenter.py`

Takes the wall mask, flood-fills enclosed regions, filters by area, extracts contours, and converts to Shapely polygons.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_room_segmenter.py
import numpy as np
import pytest
from shapely.geometry import Polygon
from backend.pipeline.room_segmenter import segment_rooms


class TestSegmentRooms:
    def _make_rooms_image(self, w=400, h=400):
        """Binary image with two enclosed rooms (walls=255)."""
        img = np.zeros((h, w), dtype=np.uint8)
        # Outer walls
        img[50:55, 50:350] = 255
        img[250:255, 50:350] = 255
        img[50:255, 50:55] = 255
        img[50:255, 345:350] = 255
        # Dividing wall creating 2 rooms
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/pipeline/test_room_segmenter.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/room_segmenter.py
"""Segment rooms from wall mask using flood-fill and contour extraction."""

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def segment_rooms(
    wall_mask: np.ndarray,
    min_area_px: int = 500,
    max_area_ratio: float = 0.5,
    simplify_tolerance: float = 2.0,
    excluded_regions: list | None = None,
) -> list[dict]:
    """Segment enclosed rooms from a wall mask.

    Args:
        wall_mask: Binary image with walls as white (255).
        min_area_px: Minimum room area in pixels to keep.
        max_area_ratio: Max fraction of total image area (filters background).
        simplify_tolerance: Douglas-Peucker simplification tolerance in pixels.
        excluded_regions: List of [x, y, w, h] bounding boxes to mask out
                         (tables, legends, title blocks).

    Returns:
        List of dicts with keys: polygon, area_px, perimeter_px,
        centroid, boundary_lengths_px, contour.
    """
    if wall_mask.size == 0 or wall_mask.max() == 0:
        return []

    h, w = wall_mask.shape
    max_area_px = h * w * max_area_ratio

    # Create inverted mask: rooms are white, walls are black
    room_mask = cv2.bitwise_not(wall_mask)

    # Mask out excluded regions (tables, legends)
    if excluded_regions:
        for region in excluded_regions:
            rx, ry, rw, rh = region
            room_mask[ry:ry+rh, rx:rx+rw] = 0

    # Find contours of enclosed regions
    contours, _ = cv2.findContours(
        room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rooms = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px or area > max_area_px:
            continue

        # Simplify contour
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Convert to Shapely polygon
        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type != "Polygon":
                    continue
        except Exception:
            continue

        centroid = poly.centroid

        # Calculate individual boundary segment lengths
        coords = list(poly.exterior.coords)
        boundary_lengths = []
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            boundary_lengths.append(float(np.sqrt(dx**2 + dy**2)))

        rooms.append({
            "polygon": poly,
            "area_px": float(poly.area),
            "perimeter_px": float(poly.length),
            "centroid": (float(centroid.x), float(centroid.y)),
            "boundary_lengths_px": boundary_lengths,
            "contour": approx,
        })

    return rooms
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/pipeline/test_room_segmenter.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/room_segmenter.py backend/tests/pipeline/test_room_segmenter.py
git commit -m "feat: room segmenter with flood-fill and polygon extraction"
```

---

### Task 6: Scale Detector

**Files:**
- Create: `backend/pipeline/scale_detector.py`
- Create: `backend/tests/pipeline/test_scale_detector.py`

Attempts to auto-detect the scale from embedded text (e.g., "1:100", "1px = 0.0169m") or falls back to manual input.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_scale_detector.py
import pytest
from backend.pipeline.scale_detector import detect_scale, parse_scale_text


class TestParseScaleText:
    def test_parses_px_to_meter(self):
        result = parse_scale_text("Scale: 1px = 0.0169m")
        assert result is not None
        assert abs(result["px_per_meter"] - (1 / 0.0169)) < 0.1

    def test_parses_ratio_scale(self):
        result = parse_scale_text("1:100")
        assert result is not None
        assert result["scale_ratio"] == 100

    def test_parses_ratio_with_prefix(self):
        result = parse_scale_text("Scale 1:200")
        assert result is not None
        assert result["scale_ratio"] == 200

    def test_no_scale_returns_none(self):
        result = parse_scale_text("No scale here")
        assert result is None

    def test_empty_text_returns_none(self):
        result = parse_scale_text("")
        assert result is None


class TestDetectScale:
    def test_detects_from_text(self):
        result = detect_scale(
            text="GPU-Accelerated Analysis | Scale: 1px = 0.0169m | Rooms: 359"
        )
        assert result is not None
        assert "px_per_meter" in result
        assert result["source"] == "auto"

    def test_manual_override(self):
        result = detect_scale(text="no scale", manual_px_per_meter=59.17)
        assert result is not None
        assert abs(result["px_per_meter"] - 59.17) < 0.01
        assert result["source"] == "manual"

    def test_no_scale_no_manual_returns_none(self):
        result = detect_scale(text="no scale text")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/pipeline/test_scale_detector.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/scale_detector.py
"""Detect scale/measurement reference from floorplan text and images."""

import re
import numpy as np


def parse_scale_text(text: str) -> dict | None:
    """Parse scale information from text strings.

    Supports formats:
        - "1px = 0.0169m" → px_per_meter
        - "1:100" → scale_ratio
        - "Scale 1:200" → scale_ratio

    Returns:
        dict with scale info or None if no scale found.
    """
    if not text:
        return None

    # Pattern: "1px = Xm" or "1 px = X m"
    px_match = re.search(r"1\s*px\s*=\s*([\d.]+)\s*m", text, re.IGNORECASE)
    if px_match:
        meters_per_px = float(px_match.group(1))
        if meters_per_px > 0:
            return {
                "px_per_meter": 1.0 / meters_per_px,
                "meters_per_px": meters_per_px,
                "format": "px_to_meter",
            }

    # Pattern: "1:100" or "Scale 1:200"
    ratio_match = re.search(r"(?:scale\s*)?1\s*:\s*(\d+)", text, re.IGNORECASE)
    if ratio_match:
        ratio = int(ratio_match.group(1))
        if ratio > 0:
            return {
                "scale_ratio": ratio,
                "format": "ratio",
            }

    return None


def detect_scale_from_image(image: np.ndarray) -> dict | None:
    """Try to detect scale using OCR on the image (requires Tesseract).

    Scans the image for text containing scale patterns.
    """
    try:
        import pytesseract
        from PIL import Image

        pil_image = Image.fromarray(image)
        ocr_text = pytesseract.image_to_string(pil_image)
        return parse_scale_text(ocr_text)
    except ImportError:
        return None
    except Exception:
        return None


def detect_scale(
    text: str = "",
    image: np.ndarray | None = None,
    manual_px_per_meter: float | None = None,
) -> dict | None:
    """Detect scale from available sources.

    Priority: manual override > embedded PDF text > OCR on image.

    Args:
        text: Embedded text from the PDF page.
        image: The floorplan image (for OCR fallback).
        manual_px_per_meter: User-provided scale override.

    Returns:
        dict with px_per_meter and source, or None.
    """
    # Manual override takes priority
    if manual_px_per_meter is not None:
        return {
            "px_per_meter": manual_px_per_meter,
            "source": "manual",
        }

    # Try parsing from embedded PDF text
    parsed = parse_scale_text(text)
    if parsed and "px_per_meter" in parsed:
        return {
            "px_per_meter": parsed["px_per_meter"],
            "source": "auto",
        }

    # Ratio-based scale needs DPI info to convert, return as-is
    if parsed and "scale_ratio" in parsed:
        return {
            "scale_ratio": parsed["scale_ratio"],
            "source": "auto",
        }

    # Try OCR on the image as fallback
    if image is not None:
        ocr_parsed = detect_scale_from_image(image)
        if ocr_parsed and "px_per_meter" in ocr_parsed:
            return {
                "px_per_meter": ocr_parsed["px_per_meter"],
                "source": "auto_ocr",
            }

    return None
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/pipeline/test_scale_detector.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/scale_detector.py backend/tests/pipeline/test_scale_detector.py
git commit -m "feat: scale detector with text parsing and manual fallback"
```

---

### Task 7: Gemini Flash Vision AI Integration

**Files:**
- Create: `backend/pipeline/vision_ai.py`
- Create: `backend/tests/pipeline/test_vision_ai.py`

Two-pass Gemini integration: (1) classify regions (floorplan vs tables), (2) label rooms with names and types.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/pipeline/test_vision_ai.py
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from backend.pipeline.vision_ai import (
    classify_regions,
    label_rooms,
    _build_classification_prompt,
    _build_labeling_prompt,
    _parse_json_response,
)


class TestBuildPrompts:
    def test_classification_prompt_is_string(self):
        prompt = _build_classification_prompt()
        assert isinstance(prompt, str)
        assert "floor" in prompt.lower()
        assert "table" in prompt.lower()

    def test_labeling_prompt_includes_room_count(self):
        prompt = _build_labeling_prompt(room_count=5)
        assert "5" in prompt


class TestParseJsonResponse:
    def test_parses_valid_json(self):
        text = '```json\n{"regions": [{"type": "floorplan"}]}\n```'
        result = _parse_json_response(text)
        assert result is not None
        assert "regions" in result

    def test_parses_raw_json(self):
        text = '{"regions": []}'
        result = _parse_json_response(text)
        assert result is not None

    def test_invalid_json_returns_none(self):
        result = _parse_json_response("not json at all")
        assert result is None


class TestClassifyRegions:
    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_returns_regions(self, mock_call):
        mock_call.return_value = json.dumps({
            "floorplan_regions": [{"x": 0, "y": 0, "width": 100, "height": 100}],
            "excluded_regions": [{"x": 200, "y": 0, "width": 50, "height": 50, "type": "table"}],
        })
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = classify_regions(img)
        assert "floorplan_regions" in result
        assert "excluded_regions" in result

    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_api_failure_returns_fallback(self, mock_call):
        mock_call.side_effect = Exception("API error")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = classify_regions(img)
        assert "floorplan_regions" in result
        assert len(result["floorplan_regions"]) == 1  # full image fallback


class TestLabelRooms:
    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_returns_labels(self, mock_call):
        mock_call.return_value = json.dumps({
            "rooms": [
                {"room_id": 0, "name": "Office 201", "type": "office", "confidence": 0.9},
            ]
        })
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        rooms = [{"centroid": (50, 50), "polygon": None}]
        result = label_rooms(img, rooms)
        assert len(result) == 1
        assert result[0]["name"] == "Office 201"

    @patch("backend.pipeline.vision_ai._call_gemini")
    def test_api_failure_returns_unnamed(self, mock_call):
        mock_call.side_effect = Exception("API error")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        rooms = [{"centroid": (50, 50), "polygon": None}]
        result = label_rooms(img, rooms)
        assert result[0]["name"] == "Unnamed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/pipeline/test_vision_ai.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/pipeline/vision_ai.py
"""Gemini Flash integration for region classification and room labeling."""

import json
import re
import os
import base64
import numpy as np
from PIL import Image
import io


def _call_gemini(image: np.ndarray, prompt: str) -> str:
    """Call Gemini Flash API with an image and prompt.

    Requires GOOGLE_API_KEY environment variable.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)

    response = model.generate_content([prompt, pil_image])
    return response.text


def _build_classification_prompt() -> str:
    return """Analyze this architectural floorplan image. Identify and return bounding boxes for:

1. **floorplan_regions**: Areas containing the actual building floor layout (rooms, corridors, walls).
2. **excluded_regions**: Areas containing tables, legends, schedules, title blocks, or any non-floorplan content.

Return JSON in this exact format:
```json
{
  "floorplan_regions": [{"x": 0, "y": 0, "width": 100, "height": 100}],
  "excluded_regions": [{"x": 200, "y": 0, "width": 50, "height": 50, "type": "table"}]
}
```

Coordinates are in pixels from the top-left corner. The "type" field for excluded regions should be one of: "table", "legend", "title_block", "schedule".

Return ONLY the JSON, no other text."""


def _build_labeling_prompt(room_count: int) -> str:
    return f"""This floorplan image has {room_count} rooms outlined with colored polygons and numbered labels.

For each numbered room, identify:
1. The **room name** (read from text labels inside or near the room boundary)
2. The **room type** (office, bathroom, corridor, meeting_room, kitchen, storage, lobby, elevator, stairwell, utility, or other)
3. Your **confidence** (0.0 to 1.0) in the identification

Return JSON in this exact format:
```json
{{
  "rooms": [
    {{"room_id": 0, "name": "Office 201", "type": "office", "confidence": 0.9}},
    {{"room_id": 1, "name": "Corridor", "type": "corridor", "confidence": 0.8}}
  ]
}}
```

Return ONLY the JSON, no other text."""


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from Gemini response text."""
    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


def classify_regions(image: np.ndarray) -> dict:
    """Classify regions in the floorplan as floor layout vs tables/legends.

    Returns:
        dict with floorplan_regions and excluded_regions lists.
    """
    h, w = image.shape[:2]

    try:
        prompt = _build_classification_prompt()
        response_text = _call_gemini(image, prompt)
        result = _parse_json_response(response_text)

        if result and "floorplan_regions" in result:
            return result

    except Exception:
        pass

    # Fallback: assume entire image is floorplan
    return {
        "floorplan_regions": [{"x": 0, "y": 0, "width": w, "height": h}],
        "excluded_regions": [],
    }


def label_rooms(image: np.ndarray, rooms: list) -> list[dict]:
    """Label rooms with names and types using Gemini Flash.

    Args:
        image: The floorplan image (will be annotated with room numbers).
        rooms: List of room dicts from the segmenter.

    Returns:
        List of dicts with room_id, name, type, confidence.
    """
    try:
        # Draw room numbers on image for Gemini
        import cv2
        annotated = image.copy()
        for i, room in enumerate(rooms):
            cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
            cv2.putText(annotated, str(i), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        prompt = _build_labeling_prompt(room_count=len(rooms))
        response_text = _call_gemini(annotated, prompt)
        result = _parse_json_response(response_text)

        if result and "rooms" in result:
            return result["rooms"]

    except Exception:
        pass

    # Fallback: return unnamed rooms
    return [
        {"room_id": i, "name": "Unnamed", "type": "unknown", "confidence": 0.0}
        for i in range(len(rooms))
    ]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/pipeline/test_vision_ai.py -v
```

Expected: All 9 tests PASS (API calls are mocked)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/vision_ai.py backend/tests/pipeline/test_vision_ai.py
git commit -m "feat: Gemini Flash integration for region classification and room labeling"
```

---

### Task 8: Room Data Model

**Files:**
- Create: `backend/models/room.py`
- Create: `backend/tests/test_models.py`

Pydantic models for rooms, projects, and excluded regions. Handles conversion between pixel coordinates and real-world measurements.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_models.py
import pytest
from shapely.geometry import Polygon
from backend.models.room import RoomData, ProjectData, to_real_measurements


class TestRoomData:
    def test_create_room(self):
        room = RoomData(
            id="room-1",
            name="Office 201",
            room_type="office",
            boundary_polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
            area_px=5000.0,
            perimeter_px=300.0,
            centroid=(50.0, 25.0),
            boundary_lengths_px=[100.0, 50.0, 100.0, 50.0],
            source="cv",
            confidence=0.9,
        )
        assert room.name == "Office 201"
        assert room.area_px == 5000.0

    def test_room_to_dict(self):
        room = RoomData(
            id="room-1",
            name="Test",
            room_type="office",
            boundary_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            area_px=100.0,
            perimeter_px=40.0,
            centroid=(5.0, 5.0),
            boundary_lengths_px=[10.0, 10.0, 10.0, 10.0],
        )
        d = room.model_dump()
        assert d["name"] == "Test"
        assert d["boundary_polygon"] == [[0, 0], [10, 0], [10, 10], [0, 10]]


class TestRealMeasurements:
    def test_convert_area(self):
        result = to_real_measurements(
            area_px=10000.0,
            perimeter_px=400.0,
            boundary_lengths_px=[100.0, 100.0, 100.0, 100.0],
            px_per_meter=100.0,
        )
        assert abs(result["area_sqm"] - 1.0) < 0.01  # 10000/(100^2)
        assert abs(result["perimeter_m"] - 4.0) < 0.01
        assert len(result["boundary_lengths_m"]) == 4
        assert abs(result["boundary_lengths_m"][0] - 1.0) < 0.01


class TestProjectData:
    def test_create_project(self):
        proj = ProjectData(
            id="proj-1",
            name="Test Building",
            pdf_path="/path/to/file.pdf",
            scale_px_per_meter=59.17,
            scale_source="auto",
        )
        assert proj.name == "Test Building"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_models.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/models/room.py
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
    bbox: list[float] = []  # [x, y, width, height]


def to_real_measurements(
    area_px: float,
    perimeter_px: float,
    boundary_lengths_px: list[float],
    px_per_meter: float,
) -> dict:
    """Convert pixel measurements to real-world units.

    Args:
        area_px: Area in square pixels.
        perimeter_px: Perimeter in pixels.
        boundary_lengths_px: Individual segment lengths in pixels.
        px_per_meter: Scale factor (pixels per meter).

    Returns:
        dict with area_sqm, perimeter_m, boundary_lengths_m.
    """
    scale_sq = px_per_meter ** 2
    return {
        "area_sqm": area_px / scale_sq,
        "perimeter_m": perimeter_px / px_per_meter,
        "boundary_lengths_m": [l / px_per_meter for l in boundary_lengths_px],
    }
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/test_models.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/models/room.py backend/tests/test_models.py
git commit -m "feat: room and project data models with unit conversion"
```

---

### Task 9: Database Layer

**Files:**
- Create: `backend/database.py`
- Create: `backend/tests/test_database.py`

SQLite operations for storing and retrieving projects, rooms, and excluded regions.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_database.py
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
            project_id=proj.id,
            name="Office",
            room_type="office",
            boundary_polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            area_px=100.0,
            perimeter_px=40.0,
            centroid=(5.0, 5.0),
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_database.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/database.py
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
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                pdf_path TEXT,
                scale_px_per_meter REAL,
                scale_source TEXT
            );
            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY,
                project_id TEXT REFERENCES projects(id),
                name TEXT,
                room_type TEXT,
                boundary_polygon TEXT,
                area_px REAL,
                perimeter_px REAL,
                area_sqm REAL,
                perimeter_m REAL,
                boundary_lengths_px TEXT,
                boundary_lengths_m TEXT,
                centroid_x REAL,
                centroid_y REAL,
                source TEXT,
                confidence REAL
            );
            CREATE TABLE IF NOT EXISTS excluded_regions (
                id TEXT PRIMARY KEY,
                project_id TEXT REFERENCES projects(id),
                region_type TEXT,
                bbox TEXT
            );
        """)

    def save_project(self, project: ProjectData):
        self.conn.execute(
            "INSERT OR REPLACE INTO projects VALUES (?, ?, ?, ?, ?, ?)",
            (project.id, project.name, project.created_at.isoformat(),
             project.pdf_path, project.scale_px_per_meter, project.scale_source),
        )
        self.conn.commit()

    def get_project(self, project_id: str) -> ProjectData | None:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if not row:
            return None
        return ProjectData(
            id=row["id"], name=row["name"], pdf_path=row["pdf_path"],
            scale_px_per_meter=row["scale_px_per_meter"],
            scale_source=row["scale_source"] or "manual",
        )

    def list_projects(self) -> list[ProjectData]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [
            ProjectData(
                id=r["id"], name=r["name"], pdf_path=r["pdf_path"],
                scale_px_per_meter=r["scale_px_per_meter"],
                scale_source=r["scale_source"] or "manual",
            )
            for r in rows
        ]

    def save_room(self, room: RoomData):
        self.conn.execute(
            "INSERT OR REPLACE INTO rooms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (room.id, room.project_id, room.name, room.room_type,
             json.dumps(room.boundary_polygon), room.area_px, room.perimeter_px,
             room.area_sqm, room.perimeter_m,
             json.dumps(room.boundary_lengths_px),
             json.dumps(room.boundary_lengths_m) if room.boundary_lengths_m else None,
             room.centroid[0], room.centroid[1], room.source, room.confidence),
        )
        self.conn.commit()

    def get_rooms(self, project_id: str) -> list[RoomData]:
        rows = self.conn.execute(
            "SELECT * FROM rooms WHERE project_id = ?", (project_id,)
        ).fetchall()
        return [
            RoomData(
                id=r["id"], project_id=r["project_id"], name=r["name"],
                room_type=r["room_type"],
                boundary_polygon=json.loads(r["boundary_polygon"]) if r["boundary_polygon"] else [],
                area_px=r["area_px"] or 0, perimeter_px=r["perimeter_px"] or 0,
                area_sqm=r["area_sqm"], perimeter_m=r["perimeter_m"],
                boundary_lengths_px=json.loads(r["boundary_lengths_px"]) if r["boundary_lengths_px"] else [],
                boundary_lengths_m=json.loads(r["boundary_lengths_m"]) if r["boundary_lengths_m"] else None,
                centroid=(r["centroid_x"] or 0, r["centroid_y"] or 0),
                source=r["source"] or "cv", confidence=r["confidence"] or 0,
            )
            for r in rows
        ]

    def update_room(self, room: RoomData):
        self.save_room(room)

    def delete_room(self, room_id: str):
        self.conn.execute("DELETE FROM rooms WHERE id = ?", (room_id,))
        self.conn.commit()

    def save_excluded_region(self, region: ExcludedRegion):
        self.conn.execute(
            "INSERT OR REPLACE INTO excluded_regions VALUES (?, ?, ?, ?)",
            (region.id, region.project_id, region.region_type, json.dumps(region.bbox)),
        )
        self.conn.commit()

    def get_excluded_regions(self, project_id: str) -> list[ExcludedRegion]:
        rows = self.conn.execute(
            "SELECT * FROM excluded_regions WHERE project_id = ?", (project_id,)
        ).fetchall()
        return [
            ExcludedRegion(
                id=r["id"], project_id=r["project_id"],
                region_type=r["region_type"],
                bbox=json.loads(r["bbox"]) if r["bbox"] else [],
            )
            for r in rows
        ]

    def close(self):
        self.conn.close()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/test_database.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/database.py backend/tests/test_database.py
git commit -m "feat: SQLite database layer for projects and rooms"
```

---

### Task 10: FastAPI Backend

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_api.py`

REST API that ties the pipeline together. Endpoints for uploading PDFs, running the pipeline, CRUD on rooms, and exporting data.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Set test database path before importing app
os.environ["DB_PATH"] = ":memory:"
from backend.main import app

client = TestClient(app)

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "input sample.pdf")


class TestHealthCheck:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProjectEndpoints:
    def test_list_projects_empty(self):
        response = client.get("/api/projects")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestProcessEndpoint:
    def test_upload_pdf(self):
        if not os.path.exists(SAMPLE_PDF):
            pytest.skip("Sample PDF not available")
        with open(SAMPLE_PDF, "rb") as f:
            response = client.post(
                "/api/process",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert "rooms" in data


class TestRoomEndpoints:
    @patch("backend.main.db")
    def test_update_room(self, mock_db):
        mock_db.get_rooms.return_value = []
        response = client.put(
            "/api/rooms/room-1",
            json={"name": "Updated Room", "room_type": "office"},
        )
        assert response.status_code == 200


class TestExportEndpoint:
    def test_export_csv_no_project(self):
        response = client.get("/api/export/nonexistent?format=csv")
        assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest backend/tests/test_api.py -v
```

- [ ] **Step 3: Write implementation**

```python
# backend/main.py
"""FastAPI backend for the floorplan processor."""

import os
import json
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2
import io

from backend.database import Database
from backend.models.room import RoomData, ProjectData, ExcludedRegion, to_real_measurements
from backend.pipeline.extractor import extract_floorplan
from backend.pipeline.preprocessor import preprocess_image
from backend.pipeline.wall_detector import detect_walls
from backend.pipeline.room_segmenter import segment_rooms
from backend.pipeline.scale_detector import detect_scale
from backend.pipeline.vision_ai import classify_regions, label_rooms

app = FastAPI(title="Floorplan Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "floorplan.db")
db = Database(DB_PATH)

# Store extracted images in memory for the GUI to display
_image_cache: dict[str, np.ndarray] = {}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/projects")
def list_projects():
    projects = db.list_projects()
    return [p.model_dump() for p in projects]


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump()


class ProcessingConfig(BaseModel):
    manual_scale: float | None = None
    min_room_area_px: int = 500
    wall_thickness_min: int = 3
    wall_thickness_max: int = 30
    simplify_tolerance: float = 2.0


@app.post("/api/process")
async def process_pdf(
    file: UploadFile = File(...),
    page_num: int = 0,
):
    """Upload and process a floorplan PDF."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: Extract image from PDF
        extraction = extract_floorplan(tmp_path, page_num=page_num)
        image = extraction["image"]

        # Create project
        project = ProjectData(
            name=file.filename or "Untitled",
            pdf_path=tmp_path,
        )

        # Step 2: Detect scale (try embedded text first, then OCR)
        scale_info = detect_scale(text=extraction["text"], image=image)
        if scale_info and "px_per_meter" in scale_info:
            project.scale_px_per_meter = scale_info["px_per_meter"]
            project.scale_source = scale_info["source"]

        db.save_project(project)
        _image_cache[project.id] = image

        # Step 3: Vision AI Pass 1 — classify regions
        regions = classify_regions(image)
        excluded_bboxes = []
        for region in regions.get("excluded_regions", []):
            er = ExcludedRegion(
                project_id=project.id,
                region_type=region.get("type", "table"),
                bbox=[region["x"], region["y"], region["width"], region["height"]],
            )
            db.save_excluded_region(er)
            excluded_bboxes.append([region["x"], region["y"], region["width"], region["height"]])

        # Step 4: Pre-process image
        processed = preprocess_image(image)

        # Step 5: Mask out excluded regions from binary before wall detection
        binary = processed["binary"].copy()
        for bbox in excluded_bboxes:
            rx, ry, rw, rh = [int(v) for v in bbox]
            binary[ry:ry+rh, rx:rx+rw] = 0

        # Step 6: Detect walls (on masked binary)
        wall_result = detect_walls(binary)

        # Step 7: Segment rooms
        raw_rooms = segment_rooms(
            wall_result["wall_mask"],
            excluded_regions=excluded_bboxes,
        )

        # Step 8: Vision AI Pass 2 — label rooms
        labels = label_rooms(image, raw_rooms)

        # Step 9: Build room data with measurements
        rooms = []
        for i, raw in enumerate(raw_rooms):
            label = labels[i] if i < len(labels) else {"name": "Unnamed", "type": "unknown", "confidence": 0.0}

            polygon_coords = [list(coord) for coord in raw["polygon"].exterior.coords[:-1]]

            room = RoomData(
                project_id=project.id,
                name=label.get("name", "Unnamed"),
                room_type=label.get("type", "unknown"),
                boundary_polygon=polygon_coords,
                area_px=raw["area_px"],
                perimeter_px=raw["perimeter_px"],
                centroid=raw["centroid"],
                boundary_lengths_px=raw["boundary_lengths_px"],
                source="cv",
                confidence=label.get("confidence", 0.0),
            )

            # Convert to real measurements if scale is available
            if project.scale_px_per_meter:
                measurements = to_real_measurements(
                    raw["area_px"], raw["perimeter_px"],
                    raw["boundary_lengths_px"], project.scale_px_per_meter,
                )
                room.area_sqm = measurements["area_sqm"]
                room.perimeter_m = measurements["perimeter_m"]
                room.boundary_lengths_m = measurements["boundary_lengths_m"]

            db.save_room(room)
            rooms.append(room)

        return {
            "project_id": project.id,
            "rooms": [r.model_dump() for r in rooms],
            "excluded_regions": [{"bbox": b} for b in excluded_bboxes],
            "scale": {
                "px_per_meter": project.scale_px_per_meter,
                "source": project.scale_source,
            },
            "image_size": {
                "width": extraction["image_width"],
                "height": extraction["image_height"],
            },
            "page_count": extraction["page_count"],
        }

    finally:
        os.unlink(tmp_path)


@app.get("/api/projects/{project_id}/rooms")
def get_rooms(project_id: str):
    rooms = db.get_rooms(project_id)
    return [r.model_dump() for r in rooms]


class RoomUpdate(BaseModel):
    name: str | None = None
    room_type: str | None = None
    boundary_polygon: list[list[float]] | None = None


@app.put("/api/rooms/{room_id}")
def update_room_endpoint(room_id: str, update: RoomUpdate):
    """Update a room's name, type, or boundary."""
    # Find the room across all projects
    row = db.conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Room not found")

    from backend.models.room import RoomData
    import json
    room = RoomData(
        id=row["id"], project_id=row["project_id"], name=row["name"],
        room_type=row["room_type"],
        boundary_polygon=json.loads(row["boundary_polygon"]) if row["boundary_polygon"] else [],
        area_px=row["area_px"] or 0, perimeter_px=row["perimeter_px"] or 0,
        area_sqm=row["area_sqm"], perimeter_m=row["perimeter_m"],
        boundary_lengths_px=json.loads(row["boundary_lengths_px"]) if row["boundary_lengths_px"] else [],
        centroid=(row["centroid_x"] or 0, row["centroid_y"] or 0),
        source=row["source"] or "cv", confidence=row["confidence"] or 0,
    )

    if update.name is not None:
        room.name = update.name
    if update.room_type is not None:
        room.room_type = update.room_type
    if update.boundary_polygon is not None:
        room.boundary_polygon = update.boundary_polygon
        # Recalculate geometry from new polygon
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(update.boundary_polygon)
        room.area_px = float(poly.area)
        room.perimeter_px = float(poly.length)
        centroid = poly.centroid
        room.centroid = (float(centroid.x), float(centroid.y))
        coords = list(poly.exterior.coords)
        room.boundary_lengths_px = [
            float(np.sqrt((coords[i+1][0]-coords[i][0])**2 + (coords[i+1][1]-coords[i][1])**2))
            for i in range(len(coords)-1)
        ]
        # Recalculate real measurements if scale available
        project = db.get_project(room.project_id)
        if project and project.scale_px_per_meter:
            m = to_real_measurements(room.area_px, room.perimeter_px, room.boundary_lengths_px, project.scale_px_per_meter)
            room.area_sqm = m["area_sqm"]
            room.perimeter_m = m["perimeter_m"]
            room.boundary_lengths_m = m["boundary_lengths_m"]
        room.source = "corrected"

    db.update_room(room)
    return room.model_dump()


@app.delete("/api/rooms/{room_id}")
def delete_room(room_id: str):
    db.delete_room(room_id)
    return {"status": "deleted"}


@app.put("/api/projects/{project_id}/scale")
def update_scale(project_id: str, px_per_meter: float):
    """Manually set the scale and recalculate all room measurements."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.scale_px_per_meter = px_per_meter
    project.scale_source = "manual"
    db.save_project(project)

    # Recalculate all room measurements
    rooms = db.get_rooms(project_id)
    for room in rooms:
        measurements = to_real_measurements(
            room.area_px, room.perimeter_px,
            room.boundary_lengths_px, px_per_meter,
        )
        room.area_sqm = measurements["area_sqm"]
        room.perimeter_m = measurements["perimeter_m"]
        room.boundary_lengths_m = measurements["boundary_lengths_m"]
        db.update_room(room)

    return {"status": "updated", "rooms_recalculated": len(rooms)}


@app.get("/api/projects/{project_id}/image")
def get_image(project_id: str):
    """Serve the extracted floorplan image."""
    if project_id not in _image_cache:
        raise HTTPException(status_code=404, detail="Image not in cache")

    img = _image_cache[project_id]
    _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.get("/api/export/{project_id}")
def export_data(project_id: str, format: str = "json"):
    """Export project data as JSON or CSV."""
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = db.get_rooms(project_id)

    if format == "json":
        return {
            "project": project.model_dump(),
            "rooms": [r.model_dump() for r in rooms],
        }
    elif format == "csv":
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "name", "type", "area_sqm", "perimeter_m", "source", "confidence"])
        for r in rooms:
            writer.writerow([r.id, r.name, r.room_type, r.area_sqm, r.perimeter_m, r.source, r.confidence])
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={project_id}.csv"},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest backend/tests/test_api.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: FastAPI backend with processing pipeline and REST API"
```

---

## Chunk 2: Frontend Application

### Task 11: Frontend Project Setup

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`

- [ ] **Step 1: Scaffold React + Vite project**

```bash
cd /Users/jemyap/Projects/Meinhardt-Group-Work/floorplan-processor
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install fabric@6 tailwindcss @tailwindcss/vite axios
```

- [ ] **Step 2: Configure Tailwind**

Replace `frontend/src/index.css` with:

```css
@import "tailwindcss";
```

Add Tailwind plugin to `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 3: Create API client**

```typescript
// frontend/src/api.ts
import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface Room {
  id: string;
  name: string;
  room_type: string;
  boundary_polygon: number[][];
  area_sqm: number | null;
  perimeter_m: number | null;
  boundary_lengths_m: number[] | null;
  centroid: [number, number];
  source: string;
  confidence: number;
}

export interface ProcessResult {
  project_id: string;
  rooms: Room[];
  excluded_regions: { bbox: number[] }[];
  scale: { px_per_meter: number | null; source: string };
  image_size: { width: number; height: number };
}

export async function processFloorplan(file: File, pageNum = 0): Promise<ProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post(`/process?page_num=${pageNum}`, formData);
  return data;
}

export async function getProjects() {
  const { data } = await api.get('/projects');
  return data;
}

export async function getRooms(projectId: string): Promise<Room[]> {
  const { data } = await api.get(`/projects/${projectId}/rooms`);
  return data;
}

export async function updateRoom(roomId: string, update: Partial<Room>) {
  const { data } = await api.put(`/rooms/${roomId}`, update);
  return data;
}

export async function deleteRoom(roomId: string) {
  const { data } = await api.delete(`/rooms/${roomId}`);
  return data;
}

export async function updateScale(projectId: string, pxPerMeter: number) {
  const { data } = await api.put(`/projects/${projectId}/scale?px_per_meter=${pxPerMeter}`);
  return data;
}

export async function exportData(projectId: string, format: 'json' | 'csv') {
  const { data } = await api.get(`/export/${projectId}?format=${format}`);
  return data;
}

export function getImageUrl(projectId: string) {
  return `/api/projects/${projectId}/image`;
}
```

- [ ] **Step 4: Verify dev server starts**

```bash
cd frontend && npm run dev
```

Expected: Vite dev server starts on http://localhost:5173

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: frontend project setup with React, Vite, Tailwind, Fabric.js"
```

---

### Task 12: Floorplan Canvas Component

**Files:**
- Create: `frontend/src/components/FloorplanCanvas.tsx`
- Create: `frontend/src/hooks/useFloorplanCanvas.ts`

The core component: renders the floorplan image on a Fabric.js canvas with room polygons overlaid. Supports pan, zoom, and polygon selection.

- [ ] **Step 1: Create the canvas hook**

```typescript
// frontend/src/hooks/useFloorplanCanvas.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { Canvas, Image as FabricImage, Polygon, FabricText, Point } from 'fabric';
import type { Room } from '../api';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

interface UseFloorplanCanvasOptions {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
}

export function useFloorplanCanvas({
  rooms, imageUrl, selectedRoomId, onRoomSelect, onRoomUpdate,
}: UseFloorplanCanvasOptions) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<Canvas | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = new Canvas(canvasRef.current, {
      selection: false,
      backgroundColor: '#111',
    });

    // Enable pan and zoom
    let isPanning = false;
    let lastPosX = 0;
    let lastPosY = 0;

    canvas.on('mouse:down', (e) => {
      if (e.e.altKey || e.e.button === 1) {
        isPanning = true;
        lastPosX = e.e.clientX;
        lastPosY = e.e.clientY;
        canvas.setCursor('grabbing');
      }
    });

    canvas.on('mouse:move', (e) => {
      if (!isPanning) return;
      const vpt = canvas.viewportTransform!;
      vpt[4] += e.e.clientX - lastPosX;
      vpt[5] += e.e.clientY - lastPosY;
      lastPosX = e.e.clientX;
      lastPosY = e.e.clientY;
      canvas.requestRenderAll();
    });

    canvas.on('mouse:up', () => {
      isPanning = false;
      canvas.setCursor('default');
    });

    canvas.on('mouse:wheel', (opt) => {
      const delta = opt.e.deltaY;
      let zoom = canvas.getZoom();
      zoom *= 0.999 ** delta;
      zoom = Math.min(Math.max(0.1, zoom), 20);
      canvas.zoomToPoint(new Point(opt.e.offsetX, opt.e.offsetY), zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });

    fabricRef.current = canvas;
    setIsReady(true);

    return () => {
      canvas.dispose();
      fabricRef.current = null;
    };
  }, []);

  // Load background image
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !imageUrl) return;

    FabricImage.fromURL(imageUrl).then((img) => {
      canvas.setDimensions({
        width: canvas.getElement().parentElement?.clientWidth || 800,
        height: canvas.getElement().parentElement?.clientHeight || 600,
      });
      img.set({ selectable: false, evented: false });
      canvas.backgroundImage = img;

      // Fit image to canvas
      const scaleX = canvas.width! / (img.width || 1);
      const scaleY = canvas.height! / (img.height || 1);
      const scale = Math.min(scaleX, scaleY);
      canvas.setZoom(scale);

      canvas.requestRenderAll();
    });
  }, [imageUrl, isReady]);

  // Draw room polygons
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    // Remove existing room polygons
    const objects = canvas.getObjects().filter((o) => o.data?.type === 'room');
    objects.forEach((o) => canvas.remove(o));

    rooms.forEach((room, i) => {
      if (!room.boundary_polygon || room.boundary_polygon.length < 3) return;

      const points = room.boundary_polygon.map((p) => ({ x: p[0], y: p[1] }));
      const color = ROOM_COLORS[i % ROOM_COLORS.length];
      const isSelected = room.id === selectedRoomId;

      const polygon = new Polygon(points, {
        fill: isSelected ? color + '40' : color + '20',
        stroke: color,
        strokeWidth: isSelected ? 3 : 1.5,
        selectable: false,
        data: { type: 'room', roomId: room.id },
      });

      polygon.on('mousedown', () => onRoomSelect(room.id));

      canvas.add(polygon);

      // Add room label at centroid
      const label = new FabricText(room.name || 'Unnamed', {
        left: room.centroid[0],
        top: room.centroid[1],
        fontSize: 12,
        fill: color,
        fontFamily: 'monospace',
        selectable: false,
        evented: false,
        data: { type: 'room' },
      });
      canvas.add(label);
    });

    canvas.requestRenderAll();
  }, [rooms, selectedRoomId, isReady]);

  const fitToView = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas || !canvas.backgroundImage) return;
    const img = canvas.backgroundImage as FabricImage;
    const scaleX = canvas.width! / (img.width || 1);
    const scaleY = canvas.height! / (img.height || 1);
    canvas.setZoom(Math.min(scaleX, scaleY));
    canvas.viewportTransform = [canvas.getZoom(), 0, 0, canvas.getZoom(), 0, 0];
    canvas.requestRenderAll();
  }, []);

  return { canvasRef, fitToView };
}
```

- [ ] **Step 2: Create the canvas component**

```tsx
// frontend/src/components/FloorplanCanvas.tsx
import { useFloorplanCanvas } from '../hooks/useFloorplanCanvas';
import type { Room } from '../api';

interface FloorplanCanvasProps {
  rooms: Room[];
  imageUrl: string | null;
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string | null) => void;
  onRoomUpdate: (roomId: string, polygon: number[][]) => void;
}

export function FloorplanCanvas(props: FloorplanCanvasProps) {
  const { canvasRef, fitToView } = useFloorplanCanvas(props);

  return (
    <div className="relative flex-1 bg-neutral-900 overflow-hidden">
      {/* Toolbar */}
      <div className="absolute top-3 left-3 z-10 flex gap-1">
        <button
          onClick={fitToView}
          className="px-3 py-1.5 bg-neutral-800 text-neutral-300 text-xs rounded hover:bg-neutral-700"
        >
          Fit View
        </button>
      </div>

      {/* Canvas */}
      <canvas ref={canvasRef} />

      {/* Hint */}
      <div className="absolute bottom-3 left-3 text-neutral-600 text-xs">
        Scroll to zoom · Alt+Drag to pan · Click room to select
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/FloorplanCanvas.tsx frontend/src/hooks/useFloorplanCanvas.ts
git commit -m "feat: floorplan canvas component with pan/zoom and polygon rendering"
```

---

### Task 13: Room Sidebar and Detail Panel

**Files:**
- Create: `frontend/src/components/RoomSidebar.tsx`
- Create: `frontend/src/components/RoomDetail.tsx`

- [ ] **Step 1: Create room sidebar**

```tsx
// frontend/src/components/RoomSidebar.tsx
import type { Room } from '../api';

const ROOM_COLORS = [
  '#4ade80', '#60a5fa', '#f97316', '#a78bfa', '#f472b6',
  '#facc15', '#2dd4bf', '#fb923c', '#818cf8', '#e879f9',
];

interface RoomSidebarProps {
  rooms: Room[];
  selectedRoomId: string | null;
  onRoomSelect: (roomId: string) => void;
  scale: { px_per_meter: number | null; source: string } | null;
}

export function RoomSidebar({ rooms, selectedRoomId, onRoomSelect, scale }: RoomSidebarProps) {
  const totalArea = rooms.reduce((sum, r) => sum + (r.area_sqm || 0), 0);

  return (
    <div className="w-64 bg-neutral-950 border-r border-neutral-800 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-neutral-800">
        <h2 className="text-xs font-bold text-sky-400 uppercase tracking-wider">
          Rooms ({rooms.length})
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {rooms.map((room, i) => (
          <button
            key={room.id}
            onClick={() => onRoomSelect(room.id)}
            className={`w-full text-left p-2 rounded-md transition-colors ${
              room.id === selectedRoomId
                ? 'bg-neutral-800'
                : 'hover:bg-neutral-900'
            }`}
            style={{ borderLeft: `3px solid ${ROOM_COLORS[i % ROOM_COLORS.length]}` }}
          >
            <div className="text-sm text-neutral-200">{room.name || 'Unnamed'}</div>
            <div className="text-xs text-neutral-500">
              {room.area_sqm ? `${room.area_sqm.toFixed(1)} m²` : '—'}
              {room.perimeter_m ? ` · ${room.perimeter_m.toFixed(1)}m` : ''}
            </div>
            {room.name === 'Unnamed' && (
              <div className="text-xs text-amber-500 mt-0.5">No label detected</div>
            )}
          </button>
        ))}
      </div>

      <div className="p-3 border-t border-neutral-800 text-xs text-neutral-500 space-y-1">
        <div>
          Scale: {scale?.px_per_meter
            ? `${(1 / scale.px_per_meter).toFixed(4)}m/px (${scale.source})`
            : 'Not set'}
        </div>
        <div>Total area: {totalArea.toFixed(1)} m²</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create room detail panel**

```tsx
// frontend/src/components/RoomDetail.tsx
import { useState, useEffect } from 'react';
import type { Room } from '../api';

interface RoomDetailProps {
  room: Room | null;
  onUpdate: (roomId: string, update: Partial<Room>) => void;
  onDelete: (roomId: string) => void;
}

export function RoomDetail({ room, onUpdate, onDelete }: RoomDetailProps) {
  const [name, setName] = useState('');
  const [roomType, setRoomType] = useState('');

  useEffect(() => {
    if (room) {
      setName(room.name);
      setRoomType(room.room_type);
    }
  }, [room?.id]);

  if (!room) {
    return (
      <div className="w-56 bg-neutral-950 border-l border-neutral-800 flex items-center justify-center">
        <p className="text-neutral-600 text-sm">Select a room</p>
      </div>
    );
  }

  const handleNameBlur = () => {
    if (name !== room.name) onUpdate(room.id, { name });
  };

  const handleTypeBlur = () => {
    if (roomType !== room.room_type) onUpdate(room.id, { room_type: roomType });
  };

  return (
    <div className="w-56 bg-neutral-950 border-l border-neutral-800 p-3 overflow-y-auto">
      <h2 className="text-xs font-bold text-sky-400 uppercase tracking-wider mb-4">
        Room Details
      </h2>

      <div className="space-y-4 text-sm">
        <div>
          <label className="block text-neutral-500 text-xs mb-1">Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={handleNameBlur}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 text-sm"
          />
        </div>

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Type</label>
          <select
            value={roomType}
            onChange={(e) => { setRoomType(e.target.value); }}
            onBlur={handleTypeBlur}
            className="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 text-sm"
          >
            {['office', 'bathroom', 'corridor', 'meeting_room', 'kitchen',
              'storage', 'lobby', 'elevator', 'stairwell', 'utility', 'other', 'unknown',
            ].map((t) => (
              <option key={t} value={t}>{t.replace('_', ' ')}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Area</label>
          <div className="text-neutral-200">
            {room.area_sqm ? `${room.area_sqm.toFixed(2)} m²` : '—'}
          </div>
        </div>

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Perimeter</label>
          <div className="text-neutral-200">
            {room.perimeter_m ? `${room.perimeter_m.toFixed(2)} m` : '—'}
          </div>
        </div>

        {room.boundary_lengths_m && (
          <div>
            <label className="block text-neutral-500 text-xs mb-1">Wall Segments</label>
            <div className="text-neutral-300 text-xs space-y-0.5">
              {room.boundary_lengths_m.map((len, i) => (
                <div key={i}>Wall {i + 1}: {len.toFixed(2)}m</div>
              ))}
            </div>
          </div>
        )}

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Vertices</label>
          <div className="text-neutral-400 text-xs">
            {room.boundary_polygon?.length || 0} points
          </div>
        </div>

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Source</label>
          <div className={`text-xs ${room.source === 'cv' ? 'text-green-400' : 'text-blue-400'}`}>
            {room.source === 'cv' ? 'Auto-detected (CV)' : room.source}
          </div>
        </div>

        <div>
          <label className="block text-neutral-500 text-xs mb-1">Confidence</label>
          <div className="text-neutral-400 text-xs">
            {(room.confidence * 100).toFixed(0)}%
          </div>
        </div>

        <button
          onClick={() => onDelete(room.id)}
          className="w-full mt-4 px-3 py-1.5 bg-red-900/30 text-red-400 text-xs rounded hover:bg-red-900/50"
        >
          Delete Room
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/RoomSidebar.tsx frontend/src/components/RoomDetail.tsx
git commit -m "feat: room sidebar and detail panel components"
```

---

### Task 14: Main App Shell

**Files:**
- Modify: `frontend/src/App.tsx`

Ties all components together: file upload, processing state, and the three-panel layout.

- [ ] **Step 1: Write App.tsx**

```tsx
// frontend/src/App.tsx
import { useState, useCallback } from 'react';
import { FloorplanCanvas } from './components/FloorplanCanvas';
import { RoomSidebar } from './components/RoomSidebar';
import { RoomDetail } from './components/RoomDetail';
import {
  processFloorplan, updateRoom, deleteRoom, getImageUrl,
  type Room, type ProcessResult,
} from './api';

type AppState = 'idle' | 'processing' | 'ready' | 'error';

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [scale, setScale] = useState<ProcessResult['scale'] | null>(null);
  const [progress, setProgress] = useState('');

  const selectedRoom = rooms.find((r) => r.id === selectedRoomId) || null;
  const imageUrl = projectId ? getImageUrl(projectId) : null;

  const handleFileUpload = useCallback(async (file: File) => {
    setState('processing');
    setError(null);
    setProgress('Uploading and processing...');

    try {
      const result = await processFloorplan(file);
      setProjectId(result.project_id);
      setRooms(result.rooms);
      setScale(result.scale);
      setState('ready');
    } catch (err: any) {
      setError(err.message || 'Processing failed');
      setState('error');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file?.type === 'application/pdf') handleFileUpload(file);
  }, [handleFileUpload]);

  const handleRoomUpdate = useCallback(async (roomId: string, update: Partial<Room>) => {
    await updateRoom(roomId, update);
    setRooms((prev) => prev.map((r) => r.id === roomId ? { ...r, ...update } : r));
  }, []);

  const handleRoomDelete = useCallback(async (roomId: string) => {
    await deleteRoom(roomId);
    setRooms((prev) => prev.filter((r) => r.id !== roomId));
    if (selectedRoomId === roomId) setSelectedRoomId(null);
  }, [selectedRoomId]);

  const handleRoomPolygonUpdate = useCallback((roomId: string, polygon: number[][]) => {
    handleRoomUpdate(roomId, { boundary_polygon: polygon });
  }, [handleRoomUpdate]);

  // Idle / Upload screen
  if (state === 'idle' || state === 'error') {
    return (
      <div
        className="h-screen bg-neutral-950 flex items-center justify-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-bold text-neutral-200">Floorplan Processor</h1>
          <p className="text-neutral-500">Drop a PDF floorplan or click to upload</p>

          <label className="inline-block px-6 py-3 bg-sky-600 text-white rounded-lg cursor-pointer hover:bg-sky-500">
            Select PDF
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </label>

          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      </div>
    );
  }

  // Processing screen
  if (state === 'processing') {
    return (
      <div className="h-screen bg-neutral-950 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full mx-auto" />
          <p className="text-neutral-400">{progress}</p>
        </div>
      </div>
    );
  }

  // Main app layout
  return (
    <div className="h-screen bg-neutral-950 flex overflow-hidden">
      <RoomSidebar
        rooms={rooms}
        selectedRoomId={selectedRoomId}
        onRoomSelect={setSelectedRoomId}
        scale={scale}
      />
      <FloorplanCanvas
        rooms={rooms}
        imageUrl={imageUrl}
        selectedRoomId={selectedRoomId}
        onRoomSelect={setSelectedRoomId}
        onRoomUpdate={handleRoomPolygonUpdate}
      />
      <RoomDetail
        room={selectedRoom}
        onUpdate={handleRoomUpdate}
        onDelete={handleRoomDelete}
      />
    </div>
  );
}
```

- [ ] **Step 2: Verify app builds**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no TypeScript errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: main app shell with upload, processing, and three-panel layout"
```

---

### Task 15: Electron Desktop Wrapper

**Files:**
- Create: `frontend/electron/main.ts`
- Modify: `frontend/package.json`

- [ ] **Step 1: Install Electron dependencies**

```bash
cd frontend
npm install --save-dev electron electron-builder concurrently wait-on
```

- [ ] **Step 2: Create Electron main process**

```typescript
// frontend/electron/main.ts
import { app, BrowserWindow } from 'electron';
import path from 'path';
import { spawn } from 'child_process';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ReturnType<typeof spawn> | null = null;

function startBackend() {
  backendProcess = spawn('python', ['-m', 'uvicorn', 'backend.main:app', '--port', '8000'], {
    cwd: path.join(__dirname, '..', '..'),
    stdio: 'pipe',
  });

  backendProcess.stdout?.on('data', (data: Buffer) => {
    console.log(`[backend] ${data}`);
  });

  backendProcess.stderr?.on('data', (data: Buffer) => {
    console.error(`[backend] ${data}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    title: 'Floorplan Processor',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // In development, load from Vite dev server
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', () => {
  startBackend();
  // Give backend a moment to start
  setTimeout(createWindow, 2000);
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  app.quit();
});
```

- [ ] **Step 3: Update package.json scripts**

Add these scripts to `frontend/package.json`:

```json
{
  "main": "electron/main.ts",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "electron:dev": "concurrently \"vite\" \"wait-on http://localhost:5173 && electron .\"",
    "electron:build": "vite build && electron-builder"
  }
}
```

- [ ] **Step 4: Test Electron launch in dev mode**

```bash
cd frontend && npm run electron:dev
```

Expected: Electron window opens with the Floorplan Processor app

- [ ] **Step 5: Commit**

```bash
git add frontend/electron/ frontend/package.json
git commit -m "feat: Electron desktop wrapper with backend process management"
```

---

## Chunk 3: Integration & Polish

### Task 16: End-to-End Integration Test

**Files:**
- Create: `backend/tests/test_integration.py`

Tests the full pipeline from PDF to room data using the actual sample PDF.

- [ ] **Step 1: Write integration test**

```python
# backend/tests/test_integration.py
"""End-to-end integration test using the sample floorplan PDF."""

import os
import pytest
import numpy as np
from backend.pipeline.extractor import extract_floorplan
from backend.pipeline.preprocessor import preprocess_image
from backend.pipeline.wall_detector import detect_walls
from backend.pipeline.room_segmenter import segment_rooms
from backend.pipeline.scale_detector import detect_scale

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "input sample.pdf")


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="Sample PDF not available")
class TestFullPipeline:
    def test_extract_preprocess_detect(self):
        # Step 1: Extract
        extraction = extract_floorplan(SAMPLE_PDF)
        assert extraction["image"].shape[0] > 1000  # large image

        # Step 2: Detect scale
        scale = detect_scale(text=extraction["text"])
        assert scale is not None
        assert scale["source"] == "auto"

        # Step 3: Pre-process
        processed = preprocess_image(extraction["image"])
        assert processed["binary"].shape == extraction["image"].shape[:2]

        # Step 4: Detect walls
        walls = detect_walls(processed["binary"])
        assert len(walls["segments"]) > 0
        print(f"Detected {len(walls['segments'])} wall segments")

        # Step 5: Segment rooms
        rooms = segment_rooms(walls["wall_mask"])
        assert len(rooms) > 0
        print(f"Detected {len(rooms)} rooms")

        # Step 6: Verify rooms have valid polygons
        for room in rooms:
            assert room["polygon"].is_valid
            assert room["area_px"] > 0

        # Step 7: Convert to real measurements
        from backend.models.room import to_real_measurements
        px_per_m = scale["px_per_meter"]
        for room in rooms:
            m = to_real_measurements(
                room["area_px"], room["perimeter_px"],
                room["boundary_lengths_px"], px_per_m,
            )
            assert m["area_sqm"] > 0
            assert m["perimeter_m"] > 0
            print(f"  Room: {m['area_sqm']:.1f} m², perimeter: {m['perimeter_m']:.1f} m")
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest backend/tests/test_integration.py -v -s
```

Expected: PASS with printed room statistics. Review the number of rooms and sizes for sanity.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "test: end-to-end pipeline integration test"
```

---

### Task 17: Run Script & Development Workflow

**Files:**
- Create: `run.sh`

A single script to start both backend and frontend for development.

- [ ] **Step 1: Create run script**

```bash
#!/bin/bash
# run.sh — Start the floorplan processor (backend + frontend)
set -e

echo "Starting Floorplan Processor..."

# Start backend
echo "Starting backend on :8000..."
cd "$(dirname "$0")"
python -m uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
```

- [ ] **Step 2: Make executable and test**

```bash
chmod +x run.sh
./run.sh
```

Expected: Both servers start. Open http://localhost:5173 and verify the upload screen appears.

- [ ] **Step 3: Commit**

```bash
git add run.sh
git commit -m "feat: development run script for backend + frontend"
```

---

### Task 18: Environment Configuration

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create environment template**

```bash
# .env.example
# Gemini Flash API key (free tier at https://aistudio.google.com/apikey)
GOOGLE_API_KEY=your_api_key_here

# Database path (default: floorplan.db in project root)
DB_PATH=floorplan.db
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: environment configuration template"
```
