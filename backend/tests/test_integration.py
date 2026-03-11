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
        assert extraction["image"].shape[0] > 1000

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
