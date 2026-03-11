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
        assert len(result["image"].shape) == 3
        assert result["image"].shape[2] == 3

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
