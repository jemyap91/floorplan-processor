import numpy as np
import pytest
from backend.pipeline.preprocessor import preprocess_image

class TestPreprocessImage:
    def _make_test_image(self, w=200, h=200):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        img[50, 20:180] = 0
        img[150, 20:180] = 0
        img[50:150, 20] = 0
        img[50:150, 180] = 0
        return img

    def test_returns_binary_image(self):
        img = self._make_test_image()
        result = preprocess_image(img)
        assert "binary" in result
        assert result["binary"].dtype == np.uint8
        assert len(result["binary"].shape) == 2
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
        img = self._make_test_image()
        result = preprocess_image(img)
        assert result["binary"][50, 100] == 255

    def test_custom_block_size(self):
        img = self._make_test_image()
        result = preprocess_image(img, block_size=31)
        assert result["binary"] is not None

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            preprocess_image(np.array([]))
