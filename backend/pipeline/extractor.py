"""Extract raster floorplan images from PDF or image files."""
import fitz
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_from_image(image_path: str) -> dict:
    """Extract floorplan data from a raster image file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # OpenCV loads BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    return {
        "image": img_rgb,
        "page_width": w,
        "page_height": h,
        "text": "",
        "page_count": 1,
        "image_width": w,
        "image_height": h,
    }


def extract_floorplan(pdf_path: str, page_num: int = 0) -> dict:
    """Extract the floorplan image and metadata from a PDF page."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(str(path))
    if page_num < 0 or page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (0-{len(doc)-1})")
    page = doc[page_num]
    # Try embedded images first; pick the largest one
    images = page.get_images(full=True)
    best_pix = None
    if images:
        for img_info in images:
            xref = img_info[0]
            candidate = fitz.Pixmap(doc, xref)
            if candidate.n != 3:
                candidate = fitz.Pixmap(fitz.csRGB, candidate)
            if best_pix is None or (candidate.width * candidate.height > best_pix.width * best_pix.height):
                best_pix = candidate
    # Use embedded image only if large enough (>1000px in both dims);
    # otherwise render the page at 300 DPI (needed for vector/CAD PDFs)
    if best_pix and best_pix.width >= 1000 and best_pix.height >= 1000:
        img_array = np.frombuffer(best_pix.samples, dtype=np.uint8).reshape(best_pix.height, best_pix.width, 3)
    else:
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
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
